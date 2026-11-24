"""Download and snapshot every external source into data/sources/, with a manifest.

Run:  .venv/bin/python -m sg2data.fetch_sources [--only idioms,wikidata,conceptnet,wiktionary]

Each source is saved as raw as practical (raw CSV, raw SPARQL JSON, raw API JSON).
data/sources/manifest.json records for every file: URL, retrieval time (UTC), sha256,
size and, for GitHub, the pinned commit. Re-running overwrites the snapshot and the
manifest entry, so run it once and commit the result.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from .common import LARGE, SOURCES, sha256_file

UA = "SemanticGravityV2-dataset/1.0 (academic research; python-requests)"
MANIFEST = SOURCES / "manifest.json"

IDIOMS_REPO = "baiango/english_idioms"
IDIOMS_COMMIT = "d47bfb40a3f76d0f08ba1867016c383d3c21c596"  # HEAD on 2026-11-25 (last commit 2024-01-18)

CONCEPTNET_API = "https://api.conceptnet.io/query?start=/c/en/scissors&rel=/r/UsedFor&limit=5"
CONCEPTNET_DUMP = "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz"
# Relations we build templates for (see build_candidates.CS_RELATIONS).
# (5.7.0 has no English HasPart edges; its PartOf edges are used inverted.)
CONCEPTNET_RELATIONS = {"UsedFor", "MadeOf", "HasProperty", "HasPart", "PartOf", "AtLocation",
                        "CapableOf", "HasPrerequisite"}

WIKTIONARY_API = "https://en.wiktionary.org/w/api.php"

# Hunspell en_US word list (SCOWL-based). Lowercase entries are common words in their
# base form; capitalised entries are proper nouns, uppercase ones abbreviations. Used to
# keep generated targets and common-sense words to ordinary dictionary words.
HUNSPELL_REPO = "LibreOffice/dictionaries"
HUNSPELL_COMMIT = "79195b017e6d70151337856c59e7d13d311d6630"  # HEAD on 2026-11-25
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"note": "Snapshots of every external source used by sg2data. Paths are relative to data/sources/.",
            "files": {}}


def record(manifest: dict, path: Path, **meta) -> None:
    rel = str(path.relative_to(SOURCES))
    manifest["files"][rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size, **meta}


def get(url: str, **kw) -> requests.Response:
    headers = {"User-Agent": UA, **kw.pop("headers", {})}
    for attempt in range(4):
        try:
            r = requests.get(url, headers=headers, timeout=kw.pop("timeout", 180), **kw)
            if r.status_code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r
        except requests.ConnectionError:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"unreachable: {url}")


# ------------------------------------------------------------------ idioms

def fetch_idioms(manifest: dict) -> None:
    out = SOURCES / "idioms"
    out.mkdir(parents=True, exist_ok=True)
    for name in ["idioms.csv", "LICENSE.md", "README.md"]:
        url = f"https://raw.githubusercontent.com/{IDIOMS_REPO}/{IDIOMS_COMMIT}/{name}"
        path = out / f"baiango_{name}"
        path.write_bytes(get(url).content)
        record(manifest, path, url=url, retrieved_at=now(),
               github_repo=IDIOMS_REPO, commit=IDIOMS_COMMIT,
               license="Unlicense" if name == "idioms.csv" else None)
    print("idioms: baiango snapshot saved")


def fetch_wiktionary_idioms(manifest: dict) -> None:
    """Supplementary idiom source: all page titles in Wiktionary 'Category:English idioms'."""
    out = SOURCES / "idioms" / "wiktionary_category_english_idioms.json"
    params = {"action": "query", "list": "categorymembers", "cmtitle": "Category:English idioms",
              "cmlimit": "500", "cmnamespace": "0", "format": "json", "formatversion": "2"}
    titles, pages = [], 0
    while True:
        data = get(WIKTIONARY_API, params=params).json()
        titles += [m["title"] for m in data["query"]["categorymembers"]]
        pages += 1
        if "continue" not in data:
            break
        params.update(data["continue"])
        time.sleep(0.5)
    payload = {"source": "Wiktionary, Category:English idioms (namespace 0 titles)",
               "api": WIKTIONARY_API, "params": {k: v for k, v in params.items() if k not in ("cmcontinue", "continue")},
               "retrieved_at": now(), "license": "CC BY-SA 4.0", "n_titles": len(titles), "titles": titles}
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=0))
    record(manifest, out, url=f"{WIKTIONARY_API}?action=query&list=categorymembers&cmtitle=Category:English_idioms",
           retrieved_at=payload["retrieved_at"], license="CC BY-SA 4.0", api_pages=pages)
    print(f"idioms: wiktionary {len(titles)} titles")


def fetch_hunspell(manifest: dict) -> None:
    out = SOURCES / "wordlists"
    out.mkdir(parents=True, exist_ok=True)
    for name in ["en/en_US.dic", "en/README_en_US.txt"]:
        url = f"https://raw.githubusercontent.com/{HUNSPELL_REPO}/{HUNSPELL_COMMIT}/{name}"
        path = out / f"hunspell_{name.split('/')[-1]}"
        path.write_bytes(get(url).content)
        record(manifest, path, url=url, retrieved_at=now(), github_repo=HUNSPELL_REPO,
               commit=HUNSPELL_COMMIT, license="SCOWL/BSD-style, see README_en_US.txt")
    print("hunspell: en_US.dic saved")


# ------------------------------------------------------------------ wikidata

# Familiarity is enforced with sitelink counts (number of Wikipedia language editions
# with an article), the standard cheap proxy for how well known an entity is.
SPARQL = {
    "country_capital": """
SELECT ?s ?sLabel ?o ?oLabel ?sl WHERE {
  ?s wdt:P31 wd:Q3624078; wdt:P36 ?o; wikibase:sitelinks ?sl.
  FILTER NOT EXISTS { ?s wdt:P576 [] }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}""",
    "country_currency": """
SELECT ?s ?sLabel ?o ?oLabel ?sl WHERE {
  ?s wdt:P31 wd:Q3624078; wdt:P38 ?o; wikibase:sitelinks ?sl.
  FILTER NOT EXISTS { ?s wdt:P576 [] }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}""",
    "country_language": """
SELECT ?s ?sLabel ?o ?oLabel ?sl WHERE {
  ?s wdt:P31 wd:Q3624078; wdt:P37 ?o; wikibase:sitelinks ?sl.
  FILTER NOT EXISTS { ?s wdt:P576 [] }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}""",
    "country_continent": """
SELECT ?s ?sLabel ?o ?oLabel ?sl WHERE {
  ?s wdt:P31 wd:Q3624078; wdt:P30 ?o; wikibase:sitelinks ?sl.
  FILTER NOT EXISTS { ?s wdt:P576 [] }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}""",
    "country_demonym": """
SELECT ?s ?sLabel ?o ?sl WHERE {
  ?s wdt:P31 wd:Q3624078; wdt:P1549 ?o; wikibase:sitelinks ?sl.
  FILTER NOT EXISTS { ?s wdt:P576 [] }
  FILTER(LANG(?o) = "en")
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}""",
    "country_highest_point": """
SELECT ?s ?sLabel ?o ?oLabel ?sl WHERE {
  ?s wdt:P31 wd:Q3624078; wdt:P610 ?o; wikibase:sitelinks ?sl.
  FILTER NOT EXISTS { ?s wdt:P576 [] }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}""",
    # Dropped after trying: city -> country (every variant timed out at the public endpoint
    # on 2026-11-25), opera -> composer (0 rows) and national sport (v1 used P241, which is
    # "military branch"; Wikidata has no national-sport property).
    # US states, Indian states, Australian states, Canadian provinces, Brazilian
    # states, German states, Mexican states.
    "region_capital": """
SELECT ?s ?sLabel ?o ?oLabel ?sl ?cls WHERE {
  VALUES ?cls { wd:Q35657 wd:Q13390680 wd:Q5852411 wd:Q11828004 wd:Q485258 wd:Q1221156 wd:Q15149663 }
  ?s wdt:P31 ?cls; wdt:P36 ?o; wikibase:sitelinks ?sl.
  FILTER NOT EXISTS { ?s wdt:P576 [] }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}""",
    # Famous people (100+ Wikipedia editions) with one of 14 headline occupations:
    # physicist, chemist, painter, composer, philosopher, mathematician, astronomer,
    # sculptor, architect, inventor, explorer, astronaut, poet, novelist. A person is
    # used only if exactly one of these applies (build_candidates).
    "person_occupation": """
SELECT ?s ?sLabel ?o ?oLabel ?sl WHERE {
  VALUES ?o { wd:Q169470 wd:Q593644 wd:Q1028181 wd:Q36834 wd:Q4964182 wd:Q170790 wd:Q11063
              wd:Q1281618 wd:Q42973 wd:Q205375 wd:Q11900058 wd:Q11631 wd:Q49757 wd:Q6625963 }
  ?s wdt:P106 ?o; wikibase:sitelinks ?sl.
  FILTER(?sl >= 100)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}""",
    "book_author": """
SELECT ?s ?sLabel ?a ?aLabel ?o ?oLabel ?sl WHERE {
  ?s wdt:P31 wd:Q7725634; wikibase:sitelinks ?sl.
  FILTER(?sl >= 40)
  ?s wdt:P50 ?a. ?a wdt:P734 ?o.
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}""",
    "painting_painter": """
SELECT ?s ?sLabel ?a ?aLabel ?o ?oLabel ?sl WHERE {
  ?s wdt:P31 wd:Q3305213; wikibase:sitelinks ?sl.
  FILTER(?sl >= 25)
  ?s wdt:P170 ?a. ?a wdt:P734 ?o.
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}""",
    "film_director": """
SELECT ?s ?sLabel ?a ?aLabel ?o ?oLabel ?sl WHERE {
  ?s wdt:P31 wd:Q11424; wikibase:sitelinks ?sl.
  FILTER(?sl >= 70)
  ?s wdt:P57 ?a. ?a wdt:P734 ?o.
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}""",
    "element_symbol": """
SELECT ?s ?sLabel ?sym ?sl WHERE {
  ?s wdt:P31 wd:Q11344; wdt:P246 ?sym; wikibase:sitelinks ?sl.
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}""",
}


def fetch_wikidata(manifest: dict, names: list[str] | None = None) -> None:
    out = SOURCES / "wikidata"
    out.mkdir(parents=True, exist_ok=True)
    for name, query in SPARQL.items():
        if names and name not in names:
            continue
        query = query.strip()
        (out / f"{name}.rq").write_text(query + "\n")
        try:
            r = get(WIKIDATA_ENDPOINT, params={"query": query, "format": "json"},
                    headers={"Accept": "application/sparql-results+json"}, timeout=120)
        except Exception as e:  # noqa: BLE001 - a failed relation is reported, not fatal
            print(f"wikidata: {name}: FAILED ({type(e).__name__}: {e})")
            continue
        path = out / f"{name}.json"
        path.write_bytes(r.content)
        n = len(r.json()["results"]["bindings"])
        record(manifest, path, url=WIKIDATA_ENDPOINT, query_file=f"wikidata/{name}.rq",
               retrieved_at=now(), rows=n, license="CC0")
        print(f"wikidata: {name}: {n} rows")
        time.sleep(1.0)


# ------------------------------------------------------------------ conceptnet

def fetch_conceptnet(manifest: dict) -> None:
    out = SOURCES / "conceptnet"
    out.mkdir(parents=True, exist_ok=True)
    api_status = None
    try:
        r = requests.get(CONCEPTNET_API, headers={"User-Agent": UA}, timeout=30)
        api_status = r.status_code
        api_ok = r.status_code == 200 and "edges" in r.json()
    except Exception as e:  # noqa: BLE001 - any failure means "API not usable"
        api_status, api_ok = f"error: {type(e).__name__}", False
    status = {"checked_at": now(), "api_url": CONCEPTNET_API, "api_status": api_status,
              "api_usable": api_ok,
              "decision": "Use the official ConceptNet 5.7.0 assertions dump (S3) in all cases: "
                          "it is the same data the API serves, is a fixed versioned file, and "
                          "makes the build reproducible."}
    (out / "api_check.json").write_text(json.dumps(status, indent=2))
    print(f"conceptnet: API status {api_status} usable={api_ok}")

    LARGE.mkdir(parents=True, exist_ok=True)
    dump = LARGE / "conceptnet-assertions-5.7.0.csv.gz"
    if not dump.exists():
        print("conceptnet: downloading dump (~475 MB)")
        with requests.get(CONCEPTNET_DUMP, headers={"User-Agent": UA}, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(dump, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        dump_time = now()
    else:
        dump_time = manifest["files"].get("large/" + dump.name, {}).get("retrieved_at") or \
            datetime.fromtimestamp(dump.stat().st_mtime, timezone.utc).replace(microsecond=0).isoformat()
    record(manifest, dump, url=CONCEPTNET_DUMP, retrieved_at=dump_time, license="CC BY-SA 4.0",
           committed=False, note="Raw dump kept out of git (over 20 MB); the filtered subset below is committed.")

    # English-English edges of the relations we use, as compact TSV (gzipped).
    sub = out / "conceptnet_5.7.0_en_subset.tsv.gz"
    n = 0
    with gzip.open(dump, "rt", encoding="utf-8") as fin, gzip.open(sub, "wt", encoding="utf-8", compresslevel=9) as fout:
        fout.write("relation\tstart\tend\tweight\tdataset\tsurface\n")
        for line in fin:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 5:
                continue
            _, rel, start, end, info = parts
            rel = rel.removeprefix("/r/")
            if rel not in CONCEPTNET_RELATIONS or not start.startswith("/c/en/") or not end.startswith("/c/en/"):
                continue
            meta = json.loads(info)
            surface = (meta.get("surfaceText") or "").replace("\t", " ")
            fout.write(f"{rel}\t{start}\t{end}\t{meta.get('weight', 1.0)}\t{meta.get('dataset', '')}\t{surface}\n")
            n += 1
    record(manifest, sub, url=CONCEPTNET_DUMP, retrieved_at=dump_time, license="CC BY-SA 4.0",
           derived_from="large/" + dump.name, rows=n,
           filter=f"relation in {sorted(CONCEPTNET_RELATIONS)}; both ends /c/en/")
    print(f"conceptnet: subset {n} edges")


def record_wordfreq(manifest: dict) -> None:
    from importlib.metadata import version
    manifest["packages"] = {"wordfreq": version("wordfreq"),
                            "note": "wordfreq ships its frequency data inside the package; the version pins it."}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="idioms,wiktionary,hunspell,wikidata,conceptnet")
    ap.add_argument("--wikidata-relations", default="", help="comma list; default all")
    args = ap.parse_args(argv)
    SOURCES.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    steps = {"idioms": fetch_idioms, "wiktionary": fetch_wiktionary_idioms, "hunspell": fetch_hunspell,
             "wikidata": lambda m: fetch_wikidata(m, [x for x in args.wikidata_relations.split(",") if x]),
             "conceptnet": fetch_conceptnet}
    for name in args.only.split(","):
        steps[name](manifest)
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    record_wordfreq(manifest)
    manifest["files"] = {k: v for k, v in manifest["files"].items() if (SOURCES / k).exists()}
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
