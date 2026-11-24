"""Build candidate prompts for idioms, facts and common sense from the snapshots, and
choose the target words for creative and OOD.

Run:  .venv/bin/python -m sg2data.build_candidates

Deterministic (seed 42). Reads only data/sources/; writes
  data/candidates/{idioms,facts,common_sense}.jsonl
  data/candidates/targets_{creative,ood}.json
  data/candidates/build_report.json   (counts, ceilings, rejection reasons)
"""
from __future__ import annotations

import csv
import gzip
import json
import random
import re
from collections import Counter, defaultdict

from wordfreq import top_n_list, zipf_frequency

from .common import (BLANK, CANDIDATES, SEED, SOURCES, ZIPF_MAX, ZIPF_MIN, code_filter,
                     prompt_id, write_jsonl)

TARGET_POOL = 1500          # candidates wanted per source category (first build)
IDIOM_EXTENSION = 1200      # second round: more Wiktionary idioms after the first judging
N_CS_TARGETS = 750          # common sense, generated (second round)
PER_TARGET_CAP = 3          # same answer at most 3 times inside one category (facts: 5)
FACTS_PER_TARGET_CAP = 5
N_GEN_TARGETS = 750         # target words per generated category (creative, ood)

# Function words make poor cloze answers ("the", "which"); excluded from generated targets.
STOPWORDS = set("""
a about above after again against all almost alone along already also although always am
among an and another any anybody anyone anything anyway anywhere are around as at away back
be became because become becomes been before behind being below beside besides between beyond
both but by can cannot could did do does doing done down during each either else enough even
ever every everybody everyone everything everywhere except few for from further get gets getting
got gotten had has have having he her here hers herself him himself his how however i if in
indeed inside instead into is it its itself just least less let like many may maybe me might
mine more most mostly much must my myself near neither never nevertheless next no nobody none
nor not nothing now nowhere of off often on once one ones only onto or other others otherwise
ought our ours ourselves out outside over own per perhaps please quite rather really same
shall she should since so some somebody someone something sometimes somewhat somewhere still
such than that the their theirs them themselves then there therefore these they this those
though through throughout thus to together too toward towards under unless until up upon us
very via was we well were what whatever when whenever where whereas wherever whether which
while who whoever whole whom whose why will with within without would yet you your yours
yourself yourselves yeah yes okay ok gonna wanna gotta lol hey oh ah um uh shall cannot
""".split())

PLACEHOLDERS = {"someone", "someone's", "one's", "oneself", "something", "somebody", "sb", "sth",
                "somewhere", "himself", "herself", "yourself", "themselves", "x", "y"}


def load_lexicon() -> set[str]:
    """Lowercase base-form entries of the Hunspell en_US list (no names, no abbreviations)."""
    words = set()
    with open(SOURCES / "wordlists" / "hunspell_en_US.dic", encoding="utf-8") as f:
        next(f)  # first line is the entry count
        for line in f:
            w = line.split("/", 1)[0].strip()
            if re.fullmatch(r"[a-z]+", w):
                words.add(w)
    return words


LEX = load_lexicon()


def rng_for(name: str) -> random.Random:
    return random.Random(f"{SEED}:{name}")


def make(category: str, question: str, target: str, template_id: str, trace: dict) -> dict:
    return {"prompt_id": prompt_id(category, question, target), "category": category,
            "question": question, "target": target, "template_id": template_id, "source_trace": trace}


def cap_per_target(rows: list[dict], cap: int = PER_TARGET_CAP) -> list[dict]:
    seen: Counter = Counter()
    out = []
    for r in rows:
        t = r["target"].lower()
        if seen[t] < cap:
            seen[t] += 1
            out.append(r)
    return out


def filter_rows(rows: list[dict], rejects: Counter) -> list[dict]:
    out, seen_q = [], set()
    for r in rows:
        reason = code_filter(r["question"], r["target"])
        if reason:
            rejects[reason.split(":")[0] if reason.startswith("leak") else reason] += 1
            continue
        key = (r["question"].lower(), r["target"].lower())
        if key in seen_q:
            rejects["duplicate"] += 1
            continue
        seen_q.add(key)
        out.append(r)
    return out


# ================================================================== idioms

IDIOM_TEMPLATES = {
    "I1": 'Complete the idiom with one word: "{x}"',
    "I2": "Fill the blank (one word): {x}",
    "I3": "Idiom completion (one word): {x}",
}


def idiom_with_blank(idiom: str) -> tuple[str, str] | None:
    """Strip terminal punctuation, blank the final word. Returns (with_blank, target)."""
    idiom = " ".join(idiom.strip().split())
    idiom = idiom.rstrip(".,!?;:\"'")
    tokens = idiom.split(" ")
    if len(tokens) < 3:
        return None
    if any(c in idiom for c in "[]{}()_/") or "__" in idiom:
        return None
    last = tokens[-1]
    if not re.fullmatch(r"[A-Za-z]+", last):
        return None
    return " ".join(tokens[:-1] + [BLANK]), last


def load_baiango() -> list[tuple[str, str]]:
    rows = []
    with open(SOURCES / "idioms" / "baiango_idioms.csv", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            m = re.match(r"\{([^}]+)\}>>\{([^}]*)\}", row[0].strip())
            if m:
                rows.append((m.group(1).strip(), m.group(2).strip()))
    return rows


def wiktionary_ok(title: str) -> bool:
    words = title.split()
    if not (3 <= len(words) <= 8) or not all(ord(c) < 128 for c in title):
        return False
    low = [w.lower().strip(",.!?;:'\"") for w in words]
    if any(w in PLACEHOLDERS for w in low) or any("'s" in w and w.startswith(("one", "some")) for w in low):
        return False
    last = low[-1]
    if last in STOPWORDS or zipf_frequency(last, "en") < 3.0:
        return False
    # every word reasonably common, so the idiom is plausibly familiar
    return all(zipf_frequency(w, "en") >= 2.5 for w in low if w.isalpha())


def build_idioms(report: dict, pool: int = TARGET_POOL) -> list[dict]:
    rng = rng_for("idioms")
    rejects: Counter = Counter()
    rows, seen_idioms = [], set()

    def add(idiom: str, source: dict):
        key = idiom.lower().rstrip(".,!?;:")
        if key in seen_idioms:
            rejects["duplicate_idiom"] += 1
            return
        seen_idioms.add(key)
        parsed = idiom_with_blank(idiom)
        if not parsed:
            rejects["malformed_or_short"] += 1
            return
        blank, target = parsed
        if target.lower() in STOPWORDS:
            rejects["function_word_target"] += 1
            return
        tid = rng.choice(sorted(IDIOM_TEMPLATES))
        rows.append(make("idioms", IDIOM_TEMPLATES[tid].format(x=blank), target, tid, {**source, "idiom": idiom}))

    baiango = load_baiango()
    for idiom, meaning in baiango:
        add(idiom, {"source": "baiango/english_idioms", "file": "idioms/baiango_idioms.csv", "meaning": meaning})
    n_baiango = len(rows)

    wik = json.loads((SOURCES / "idioms" / "wiktionary_category_english_idioms.json").read_text())["titles"]
    wik_ok = [t for t in wik if wiktionary_ok(t)]
    rng.shuffle(wik_ok)
    for t in wik_ok:
        add(t, {"source": "wiktionary:Category:English_idioms",
                "file": "idioms/wiktionary_category_english_idioms.json"})

    rows = filter_rows(rows, rejects)
    baiango_rows = [r for r in rows if r["source_trace"]["source"] == "baiango/english_idioms"]
    wik_rows = [r for r in rows if r["source_trace"]["source"] != "baiango/english_idioms"]
    rows = cap_per_target(baiango_rows + wik_rows)
    n_b = sum(r["source_trace"]["source"] == "baiango/english_idioms" for r in rows)
    wik_kept = [r for r in rows if r["source_trace"]["source"] != "baiango/english_idioms"]
    out = [r for r in rows if r["source_trace"]["source"] == "baiango/english_idioms"] + \
        wik_kept[: max(0, pool - n_b)]
    report["idioms"] = {"baiango_rows": len(baiango), "baiango_parsed": n_baiango,
                        "baiango_after_filters": n_b, "wiktionary_titles": len(wik),
                        "wiktionary_pass_prefilter": len(wik_ok),
                        "wiktionary_after_filters": len(wik_kept),
                        "wiktionary_used": len(out) - n_b, "candidates": len(out),
                        "rejections": dict(rejects)}
    return out


# ================================================================== facts

FACT_TEMPLATES = {
    "F1": "Fill the blank with one word: The capital of {s} is ____.",
    "F2": "One-word answer: {s}'s capital is ____.",
    "F3": "Complete with one word: The currency of {s} is ____.",
    "F4": "One-word answer: An official language of {s} is ____.",
    "F5": "Fill the blank (one word): {s} is in ____.",
    "F6": "Fill the blank (one word): The highest mountain in {s} is ____.",
    "F12": "Fill the blank (one word): A one-word occupation associated with {s} is ____.",
    "F13": "Fill the blank (one word): A demonym for {s} is ____.",
    # v2 additions (same style), for well-known works and elements:
    "F16": "Fill the blank (one word): The book {s} was written by ____.",
    "F17": "Fill the blank (one word): The painting {s} was painted by ____.",
    "F18": "Fill the blank (one word): The film {s} was directed by ____.",
    "F19": "Fill the blank (one word): The chemical element with the symbol {s} is ____.",
}

FACT_RELATIONS = {
    # relation: (templates, target transform)
    "country_capital": ["F1", "F2"],
    "region_capital": ["F1", "F2"],
    "country_currency": ["F3"],
    "country_language": ["F4"],
    "country_continent": ["F5"],
    "country_highest_point": ["F6"],
    "person_occupation": ["F12"],
    "country_demonym": ["F13"],
    "book_author": ["F16"],
    "painting_painter": ["F17"],
    "film_director": ["F18"],
    "element_symbol": ["F19"],
}
PER_RELATION_CAP = 250
# A subject may have this many answers in Wikidata in total, of which exactly one is a
# usable single word (e.g. a person who is "physicist" and "university teacher").
MAX_RAW_ANSWERS = {"country_language": 2}


def _val(b: dict, k: str) -> str:
    return b.get(k, {}).get("value", "")


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def fact_target(relation: str, label: str) -> str | None:
    label = label.strip()
    if relation == "country_currency":
        words = label.split()
        if words and all(re.fullmatch(r"[A-Za-z]+", w) for w in words):
            return words[-1]           # "Japanese yen" -> "yen"
        return None
    if relation == "country_highest_point":
        label = re.sub(r"^(Mount|Mt\.?)\s+", "", label)
    if relation == "element_symbol":
        label = label.lower()
        if zipf_frequency(label, "en") < 2.5:   # drops obscure and placeholder names (unquadoctium)
            return None
    return label if re.fullmatch(r"[A-Za-z]+", label) else None


def subject_ok(s: str) -> bool:
    return bool(s) and all(ord(c) < 128 for c in s) and not re.fullmatch(r"Q\d+", s) and len(s) <= 80


def build_facts(report: dict) -> list[dict]:
    rng = rng_for("facts")
    rejects: Counter = Counter()
    per_rel: dict[str, list[dict]] = {}
    rel_stats = {}
    country_labels = {_val(b, "sLabel") for b in json.loads(
        (SOURCES / "wikidata" / "country_capital.json").read_text())["results"]["bindings"]}

    for relation, tids in FACT_RELATIONS.items():
        path = SOURCES / "wikidata" / f"{relation}.json"
        if not path.exists():
            rel_stats[relation] = {"rows": 0, "note": "snapshot missing"}
            continue
        bindings = json.loads(path.read_text())["results"]["bindings"]
        # group all answers per subject; keep subjects with exactly one usable answer
        by_subj: dict[str, dict] = {}
        for b in bindings:
            s_uri = _val(b, "s")
            s_label = _val(b, "sLabel")
            raw = _val(b, "sym") if relation == "element_symbol" else (_val(b, "oLabel") or _val(b, "o"))
            if relation == "element_symbol":
                s_label, raw = raw, _val(b, "sLabel")   # subject is the symbol, answer the name
            d = by_subj.setdefault(s_uri, {"label": s_label, "answers": set(), "raw": set(),
                                           "sl": int(_val(b, "sl") or 0), "creators": set()})
            d["raw"].add(raw)
            if _val(b, "a"):
                d["creators"].add(_val(b, "a"))
            t = fact_target(relation, raw)
            if t:
                d["answers"].add(t)
        items = []
        for s_uri, d in by_subj.items():
            if relation == "region_capital" and d["label"] in country_labels:
                rejects["region_name_is_country"] += 1
                continue
            if relation == "country_demonym":
                # drop plural forms when the singular is present ("Kenyans" vs "Kenyan")
                d["answers"] = {a for a in d["answers"] if not (a.endswith("s") and a[:-1] in d["answers"])}
            max_raw = MAX_RAW_ANSWERS.get(relation, 1)
            if len(d["creators"]) > 1 or len(d["answers"]) != 1 or len(d["raw"]) > max_raw:
                rejects["not_single_answer"] += 1
                continue
            if not subject_ok(d["label"]):
                rejects["bad_subject"] += 1
                continue
            target = next(iter(d["answers"]))
            items.append((d["sl"], s_uri, d["label"], target))
        items.sort(key=lambda x: (-x[0], x[1]))   # most familiar (sitelinks) first
        rows = []
        for sl, s_uri, s_label, target in items:
            tid = tids[0] if len(tids) == 1 else rng.choice(tids)
            q = FACT_TEMPLATES[tid].format(s=s_label)
            rows.append(make("facts", q, target, tid, {
                "source": "wikidata", "relation": relation, "subject_qid": _qid(s_uri),
                "subject": s_label, "sitelinks": sl, "file": f"wikidata/{relation}.json"}))
        rows = cap_per_target(filter_rows(rows, rejects), FACTS_PER_TARGET_CAP)
        rel_stats[relation] = {"subjects": len(by_subj), "usable": len(rows)}
        per_rel[relation] = rows[:PER_RELATION_CAP]
        rel_stats[relation]["used"] = len(per_rel[relation])

    out = cap_per_target([r for rows in per_rel.values() for r in rows], FACTS_PER_TARGET_CAP)
    report["facts"] = {"relations": rel_stats, "candidates": len(out),
                       "ceiling_before_caps": sum(v.get("usable", 0) for v in rel_stats.values()),
                       "rejections": dict(rejects)}
    return out


# ================================================================== common sense

CS_TEMPLATES = {
    "UsedFor": ("C1_UsedFor", "Fill the blank (one word): You use {a} {s} to ____."),
    "MadeOf": ("C2_MadeOf", "Fill the blank (one word): {A} {s} is made of ____."),
    "HasProperty": ("C3_HasProperty", "Fill the blank (one word): A typical {s} is ____."),
    # HasPart (C4, from inverted PartOf) was tried and dropped: its edges are WordNet part
    # lists ("A city has a ____" -> "uptown"), almost never one clear answer.
    "AtLocation": ("C5_AtLocation", "Fill the blank (one word): You usually find {a} {s} in the ____."),
    "CapableOf": ("C6_CapableOf", "Fill the blank (one word): {A} {s} can ____."),
    # ConceptNet HasPrerequisite subjects are activities ("skiing"), so no article.
    "HasPrerequisite": ("C8_Requires", "Fill the blank (one word): {S} requires ____."),
}
CS_MIN_TOP_WEIGHT = 1.0     # any asserted answer; stronger ones are used first
CS_MARGIN = 1.5             # and clearly ahead of the runner-up
SUBJECT_MAX_USES = 2


def article(word: str) -> str:
    w = word.lower()
    if w.startswith(("hour", "honest", "honor", "heir")):
        return "an"
    if w.startswith(("uni", "use", "usu", "eu", "one", "once")):
        return "a"
    return "an" if w[0] in "aeiou" else "a"


# ConceptNet contains crowd-sourced stereotypes about groups of people. Any subject or
# answer naming a religious, ethnic, national, gender or sexual-orientation group is dropped.
BLOCKED_TERMS = set("""
jew jews jewish muslim muslims islam islamic christian christians catholic catholics hindu hindus
buddhist sikh mormon atheist arab arabs black blacks white whites asian asians african africans
mexican mexicans chinese japanese indian indians irish italian italians german germans french
russian russians american americans gay gays lesbian lesbians homosexual transgender woman women
man men girl girls boy boys wife husband female male negro nigger gypsy gypsies immigrant
immigrants refugee refugees nazi nazis terrorist terrorists slave slaves
""".split())


def is_plural(word: str, z) -> bool:
    if word.endswith("es") and z(word[:-2]) >= 3.0 and z(word[:-2]) >= z(word):
        return True
    if word.endswith("ies") and z(word[:-3] + "y") >= 3.0:
        return True
    if word.endswith("s") and not word.endswith("ss") and z(word[:-1]) >= 3.0 and z(word[:-1]) >= z(word):
        return True
    return False


def concept_term(uri: str) -> str:
    return uri.split("/")[3].replace("_", " ")


def build_common_sense(report: dict) -> list[dict]:
    rng = rng_for("common_sense")
    rejects: Counter = Counter()
    groups: dict[tuple[str, str], Counter] = defaultdict(Counter)
    with gzip.open(SOURCES / "conceptnet" / "conceptnet_5.7.0_en_subset.tsv.gz", "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            rel, start, end, weight, _dataset, _surface = line.rstrip("\n").split("\t")
            s, t = concept_term(start), concept_term(end)
            if rel == "PartOf":            # PartOf(wheel, car) == HasPart(car, wheel)
                rel, s, t = "HasPart", t, s
            groups[(rel, s)][t] += float(weight)

    zipf_cache: dict[str, float] = {}

    def z(w: str) -> float:
        if w not in zipf_cache:
            zipf_cache[w] = zipf_frequency(w, "en")
        return zipf_cache[w]

    by_rel: dict[str, list[tuple]] = defaultdict(list)
    for (rel, s), answers in groups.items():
        if rel not in CS_TEMPLATES:
            continue
        # single common word; activities ("skiing") only for HasPrerequisite, objects
        # (not activities, not plurals) for the article-based templates
        if not re.fullmatch(r"[a-z]{3,}", s) or z(s) < 3.0:
            continue
        if (rel == "HasPrerequisite") != s.endswith("ing"):
            rejects["subject_form"] += 1
            continue
        if rel != "HasPrerequisite" and (s not in LEX or is_plural(s, z)):
            rejects["subject_not_singular_dictionary_word"] += 1
            continue
        if rel == "HasPrerequisite" and not (s[:-3] in LEX or s[:-3] + "e" in LEX or s[:-4] in LEX):
            rejects["subject_not_dictionary_activity"] += 1
            continue
        if s in BLOCKED_TERMS:
            rejects["blocked_term"] += 1
            continue
        # rank only answers that can be a one-word target (multi-word ConceptNet ends dropped)
        # verbs after "to ____" / "can ____" must be base forms; nouns may be plural
        def target_ok(t: str) -> bool:
            if rel in ("UsedFor", "CapableOf"):
                return t in LEX and not t.endswith(("ing", "s", "tion", "sion", "ment", "ness", "ity", "ance", "ence"))
            if rel == "HasPart":           # "has a ____" needs a singular noun
                return t in LEX and not is_plural(t, z)
            return t in LEX or (t.endswith("s") and t[:-1] in LEX) or (t.endswith("es") and t[:-2] in LEX)
        ranked = [(t, w) for t, w in answers.most_common()
                  if re.fullmatch(r"[a-z]+", t) and t not in STOPWORDS and t != s
                  and t not in BLOCKED_TERMS and z(t) >= 3.0 and target_ok(t)]
        if not ranked:
            rejects["no_common_one_word_answer"] += 1
            continue
        top_t, top_w = ranked[0]
        second_w = ranked[1][1] if len(ranked) > 1 else 0.0
        if top_w < CS_MIN_TOP_WEIGHT:
            rejects["weak_top_answer"] += 1
            continue
        if second_w and top_w < CS_MARGIN * second_w:
            rejects["no_clear_top_answer"] += 1
            continue
        by_rel[rel].append((top_w, z(s), s, top_t, second_w, len(ranked)))

    pools = {}
    for rel, items in by_rel.items():
        rng.shuffle(items)
        items.sort(key=lambda x: (-x[0], -x[1]))          # strongest consensus, then commoner subject
        rows = []
        tid, tmpl = CS_TEMPLATES[rel]
        for top_w, _zs, s, t, second_w, n_ans in items:
            a = article(s)
            q = tmpl.format(s=s, a=a, A=a.capitalize(), S=s.capitalize())
            rows.append(make("common_sense", q, t, tid, {
                "source": "conceptnet-5.7.0", "relation": rel, "subject": s,
                "edge_weight": round(top_w, 3), "runner_up_weight": round(second_w, 3),
                "n_answers_in_conceptnet": n_ans, "file": "conceptnet/conceptnet_5.7.0_en_subset.tsv.gz"}))
        pools[rel] = filter_rows(rows, rejects)

    # round-robin across relations for balance; cap subject and target reuse
    out, subj_uses, tgt_uses = [], Counter(), Counter()
    idx = {rel: 0 for rel in pools}
    rels = sorted(pools)
    while len(out) < TARGET_POOL and any(idx[r] < len(pools[r]) for r in rels):
        for rel in rels:
            while idx[rel] < len(pools[rel]):
                r = pools[rel][idx[rel]]
                idx[rel] += 1
                s, t = r["source_trace"]["subject"], r["target"]
                if subj_uses[s] >= SUBJECT_MAX_USES or tgt_uses[t] >= PER_TARGET_CAP:
                    continue
                subj_uses[s] += 1
                tgt_uses[t] += 1
                out.append(r)
                break
            if len(out) >= TARGET_POOL:
                break
    report["common_sense"] = {
        "pool_per_relation_after_filters": {r: len(v) for r, v in pools.items()},
        "ceiling_before_caps": sum(len(v) for v in pools.values()),
        "used_per_relation": dict(Counter(r["source_trace"]["relation"] for r in out)),
        "candidates": len(out), "rejections": dict(rejects),
        "rules": {"min_top_weight": CS_MIN_TOP_WEIGHT, "margin_over_runner_up": CS_MARGIN,
                  "subject_max_uses": SUBJECT_MAX_USES, "per_target_cap": PER_TARGET_CAP}}
    return out


# ================================================================== global rule + targets

def enforce_two_categories(cats: dict[str, list[dict]], report: dict) -> None:
    """A target word may appear in at most 2 categories. For a word in all three source
    categories, drop it from the category where it has the fewest candidates (ties: the
    later category in the order idioms, facts, common_sense)."""
    order = ["idioms", "facts", "common_sense"]
    counts = {c: Counter(r["target"].lower() for r in cats[c]) for c in order}
    dropped = Counter()
    for w in set().union(*[set(c) for c in counts.values()]):
        present = [c for c in order if counts[c][w]]
        while len(present) > 2:
            victim = min(present, key=lambda c: (counts[c][w], -order.index(c)))
            cats[victim] = [r for r in cats[victim] if r["target"].lower() != w]
            dropped[victim] += counts[victim][w]
            present.remove(victim)
    report["two_category_rule_dropped"] = dict(dropped)


def choose_gen_targets(cats: dict[str, list[dict]], report: dict) -> dict[str, list[str]]:
    in_cats = Counter()
    for c, rows in cats.items():
        for w in {r["target"].lower() for r in rows}:
            in_cats[w] += 1
    pool = []
    for w in top_n_list("en", 60000):
        if not re.fullmatch(r"[a-z]+", w) or len(w) < 3 or w in STOPWORDS or w not in LEX:
            continue
        zf = zipf_frequency(w, "en")
        if ZIPF_MIN <= zf <= ZIPF_MAX and in_cats[w] < 2:
            pool.append(w)
    pool = sorted(set(pool))
    rng_for("gen_targets").shuffle(pool)
    out = {"creative": sorted(pool[:N_GEN_TARGETS]), "ood": sorted(pool[N_GEN_TARGETS:2 * N_GEN_TARGETS])}
    report["generated_targets"] = {"eligible_pool": len(pool), "per_category": N_GEN_TARGETS,
                                   "rule": "wordfreq 'en' Zipf 3.5-6.0, [a-z]+, len>=3, not a function word, lowercase base entry in Hunspell en_US, "
                                           "in <2 source categories; seeded shuffle, disjoint creative/ood"}
    return out


def extend_idioms(base: list[dict], facts: list[dict], gen_targets: dict[str, list[str]],
                  report: dict) -> list[dict]:
    """Second round: the next IDIOM_EXTENSION Wiktionary idioms in the same seeded order.
    build_idioms with a larger pool returns the base rows as a prefix, so the first-round
    candidates (and their judge units) are unchanged. A new idiom is dropped if its target
    is already in two other categories."""
    full = build_idioms({}, pool=TARGET_POOL + 10 * IDIOM_EXTENSION)
    base_ids = {r["prompt_id"] for r in base}
    other = Counter()
    for words in [{r["target"].lower() for r in facts}, set(gen_targets["creative"]), set(gen_targets["ood"])]:
        for w in words:
            other[w] += 1
    ext, dropped = [], 0
    seen_t = Counter(r["target"].lower() for r in base)
    for r in full:
        if r["prompt_id"] in base_ids:
            continue
        t = r["target"].lower()
        if other[t] >= 2 or seen_t[t] >= PER_TARGET_CAP:
            dropped += 1
            continue
        seen_t[t] += 1
        ext.append(r)
        if len(ext) == IDIOM_EXTENSION:
            break
    report["idioms_extension"] = {"added": len(ext), "dropped_two_category_or_cap": dropped}
    return ext


def choose_common_sense_targets(cats: dict[str, list[dict]], gen_targets: dict[str, list[str]],
                                report: dict) -> list[str]:
    """Target words for generated common sense. Everyday concreteness is approximated by
    ConceptNet: a word must appear in at least CS_MIN_EDGES English edges of the snapshot's
    common-sense relations (things used for, made of, found at, capable of ...). Same
    wordfreq band and Hunspell base-form rule as creative/ood; disjoint from the creative
    and ood targets; not a word already in two categories."""
    edges = Counter()
    with gzip.open(SOURCES / "conceptnet" / "conceptnet_5.7.0_en_subset.tsv.gz", "rt", encoding="utf-8") as f:
        next(f)
        for line in f:
            _rel, start, end, *_ = line.split("\t")
            for u in (start, end):
                edges[concept_term(u)] += 1
    in_cats = Counter()
    for rows in cats.values():
        for w in {r["target"].lower() for r in rows}:
            in_cats[w] += 1
    used = set(gen_targets["creative"]) | set(gen_targets["ood"])
    pool = sorted(w for w, n in edges.items()
                  if n >= CS_MIN_EDGES and re.fullmatch(r"[a-z]{3,}", w) and w in LEX
                  and w not in STOPWORDS and w not in BLOCKED_TERMS and w not in used
                  and in_cats[w] < 2 and ZIPF_MIN <= zipf_frequency(w, "en") <= ZIPF_MAX)
    rng_for("cs_targets").shuffle(pool)
    report["common_sense_targets"] = {"eligible_pool": len(pool), "chosen": N_CS_TARGETS,
                                      "min_conceptnet_edges": CS_MIN_EDGES}
    return sorted(pool[:N_CS_TARGETS])


CS_MIN_EDGES = 10


def write_targets(category: str, words: list[str]) -> None:
    from importlib.metadata import version
    (CANDIDATES / f"targets_{category}.json").write_text(json.dumps(
        {"category": category, "seed": SEED, "zipf_band": [ZIPF_MIN, ZIPF_MAX], "wordfreq": version("wordfreq"),
         "n": len(words), "words": words}, indent=1))
    print(f"targets_{category}: {len(words)} words")


def main() -> None:
    """Round 1 (unchanged): idioms (1,500), facts, ConceptNet common sense, creative/ood
    targets. Round 2, after the first judging: ConceptNet common sense is replaced by
    generation (its candidates are kept only as a record), and 1,200 more idioms are added.
    Round-1 outputs are reproduced byte for byte, so existing units stay valid."""
    report: dict = {"seed": SEED}
    cats = {"idioms": build_idioms(report), "facts": build_facts(report),
            "common_sense": build_common_sense(report)}
    enforce_two_categories(cats, report)
    gen_targets = choose_gen_targets(cats, report)
    for c, words in gen_targets.items():
        write_targets(c, words)

    # round 2
    cs_conceptnet = cats.pop("common_sense")
    write_jsonl(CANDIDATES / "common_sense_conceptnet.jsonl", cs_conceptnet)
    report["common_sense"]["written"] = len(cs_conceptnet)
    report["common_sense"]["note"] = ("round 1 only; judge accepted 46 of 1,433, so common sense "
                                      "is generated instead (common_sense_conceptnet.jsonl is a record)")
    cats["idioms"] = cats["idioms"] + extend_idioms(cats["idioms"], cats["facts"], gen_targets, report)
    for c, rows in cats.items():
        n = write_jsonl(CANDIDATES / f"{c}.jsonl", rows)
        report[c]["written"] = n
        print(f"{c}: {n} candidates")
    write_targets("common_sense", choose_common_sense_targets(cats, gen_targets, report))
    (CANDIDATES / "build_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
