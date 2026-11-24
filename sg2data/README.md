# sg2data: the v2 dataset pipeline

Builds the candidate prompts for the five categories in DESIGN_v2.md ("Data"). Everything
is deterministic (seed 42) and reads only the snapshots in `data/sources/`, so the
dataset can be rebuilt from the repo alone. Nothing here calls a paid API: the LLM work
(generation for creative/OOD, judging for all) is written out as self-contained unit files
for any backend to run.

## Commands

```bash
PY=.venv/bin/python            # needs: pip install -r sg2data/requirements.txt

$PY -m sg2data.fetch_sources                   # 1. snapshot sources (done; committed)
$PY -m sg2data.build_candidates                # 2. idioms/facts/common_sense + target words
$PY -m sg2data.llm_tasks gen                   # 3. 2,250 generation units (creative, ood, common_sense)
#    run every data/llm_units/gen/*.json  ->  data/llm_outputs/gen/<unit_id>.json
$PY -m sg2data.llm_collect gen --model NAME    # 4. -> data/candidates/{creative,ood}.jsonl
$PY -m sg2data.llm_tasks judge                 # 5. judge units, <= 25 items each
#    run every data/llm_units/judge/*.json -> data/llm_outputs/judge/<unit_id>.json
$PY -m sg2data.llm_collect judge --model NAME  # 6. -> data/judged/{category}.jsonl
```

Collecting writes `data/llm_outputs/{gen,judge}/rerun.txt` listing every unit whose
output is missing, not JSON, off-schema or semantically wrong (wrong item count, wrong
ids, scores out of range). Rerun those and collect again; collecting is idempotent.
Rebuilding candidates changes nothing if the sources are unchanged, but if candidates
do change, rebuild the judge units too (stale units are flagged).

### Running a unit

Each unit has `prompt` (the full text to send) and `output_schema` (JSON Schema using only
keywords strict structured outputs accept). Save the model's JSON reply as the output file.

Codex CLI (run from an empty directory, so the agent cannot read the repo; judge units
carry no target words anyway):

```bash
u=data/llm_units/judge/unit_facts_0001.json
jq .output_schema $u > /tmp/schema.json
(cd "$(mktemp -d)" && codex exec --skip-git-repo-check --output-schema /tmp/schema.json \
   -o "$OLDPWD/data/llm_outputs/judge/$(basename $u)" "$(jq -r .prompt "$OLDPWD/$u")")
```

OpenAI API: send `prompt` as the user message with
`response_format={"type": "json_schema", "json_schema": {"name": "unit", "strict": true, "schema": <output_schema>}}`
and save `choices[0].message.content`.

## Outputs

`data/candidates/{category}.jsonl`, one candidate per line:
`prompt_id` (v1 scheme: `{category}_{target}_{sha256(question||target||category)[:8]}`),
`category`, `question`, `target`, `template_id`, `source_trace` (source file, ids,
relation, edge weights, unit id, ...). `data/judged/{category}.jsonl` adds `judge`
(best answer, top3, `is_X_best`, ambiguity, leaks, naturalness, offensive) and `accepted`.
`data/candidates/build_report.json` has every count and rejection reason.

## Sources and ceilings (snapshots of 2026-11-25)

| Category | Source | Candidates | Ceiling |
|---|---|---|---|
| idioms | baiango/english_idioms @ d47bfb4 (1,024 idioms) + Wiktionary "Category:English idioms" (10,757 titles) | 1,500 (740 baiango + 760 Wiktionary) | 740 + 2,563 = 3,303 |
| facts | Wikidata SPARQL, 12 relations, raw JSON in `data/sources/wikidata/` | 1,079 | 1,082 |
| common_sense | ConceptNet 5.7.0 assertions dump (S3); English subset committed | 1,458 | 2,390 |
| creative | 750 target words x K=5 (LLM) | up to 3,750 | - |
| ood | 750 target words x K=5 (LLM) | up to 3,750 | - |

`data/sources/manifest.json` records URL, retrieval time, sha256, size and (GitHub)
commit for every file. The raw ConceptNet dump (475 MB) is git-ignored in
`data/sources/large/`; its URL and sha256 are in the manifest.

## Round 2 (after the first judging)

The blind judge accepted 595/1,375 idioms, 828/981 facts but only 46/1,433 ConceptNet
common-sense items (the templates are mostly ambiguous or garbled). So:

- **Common sense is now generated** like creative/ood: 750 target words, K=5, instruction
  `COMMON_SENSE_GEN` (everyday uses, materials, places, typical behaviour, cause and effect;
  plain realistic sentences; shown as "Fill the blank (one word): ..."), 57 realistic
  settings. Targets: same wordfreq/Hunspell rules, plus at least 10 ConceptNet edges as an
  everyday-concreteness proxy; disjoint from creative/ood targets; two-category rule kept.
  The ConceptNet candidates, their judge units, outputs and judged file are kept as a
  record (`*_conceptnet*`) and are **not** used, including the 46 accepted ones: one
  construction method per category is simpler to describe and 46 items would not matter.
- **1,200 more idioms** from Wiktionary (same filters, same seeded order; the first 1,500
  are byte-identical), judged by new units `unit_idioms_0061`-`0108`
  (`llm_tasks judge --categories idioms --append`).
- `build_candidates` reproduces round 1 byte for byte and then applies round 2.

## Decisions

- **Judge is blind.** The judge never sees the target: it answers each question itself and
  rates it. `is_X_best` is computed in code as best answer == target (case-insensitive).
  v1 showed the judge the target and asked "is X best?", which invites agreement. The
  acceptance thresholds are v1's (ambiguity <= 3, not leaking, naturalness >= 7), plus one
  addition: `offensive` must be false, because the data is published. Judge batches never
  hold two candidates with the same target (siblings would hint at each other's answer).
- **ConceptNet:** api.conceptnet.io returned 502 on 2026-11-25 (`conceptnet/api_check.json`).
  The official 5.7.0 dump is used; it is the data the API served and is versioned.
- **Common sense** uses ConceptNet relations UsedFor, MadeOf, HasProperty, AtLocation,
  CapableOf and HasPrerequisite (v1 templates C1-C3, C5, C6, C8). v1's UsedBy, Contains and
  WornOn have no ConceptNet relation; HasPart (5.7.0 has it only as inverted PartOf) was
  tried and dropped, because its WordNet part lists rarely have one clear answer
  ("A city has a ____" -> "uptown"). Per (subject,
  relation) the strongest one-word answer is used only if it beats the runner-up by 1.5x in
  edge weight; subjects are singular dictionary words; verbs after "to"/"can" are base
  forms; groups of people (religions, ethnicities, genders) are blocked because ConceptNet
  holds crowd-sourced stereotypes. Articles are a/an by the first letter (v1 always used "a").
  Each subject is used at most twice; relations are filled round-robin. Expect a low judge
  acceptance rate here (HasProperty especially); 1,458 candidates is the headroom.
- **Facts:** v1's relations were weak (random people's birth months; P241 is "military
  branch", not national sport). v2 uses familiar entities (Wikipedia sitelinks as the
  familiarity filter): country capital/currency/official language/continent/highest
  point/demonym, capitals of US/Indian/Australian/Brazilian/German/Mexican states and
  Canadian provinces, and occupation of famous people (one of 14 headline occupations),
  plus new templates F16-F19: book -> author surname, painting -> painter surname, film ->
  director surname, element symbol -> element name. City -> country timed out at the public
  endpoint and was dropped. A subject is used only when Wikidata gives it exactly one
  usable answer. The real ceiling is about 1,080, short of 1,500.
- **Idioms:** baiango gives only 740 usable idioms, so Wiktionary's idiom category fills to
  1,500 (seeded sample of idioms with 3-8 words, no "someone/one's" placeholders, all words
  common). Idioms ending in a function word ("rub shoulders with") are dropped.
- **Same answer within a category** at most 3 times (facts: 5), so no single word dominates.
- **Two-category rule** is enforced at candidate level (3 words dropped). Creative/OOD
  targets exclude words already in two categories, and the two lists are disjoint.
- **Target words** for creative/OOD: wordfreq 3.1.1, Zipf 3.5-6.0, lowercase letters, 3+
  letters, not a function word, and a lowercase base entry in the Hunspell en_US list
  (removes names like "cleveland" and abbreviations like "cdc"); seeded shuffle, 750 each
  (about 700 asked; extra headroom because the generator may mark a word unusable).
- **Generation:** one unit per word, K=5, each item assigned a different setting (creative,
  64 settings) or framing (OOD, 32 framings) by seed, so 750 independent requests do not
  converge on the same scenes. Creative keeps v1's format (two-sentence story, blank last).
  OOD keeps "Fill the blank with one word:" but frees the final sentence: v1's two fixed
  endings (O1/O2) would make every OOD item end the same way.
- **Code filters** reject a target that appears in the question as a word, as a regular
  inflection, or as the start of a longer word (4+ letters, e.g. "lighthouse" for "light");
  non-ASCII; not exactly one blank; length outside 20-400 characters.
