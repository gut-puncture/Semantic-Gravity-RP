"""Instruction texts and JSON Schemas for the LLM work units (generation and judging).

The schemas use only keywords every structured-output backend accepts (type, properties,
required, additionalProperties, enum, items), so the same file works with the OpenAI API
(strict structured outputs) and `codex exec --output-schema`. Ranges, counts and other
semantic rules are checked in llm_collect.py.
"""
from __future__ import annotations

K_PER_TARGET = 5

# Settings and framings are assigned to items (5 distinct per unit, seeded) so that 750
# independent requests do not all drift to the model's favourite scenes.
CREATIVE_SETTINGS = [
    "a busy restaurant kitchen", "a farm at dawn", "a hospital night shift", "a primary school classroom",
    "a fishing boat at sea", "a mountain hiking trail", "a city subway at rush hour", "a wedding reception",
    "a football match", "a family road trip", "a bakery before opening", "a camping trip in the woods",
    "a science laboratory", "a courtroom", "an airport departure gate", "a small-town library",
    "a construction site", "a child's birthday party", "a garden in spring", "a snowy evening at home",
    "a desert expedition", "an artist's studio", "a car repair garage", "a jazz club",
    "a supermarket checkout", "a rainy afternoon in an old apartment", "a newspaper office", "a beach holiday",
    "a zoo", "a theatre rehearsal", "a job interview", "moving day", "a long train journey",
    "a village market", "a museum", "a dentist's waiting room", "a rock concert", "a chess tournament",
    "a police station", "a hotel reception desk", "a ski resort", "a pirate ship", "a medieval castle",
    "a space station", "a 1940s detective's office", "a fairy-tale forest", "a post office", "a pet shop",
    "a carpentry workshop", "a tailor's shop", "a harvest festival", "a coffee shop", "a final exam",
    "a care home", "a flood evacuation", "a sailing race", "a recording studio", "a bank", "a gym",
    "a vineyard", "a laundromat", "a lighthouse in a storm", "a refugee camp kitchen", "a submarine",
]

OOD_FRAMINGS = [
    "the rulebook of an invented board game", "the ritual manual of an imaginary order of monks",
    "firmware release notes for an impossible machine", "a customs form for travellers between dreams",
    "a cookbook for cooking weather", "safety instructions for a time machine",
    "the laws of physics in a pocket universe", "an etiquette guide for a planet of sentient clouds",
    "an entry in a wizard's spellbook", "an insurance policy for ghosts",
    "a museum placard from the year 4000", "the tax code of a kingdom of dragons",
    "an error log from a sentient refrigerator", "a conversion table between colours and smells",
    "a training manual for mermaid lifeguards", "a recipe written by a robot that has never eaten",
    "the minutes of a meeting of household objects", "a law passed by a parliament of cats",
    "a weather report for the inside of a teapot", "the user guide for a door that only opens to riddles",
    "the rulebook of zero-gravity tea parties", "a dating profile written by a volcano",
    "a tourist brochure for the basement of the moon", "an entry in an alien encyclopedia of Earth",
    "a contract between the sun and the night", "a lab protocol for growing music",
    "a prophecy written as a spreadsheet formula", "assembly instructions for a pocket-sized ocean",
    "a court ruling in the trial of a shadow", "the operating manual of a cloud factory",
    "a census form for the inhabitants of a dream", "a patent for a machine that runs on silence",
]

CREATIVE_TEMPLATE = "Fill the blank with one word: {text}"
OOD_TEMPLATE = "Fill the blank with one word: {text}"

GEN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["target_word", "target_usable", "unusable_reason", "items"],
    "properties": {
        "target_word": {"type": "string"},
        "target_usable": {"type": "boolean"},
        "unusable_reason": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["setting", "text"],
                "properties": {"setting": {"type": "string"}, "text": {"type": "string"}},
            },
        },
    },
}

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "best_one_word_answer", "top3", "ambiguity", "leaks_answer", "naturalness",
                             "offensive"],
                "properties": {
                    "id": {"type": "string"},
                    "best_one_word_answer": {"type": "string"},
                    "top3": {"type": "array", "items": {"type": "string"}},
                    "ambiguity": {"type": "integer"},
                    "leaks_answer": {"type": "boolean"},
                    "naturalness": {"type": "integer"},
                    "offensive": {"type": "boolean"},
                },
            },
        },
    },
}

# ----------------------------------------------------------------- generation

_GEN_COMMON_RULES = """\
Rules for every item:
1. "{X}" must be the answer that nearly every fluent English speaker would give, in exactly
   this form. Make the grammar around the blank demand this form (singular or plural,
   tense, part of speech), so that a different form of the word does not fit.
2. The text must not contain "{X}" in any form: no inflection (plural, past tense, -ing,
   -ly, -er), no derived word, no compound or longer word that contains it (for "rain":
   no "raindrop", "rainy", "rainbow"). Also no translation of it, no spelling or letter
   hints, no rhymes, no quotation of the word, and no talk about "the word" or "the answer".
3. Rule out close rivals. Before you finalise an item, list to yourself the other words
   that could fill the blank; if any of them fits nearly as well as "{X}", add a concrete
   detail that excludes it, or rewrite the item.
4. Reach the answer through meaning, not through a dictionary definition: a situation, a
   cause and its effect, a familiar collocation, an action a character takes, a sensory
   detail. A reader should feel the word is inevitable.
5. Exactly one blank, written as four underscores: ____
6. Plain ASCII characters only. Fluent, grammatical English. Between 15 and 60 words.
7. The five items must be genuinely different from each other: use the five assigned
   settings below (one per item, in order), and vary the sentence structure and the way
   the answer is forced. Do not reuse a phrase across items.
8. Nothing hateful, sexual or gory.

If "{X}" cannot be the single best answer of any natural item, for example because it is
a function word, a personal name, a slur, an abbreviation or not standard English, set
"target_usable" to false, give a short reason in "unusable_reason", and return an empty
"items" list. Otherwise set "target_usable" to true, "unusable_reason" to "", and return
exactly five items. In each item, "setting" names the setting you used (the assigned one,
or the one you switched to if the assigned setting truly cannot force "{X}")."""

CREATIVE_GEN = """\
You write test items for a research study of how language models answer one-word
fill-in-the-blank questions. Each item is shown to a reader as

  Fill the blank with one word: <your text>

and the reader must reply with exactly one English word.

TARGET WORD: "{X}"

Write five items whose single clearly best answer is "{X}". Each item is a micro-story of
exactly two sentences. The blank is the final word of the second sentence, followed by a
full stop, like "... the endless ____." Write only the story text (the "text" field); do
not write "Fill the blank with one word:".

""" + _GEN_COMMON_RULES + """

Good examples:
- GOOD (target "ladder"): "The kitten had climbed to the top of the old oak and refused to come down. Grandpa leaned something with twelve metal rungs against the trunk: his trusty ____."
- GOOD (target "umbrella"): "The rain hammered the pavement outside the theatre. The doorman snapped open a huge black ____."
- GOOD (target "whisper"): "The baby had finally fallen asleep in the next room. Bursting with news, Dad tiptoed in, leaned close to Mum and began to ____."

Bad examples:
- BAD (target "ladder", rivals): "The kitten climbed the tree and could not get down. Grandpa went to get help from the ____." Neighbours, firemen and many other answers fit.
- BAD (target "rain", leak): "Dark clouds gathered and the first raindrops fell. Soon heavy ____." "raindrops" contains the target.

Assigned settings, one per item in order:
{SETTINGS}

Return JSON only, matching the provided schema, with "target_word" set to "{X}".
"""

OOD_GEN = """\
You write test items for a research study of how language models answer one-word
fill-in-the-blank questions. This set is "out of distribution": the items must use framings
that almost never occur in ordinary text, such as rules of invented games, manuals of
imaginary rituals, pseudo-technical specifications of impossible machines, surreal physics,
paperwork from fictional agencies, or the customs of non-human beings. Each item is shown to
a reader as

  Fill the blank with one word: <your text>

and the reader must reply with exactly one English word.

TARGET WORD: "{X}"

Write five items whose single clearly best answer is "{X}". Each item is one to three
sentences of the strange framing, with the blank in the last sentence (at the end or inside
it). Write only the text (the "text" field); do not write "Fill the blank with one word:".

The strangeness must not make the answer a guess. The text must force "{X}" by its own
internal logic: state an invented rule, law or procedure and make the blank its direct
consequence, or carry a strong everyday association of "{X}" into the strange world so that
it still decides the blank. A reader who accepts the premise should find "{X}" inevitable.

""" + _GEN_COMMON_RULES + """

Good examples:
- GOOD (target "winter"): "Release note 3.2 for the Seasonal Engine: once autumn finishes its leaf-drop routine, the scheduler always hands control to the next season in the cycle. Expect frost and snowfall when the engine switches to ____."
- GOOD (target "mirror"): "Regulation 9 of the Ghost Customs Office: a spirit may re-enter the living world only after proving it casts a reflection. Applicants are therefore led to the office's single piece of furniture, a tall silvered ____."
- GOOD (target "sneeze"): "In the Pepper Kingdom a law passes only if every councillor's nose itches at once. When the Grand Pepperer shook his dust over the assembly, the whole chamber let out one thunderous ____."

Bad examples:
- BAD (target "moon", nothing forces the answer): "In the realm of floating teacups the clocks sing at midnight. The wise owl declared that the secret is the ____." Any word fits.
- BAD (target "knight", not out of distribution): "In chess, the piece that moves in an L shape is the ____." This is an ordinary fact, not an unusual framing.

Assigned framings, one per item in order:
{SETTINGS}

Return JSON only, matching the provided schema, with "target_word" set to "{X}".
"""

# ----------------------------------------------------------------- judging

JUDGE = """\
You check fill-in-the-blank test items for a research study. Each item is a question that a
reader must answer with exactly one English word. The reader sees exactly:

  Answer with exactly one English word.
  Question: <question>
  Answer:

For each item below, first answer it yourself, as a careful, well-informed reader would,
then rate it. The items are unrelated to each other: judge each one on its own, and do not
let one item influence another. You are not told what answer the item's author intended, and
you should not try to guess it; give the answer you yourself think is best.

Return one entry per item, in the same order, with these fields:
- id: copied from the input.
- best_one_word_answer: the single word most fluent English speakers would give. Exactly one
  word, letters only, in the exact form that fits the blank (singular or plural, tense).
  Lowercase, except proper nouns (capitalised as usual). For factual items give the true
  answer.
- top3: the three most likely one-word answers, best first; the first must equal
  best_one_word_answer. If fewer than three are plausible, still fill three places with the
  next-best words, however weak.
- ambiguity: integer 0-10, how strongly other answers compete with your best answer.
  0 = only one sensible answer; 1-3 = one answer clearly dominant, alternatives rare or
  clearly worse; 4-6 = two or more answers about equally good; 7-10 = many answers fit.
- leaks_answer: true if the question gives its answer away instead of making the reader
  produce it: it contains your best answer or a form of it (plural, past tense, -ly, part of
  a compound such as "raindrop" for "rain"), or spells it, hints its letters, rhymes with it
  on purpose, or translates it. Completing a well-known idiom, phrase or fact is NOT a leak.
- naturalness: integer 0-10, is the question well-formed, fluent English a careful writer
  would produce? 10 = flawless; 7-9 = fine, perhaps slightly stiff; 4-6 = awkward or partly
  confusing; 0-3 = ungrammatical, garbled or factually broken. Template-style wording such as
  "Fill the blank (one word): The capital of Peru is ____." is normal for this test and
  scores high when grammatical.{CATEGORY_NOTE}
- offensive: true if the item is hateful, demeaning or stereotyping about a group of people,
  sexually explicit, or gory. The items will be published, so flag anything doubtful.

Examples (question -> output):
1. GOOD. "Fill the blank (one word): spill the ____"
   -> best "beans", top3 ["beans", "tea", "milk"], ambiguity 1, leaks_answer false, naturalness 9, offensive false
2. GOOD. "Fill the blank with one word: The capital of Peru is ____."
   -> best "Lima", top3 ["Lima", "Cusco", "Arequipa"], ambiguity 0, leaks_answer false, naturalness 10, offensive false
3. GOOD. "Fill the blank with one word: The rain hammered the pavement outside the theatre. The doorman snapped open a huge black ____."
   -> best "umbrella", top3 ["umbrella", "parasol", "door"], ambiguity 1, leaks_answer false, naturalness 9, offensive false
4. BAD, ambiguous. "Fill the blank (one word): A typical apple is ____."
   -> best "red", top3 ["red", "sweet", "round"], ambiguity 7, leaks_answer false, naturalness 8, offensive false
5. BAD, leak and awkward. "Fill the blank (one word): You use a bread knife to cut the loaf of ____."
   -> best "bread", top3 ["bread", "cake", "rye"], ambiguity 1, leaks_answer true, naturalness 5, offensive false

Items (category: {CATEGORY}):
{ITEMS}

Return JSON only, matching the provided schema.
"""

JUDGE_CATEGORY_NOTES = {
    "ood": ("\n  This batch is out of distribution on purpose: surreal, invented or pseudo-technical\n"
            "  premises are intended. Judge only whether the text is well-formed and internally\n"
            "  coherent, not whether it is realistic. Ambiguity still applies: if the invented\n"
            "  premise does not force one answer, the ambiguity is high."),
    "facts": ("\n  If the question is factually unanswerable or rests on a false premise, give your best\n"
              "  answer anyway and set naturalness to 3 or less."),
}
