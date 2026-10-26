"""APTM-style attribute prompt templates (arXiv:2306.02898, Suppl. B, Table 9).

Instead of one long concatenated sentence (the DEFAULT builder), each known
attribute becomes ONE short sentence built from one of five templates:

    1. the person is {x}             -> gender, age
    2. the person with {x}           -> hair, hat/backpack/bag/... (confirmed present)
    3. the person without {x}        -> hat/backpack/bag/... (confirmed ABSENT)   [NEG]
    4. the person wears {x}          -> clothing types + upper/lower colors
    5. the person does not wear {x}  -> colors explicitly ruled out ("not red")   [NEG]

Tri-state semantics (differs from the DEFAULT builder, which folds "no" into
"no information"): for presence-like fields an investigator's explicit "No" is
discriminative signal -> negative template; a blank/unknown cell -> no sentence.

Public API:
    build_templated_prompts(profile, include_negatives) -> list[str]
    build_templated_query(profile, include_negatives)   -> str   (<= 77 CLIP tokens)
"""
from __future__ import annotations

import logging
import math
import re
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:                                  # avoid import cycle at runtime
    from ..profile_schema import PersonProfileSchema

log = logging.getLogger("sar_person.prompting.templates")

# CLIP text encoder context length (tokens, incl. SOT/EOT)
CLIP_MAX_TOKENS = 77

# ---- value normalisation (R4: TRIM + LOWER; "unknown" never prompts) ---------
UNKNOWN_TOKENS = {"", "unknown", "n/a", "na", "nan", "-", "—", "?"}
ABSENT_TOKENS = {"no", "no.", "none", "n", "without"}
PRESENT_TOKENS = {"yes", "y", "yes."}
# explicit rule-outs inside a value, e.g. "not red", "definitely not red"
_NEG_VALUE_RE = re.compile(r"^(?:definitely\s+|surely\s+)?not\s+(?P<x>.+)$")


def clean(v) -> str:
    """strip + lower (same normalisation contract as the DEFAULT builder)."""
    return "" if v is None else str(v).strip().lower()


def tri_state(v) -> str:
    """'yes' (confirmed present) / 'no' (confirmed absent) / 'unknown'."""
    s = clean(v)
    if s in UNKNOWN_TOKENS:
        return "unknown"
    if s in ABSENT_TOKENS:
        return "no"
    return "yes"          # explicit yes OR a descriptive value ("white bag")


# ---- the five APTM templates --------------------------------------------------
class AttributeTemplate(Enum):
    IS = "the person is {}"
    WITH = "the person with {}"
    WITHOUT = "the person without {}"                 # negative
    WEARS = "the person wears {}"
    DOES_NOT_WEAR = "the person does not wear {}"     # negative

    def render(self, label_text: str) -> str:
        return self.value.format(label_text)

    @property
    def negative(self) -> bool:
        return self in (AttributeTemplate.WITHOUT, AttributeTemplate.DOES_NOT_WEAR)


# ---- priority groups for 77-token truncation (lower kept first) --------------
PRIO_CLOTHING_COLOR = 0
PRIO_CLOTHING_TYPE = 1
PRIO_ACCESSORY = 2
PRIO_DISTINGUISHING = 3
PRIO_PHYSICAL = 4
PRIO_IDENTITY = 5


def _age_phrase(v: str) -> "str | None":
    """APTM/PA-100K age buckets from a free-text age ('30s', '30-40', 'mid 60s')."""
    m = re.search(r"\d+", v)
    if not m:
        return None
    age = int(m.group())
    if age < 18:
        return "younger than 18 years old"
    if age > 60:
        return "older than 60 years old"
    return "older than 18 years old"


def _gender_phrase(v: str) -> "str | None":
    if v.startswith("m"):
        return "a man"
    if v.startswith("f") or v.startswith("w"):
        return "a woman"
    return None


def _hair_phrase(v: str) -> str:
    return v if "hair" in v else f"{v} hair"


def _color_sentences(field: str, v: str, part: str, include_negatives: bool):
    """upper_color / lower_color -> WEARS, or DOES_NOT_WEAR on explicit 'not X'."""
    neg = _NEG_VALUE_RE.match(v)
    if neg:
        if include_negatives:
            yield AttributeTemplate.DOES_NOT_WEAR.render(
                f"{neg.group('x')} {part} clothes")
        return
    yield AttributeTemplate.WEARS.render(f"{v} {part} clothes")


# presence-like fields -> (canonical noun for WITH/WITHOUT)
PRESENCE_NOUNS = {
    "hat": "a hat",
    "backpack": "a backpack",
    "bag": "a bag",
    "glasses": "glasses",
    "high_vis": "a high-visibility vest",
    "trekking_poles": "trekking poles",
    "umbrella": "an umbrella",
}

# field -> (template used for positives, priority) — the registry (R1).
ATTRIBUTE_TEMPLATE_REGISTRY = {
    "gender":         (AttributeTemplate.IS, PRIO_IDENTITY),
    "age":            (AttributeTemplate.IS, PRIO_IDENTITY),
    "height":         (AttributeTemplate.IS, PRIO_PHYSICAL),
    "build":          (AttributeTemplate.IS, PRIO_PHYSICAL),
    "hair_color":     (AttributeTemplate.WITH, PRIO_PHYSICAL),
    "hair_style":     (AttributeTemplate.WITH, PRIO_PHYSICAL),
    "skin_tone":      (AttributeTemplate.WITH, PRIO_PHYSICAL),
    "facial_hair":    (AttributeTemplate.WITH, PRIO_PHYSICAL),
    "upper_type":     (AttributeTemplate.WEARS, PRIO_CLOTHING_TYPE),
    "upper_color":    (AttributeTemplate.WEARS, PRIO_CLOTHING_COLOR),
    "upper_pattern":  (AttributeTemplate.WEARS, PRIO_CLOTHING_TYPE),
    "lower_type":     (AttributeTemplate.WEARS, PRIO_CLOTHING_TYPE),
    "lower_color":    (AttributeTemplate.WEARS, PRIO_CLOTHING_COLOR),
    "footwear":       (AttributeTemplate.WEARS, PRIO_CLOTHING_TYPE),
    "hat":            (AttributeTemplate.WITH, PRIO_ACCESSORY),
    "high_vis":       (AttributeTemplate.WITH, PRIO_ACCESSORY),
    "backpack":       (AttributeTemplate.WITH, PRIO_ACCESSORY),
    "bag":            (AttributeTemplate.WITH, PRIO_ACCESSORY),
    "trekking_poles": (AttributeTemplate.WITH, PRIO_ACCESSORY),
    "umbrella":       (AttributeTemplate.WITH, PRIO_ACCESSORY),
    "other_carried":  (AttributeTemplate.WITH, PRIO_ACCESSORY),
    "glasses":        (AttributeTemplate.WITH, PRIO_ACCESSORY),
    "tattoo":         (AttributeTemplate.WITH, PRIO_DISTINGUISHING),
    "scar":           (AttributeTemplate.WITH, PRIO_DISTINGUISHING),
    "other_feature":  (AttributeTemplate.WITH, PRIO_DISTINGUISHING),
}


def _sentences(profile: "PersonProfileSchema",
               include_negatives: bool) -> list[tuple[str, int, str]]:
    """[(sentence, priority, field)] in reading order (identity first)."""
    def raw(field: str) -> str:
        # prefer the raw ingest value (keeps explicit "No"); fall back to field
        return clean(profile.raw_value(field)) if hasattr(profile, "raw_value") \
            else clean(getattr(profile, field, ""))

    out: list[tuple[str, int, str]] = []

    def add(field: str, sentence: "str | None", prio: int):
        if sentence:
            out.append((sentence, prio, field))

    # -- identity ---------------------------------------------------------
    g = raw("gender")
    if g not in UNKNOWN_TOKENS:
        add("gender", (lambda p: AttributeTemplate.IS.render(p) if p else None)(
            _gender_phrase(g)), PRIO_IDENTITY)
    a = raw("age")
    if a not in UNKNOWN_TOKENS:
        p = _age_phrase(a)
        add("age", AttributeTemplate.IS.render(p) if p else None, PRIO_IDENTITY)

    # -- physical ---------------------------------------------------------
    for field, phrase in (("hair_style", _hair_phrase), ("hair_color", _hair_phrase),
                          ("facial_hair", lambda v: v),
                          ("skin_tone", lambda v: f"{v} skin")):
        v = raw(field)
        if v not in UNKNOWN_TOKENS and v not in ABSENT_TOKENS:
            add(field, AttributeTemplate.WITH.render(phrase(v)), PRIO_PHYSICAL)
    v = raw("build")
    if v not in UNKNOWN_TOKENS:
        add("build", AttributeTemplate.IS.render(v), PRIO_PHYSICAL)
    v = raw("height")
    if v not in UNKNOWN_TOKENS:
        add("height", AttributeTemplate.IS.render(f"about {v} tall"), PRIO_PHYSICAL)

    # -- clothing (template 4 + negative template 5) ------------------------
    for field, part in (("upper_color", "upper"), ("lower_color", "lower")):
        v = raw(field)
        if v not in UNKNOWN_TOKENS and v not in ABSENT_TOKENS:
            for s in _color_sentences(field, v, part, include_negatives):
                add(field, s, PRIO_CLOTHING_COLOR)
    for field, fmt in (("upper_type", "a {}"), ("lower_type", "{}"),
                       ("footwear", "{}")):
        v = raw(field)
        if v not in UNKNOWN_TOKENS and v not in ABSENT_TOKENS:
            add(field, AttributeTemplate.WEARS.render(fmt.format(v)),
                PRIO_CLOTHING_TYPE)
    v = raw("upper_pattern")
    if v not in UNKNOWN_TOKENS and v not in ABSENT_TOKENS and v != "solid":
        add("upper_pattern",
            AttributeTemplate.WEARS.render(f"{v} patterned upper clothes"),
            PRIO_CLOTHING_TYPE)

    # -- accessories: tri-state presence (templates 2 & 3) ------------------
    for field, noun in PRESENCE_NOUNS.items():
        state = tri_state(raw(field))
        if state == "yes":
            add(field, AttributeTemplate.WITH.render(noun), PRIO_ACCESSORY)
        elif state == "no" and include_negatives:
            add(field, AttributeTemplate.WITHOUT.render(noun), PRIO_ACCESSORY)
    v = raw("other_carried")
    if v not in UNKNOWN_TOKENS and v not in ABSENT_TOKENS:
        add("other_carried", AttributeTemplate.WITH.render(v), PRIO_ACCESSORY)

    # -- distinguishing ------------------------------------------------------
    for field, fmt in (("tattoo", "a tattoo on the {}"), ("scar", "a scar on the {}"),
                       ("other_feature", "{}")):
        v = raw(field)
        if v not in UNKNOWN_TOKENS and v not in ABSENT_TOKENS:
            add(field, AttributeTemplate.WITH.render(fmt.format(v)),
                PRIO_DISTINGUISHING)

    # de-dup, keep first occurrence
    seen, uniq = set(), []
    for s, p, f in out:
        if s not in seen:
            seen.add(s)
            uniq.append((s, p, f))
    return uniq


def build_templated_prompts(profile: "PersonProfileSchema",
                            include_negatives: bool) -> list[str]:
    """One short APTM-style sentence per known attribute (reading order)."""
    return [s for s, _, _ in _sentences(profile, include_negatives)]


# ---- 77-token budget -----------------------------------------------------------
_tokenizer = None


def _count_tokens(text: str) -> int:
    """CLIP BPE token count (incl. SOT/EOT); falls back to a safe heuristic."""
    global _tokenizer
    if _tokenizer is None:
        try:
            from utils.simple_tokenizer import SimpleTokenizer
            _tokenizer = SimpleTokenizer()
        except Exception:
            _tokenizer = False
    if _tokenizer:
        return len(_tokenizer.encode(text)) + 2
    # heuristic calibrated against CLIP BPE: ~1 token per word/punctuation,
    # 2 for long words (>8 chars), + SOT/EOT. Matches the real tokenizer on
    # these template sentences (e.g. example query: 66 == 66).
    parts = re.findall(r"\w+|[^\w\s]", text)
    return 2 + sum(max(1, math.ceil(len(p) / 8)) for p in parts)


def build_templated_query(profile: "PersonProfileSchema",
                          include_negatives: bool,
                          max_tokens: int = CLIP_MAX_TOKENS) -> str:
    """Join sentences with '. ' into one CLIP query, keeping within max_tokens.

    On overflow, sentences are kept in priority order (clothing colors >
    accessories > distinguishing > physical > identity) and the dropped
    attributes are logged.
    """
    sents = _sentences(profile, include_negatives)
    # stable sort: priority group, then reading order within the group
    ordered = sorted(range(len(sents)), key=lambda i: (sents[i][1], i))

    kept, dropped = [], []
    for i in ordered:
        s, _, field = sents[i]
        candidate = ". ".join(x for x, _ in kept + [(s, field)])
        if _count_tokens(candidate) <= max_tokens:
            kept.append((s, field))
        else:
            dropped.append(field)
    if dropped:
        log.warning(
            "templated query for %s exceeds %d CLIP tokens; dropped "
            "lower-priority attributes: %s",
            getattr(profile, "person_id", "?") or "?", max_tokens,
            ", ".join(dropped))
    return ". ".join(s for s, _ in kept)
