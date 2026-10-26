"""NOMAD ``annotations/metadata.json`` -> ``PersonProfileSchema`` -> CLIP prompts.

NOMAD (WACV 2024) describes each of its 100 actors ONCE, in
``annotations/metadata.json``: demographic block (age, gender, race, weight,
height, facial hair, hair) + outfit block (clothes[], shoes, hat, gloves,
glasses). Every selected frame of that actor -- 5 altitudes x ~85..150 frames --
therefore shares a single appearance description. That is exactly the
"many crops, one caption" supervision this module produces.

The NOMAD field names do NOT match the project's "Missing Person Profile"
schema, so this module is the adapter: it lowers each actor's metadata onto a
``PersonProfileSchema`` instance and then reuses the *existing*, unmodified
prompt builders so NOMAD captions are phrased identically to every other
dataset in the project:

    profile.to_prompt()                       -> DEFAULT (comma concatenation)
    profile.get_prompt(PromptStyle.TEMPLATED) -> APTM attribute templates
    profile.get_prompt(TEMPLATED_NEG)         -> + "without"/"does not wear"

Tri-state discipline (the reason ``raw`` is populated as well as the normalised
fields): NOMAD encodes "this actor has no hat" as ``"hat": false``, which is
*confirmed absent* and a discriminative signal -> raw value ``"no"`` -> the APTM
negative template. Attributes NOMAD simply never recorded (backpack, bag,
trekking poles, umbrella, high-vis, tattoo, scar, upper/lower garment TYPE) are
left blank -> *unknown* -> no sentence at all. Never write "no" for those; it
would assert a fact the dataset does not support.

Public API
----------
    load_nomad_metadata(path)              -> {actor_id: raw record}
    build_profile(record, ...)             -> PersonProfileSchema
    build_actor_captions(record, ...)      -> {"prompt_default", "prompt_templated", ...}
    actor_dir_name(actor_id)               -> "Actor001"
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .profile_schema import PersonProfileSchema

# --------------------------------------------------------------------------- #
# NOMAD categorical vocabularies -> the project's descriptor vocabulary       #
# --------------------------------------------------------------------------- #

# metadata.demographic.outfit.shoes.type
SHOE_TYPES = {
    "close informal": "sneakers",
    "open informal": "sandals",
    "close formal": "dress shoes",
    "open formal": "formal sandals",
    "boots": "boots",
    "bare": "barefoot",
}

# metadata.demographic.outfit.clothes[].pattern. "plain" is NOMAD's word for
# "no pattern"; the prompt builders already drop the token "solid", so mapping
# plain -> solid makes the uninformative case disappear from every prompt.
PATTERNS = {
    "plain": "solid",
    "random": "multicolored",
    "both": "striped and patterned",
    "horizontal stripes": "horizontally striped",
    "vertical stripes": "vertically striped",
}

# metadata.demographic.outfit.glasses
GLASSES = {"transparent": "glasses", "dark": "sunglasses"}

# metadata.demographic.race -> coarse skin-tone descriptor. OFF by default:
# race is a protected attribute and NOMAD's label is self-reported ancestry,
# not an appearance measurement, so deriving a visual descriptor from it is an
# assumption the dataset does not license. Enable only if your missing-person
# report schema genuinely carries this field (pass skin_tone="coarse").
RACE_TO_SKIN_TONE = {
    "african descent": "dark",
    "south asian": "brown",
    "latino/hispanic": "medium",
    "middle eastern": "medium",
    "pacific islander": "medium",
    "asian": "light",
    "white/caucasian": "fair",
}

# values NOMAD uses for "field not applicable / not recorded"
_NOMAD_NULL = {"", "false", "none", "unknown", "n/a", "nan"}


def _s(v) -> str:
    """NOMAD value -> clean lowercase string, '' for its null spellings."""
    if v is None or v is False:
        return ""
    s = str(v).strip().lower()
    return "" if s in _NOMAD_NULL else s


def actor_dir_name(actor_id: int) -> str:
    """1 -> 'Actor001' (the directory naming used under images/ and labels/)."""
    return f"Actor{int(actor_id):03d}"


# --------------------------------------------------------------------------- #
# derived descriptors                                                         #
# --------------------------------------------------------------------------- #

def height_phrase(inches) -> str:
    """64 -> \"5 ft 4 in\" (renders as 'the person is about 5 ft 4 in tall')."""
    try:
        n = int(inches)
    except (TypeError, ValueError):
        return ""
    if n <= 0:
        return ""
    return f"{n // 12} ft {n % 12} in"


def build_phrase(weight_lbs, height_in) -> str:
    """BMI -> a one-word build descriptor.

    NOMAD gives weight+height but no build label, and BMI is the only defensible
    bridge. Buckets follow the WHO cut-offs; the wording is chosen so it reads
    correctly through BOTH builders ("with a {v} build" / "the person is {v}").
    """
    try:
        w, h = float(weight_lbs), float(height_in)
    except (TypeError, ValueError):
        return ""
    if w <= 0 or h <= 0:
        return ""
    bmi = 703.0 * w / (h * h)
    if bmi < 18.5:
        return "slim"
    if bmi < 25.0:
        return "medium"
    if bmi < 30.0:
        return "stocky"
    return "heavy"


def hair_style_phrase(hair: dict) -> str:
    """{'lenght': 'medium', 'descriptor': 'ponytail'} -> 'medium ponytail'."""
    length = _s(hair.get("lenght") or hair.get("length"))
    desc = _s(hair.get("descriptor"))
    if length == "bald":
        return "bald"
    parts = [p for p in (length, desc) if p]
    return " ".join(parts)


def facial_hair_phrase(fh: dict) -> str:
    beard, mustache = bool(fh.get("beard")), bool(fh.get("mustache"))
    if beard and mustache:
        return "a beard and a mustache"
    if beard:
        return "a beard"
    if mustache:
        return "a mustache"
    return ""                                  # -> 'no' (confirmed clean-shaven)


def footwear_phrase(shoes: dict) -> str:
    kind = SHOE_TYPES.get(_s(shoes.get("type")), _s(shoes.get("type")))
    color = _s(shoes.get("color"))
    if kind == "barefoot" or color == "bare":
        return "barefoot"
    return f"{color} {kind}".strip() if kind else ""


def _split_garments(clothes: list, upper_layer: str = "last") \
        -> tuple[dict, dict, list]:
    """clothes[] -> (primary upper, lower, other upper layers).

    15 of NOMAD's 100 actors list TWO upper garments (a base layer plus a jacket
    or fleece), in file order ``[upper, lower, upper]`` -- the extra upper comes
    last. Which one should the caption call "the top"?

    ``upper_layer="last"`` (default) picks the outermost, because these are
    top-down aerial views: a drone sees the jacket, not the shirt underneath, so
    the outer colour is the one actually present in the crop pixels. Pass
    ``"first"`` to caption the base layer instead (matches a witness statement
    like "he was wearing a red t-shirt"). Either way the other layer is not
    discarded -- it becomes an ``other_feature`` clause.

    ``location: "both"`` is NOMAD's one-piece garment (dress / coverall): it is
    the primary upper AND supplies the lower colour.
    """
    uppers = [c for c in clothes if _s(c.get("location")) == "upper"]
    lowers = [c for c in clothes if _s(c.get("location")) == "lower"]
    boths = [c for c in clothes if _s(c.get("location")) == "both"]
    if boths:
        return boths[0], boths[0], uppers + lowers
    if not uppers:
        return {}, (lowers[0] if lowers else {}), []
    idx = -1 if upper_layer == "last" else 0
    primary = uppers[idx]
    extra = [c for i, c in enumerate(uppers) if c is not primary]
    return primary, (lowers[0] if lowers else {}), extra


# --------------------------------------------------------------------------- #
# metadata record -> PersonProfileSchema                                      #
# --------------------------------------------------------------------------- #

def build_profile(record: dict, skin_tone: str = "none",
                  upper_layer: str = "last") -> PersonProfileSchema:
    """One ``metadata.json`` record -> a populated ``PersonProfileSchema``.

    Parameters
    ----------
    record      : one element of ``annotations/metadata.json``.
    skin_tone   : ``"none"`` (default, no skin-tone attribute is emitted) or
                  ``"coarse"`` (derive one from ``demographic.race`` via
                  :data:`RACE_TO_SKIN_TONE`).
    upper_layer : which of several upper garments is "the top" -- ``"last"``
                  (outermost, what a drone sees) or ``"first"`` (base layer).
                  See :func:`_split_garments`.
    """
    demo = record.get("demographic", {}) or {}
    outfit = demo.get("outfit", {}) or {}
    clothes = outfit.get("clothes", []) or []
    up, low, extra = _split_garments(clothes, upper_layer)

    one_piece = bool(clothes) and _s((up or {}).get("location")) == "both"

    # distinguishing marks: garment graphics + extra layers + gloves. NOMAD
    # records these as free text ("notre dame logo", "44 number on back"), which
    # is precisely the kind of detail a witness would report.
    # Kept deliberately terse: other_feature is PRIO_DISTINGUISHING, which
    # outranks physical/identity attributes in the APTM 77-token budget, so a
    # verbose graphic description would silently evict gender and age from the
    # query. NOMAD's own figure text is already descriptive ("notre dame logo").
    features: list[str] = []
    for c in clothes:
        fig = _s(c.get("figure"))
        if fig:
            where = _s(c.get("location"))
            features.append(fig if where in ("upper", "both", "")
                            else f"{fig} ({where})")
    for c in extra:
        col = _s(c.get("color"))
        if col:
            features.append(f"{col} layer" if upper_layer == "last"
                            else f"{col} outer layer")
    gloves = _s(outfit.get("gloves"))
    if gloves:
        features.append(f"{gloves} gloves")

    fields: Dict[str, str] = {
        # ---- identity ----
        "age": str(demo.get("age") or "").strip(),
        "gender": _s(demo.get("gender")),
        # ---- physical ----
        "height": height_phrase(demo.get("height[in]")),
        "build": build_phrase(demo.get("weight[lbs]"), demo.get("height[in]")),
        "hair_color": _s((demo.get("hair") or {}).get("color")),
        "hair_style": hair_style_phrase(demo.get("hair") or {}),
        "skin_tone": (RACE_TO_SKIN_TONE.get(_s(demo.get("race")), "")
                      if skin_tone == "coarse" else ""),
        "facial_hair": facial_hair_phrase(demo.get("facial_hair") or {}),
        # ---- clothing ----
        "upper_type": "dress" if one_piece else "",
        "upper_color": _s((up or {}).get("color")),
        "upper_pattern": PATTERNS.get(_s((up or {}).get("pattern")),
                                      _s((up or {}).get("pattern"))),
        "lower_type": "",                       # NOMAD records colour, not type
        "lower_color": _s((low or {}).get("color")),
        "footwear": footwear_phrase(outfit.get("shoes") or {}),
        # ---- accessories ----
        "hat": (f"{_s(outfit.get('hat'))} hat" if _s(outfit.get("hat")) else ""),
        "glasses": GLASSES.get(_s(outfit.get("glasses")), ""),
        "other_feature": ", ".join(features),
        # ---- not recorded by NOMAD: stay blank == UNKNOWN, never "no" ----
        "high_vis": "", "backpack": "", "bag": "", "trekking_poles": "",
        "umbrella": "", "other_carried": "", "tattoo": "", "scar": "",
        "full_name": "", "nationality": "",
    }

    # Tri-state raw values. NOMAD's explicit ``false`` is CONFIRMED ABSENT for
    # the three attributes it actually surveys, so those get "no" and drive the
    # APTM negative templates; everything else mirrors the normalised field.
    raw = dict(fields)
    for key, present in (("hat", bool(_s(outfit.get("hat")))),
                         ("glasses", outfit.get("glasses") is not False),
                         ("facial_hair", bool(fields["facial_hair"]))):
        if not present:
            raw[key] = "no"

    return PersonProfileSchema(person_id=actor_dir_name(record.get("id", 0)),
                               row_index=int(record.get("id", 0)),
                               raw=raw, **fields)


def build_actor_captions(record: dict, skin_tone: str = "none",
                         upper_layer: str = "last") -> dict:
    """Every prompt flavour for one actor, plus the attributes behind them."""
    from .prompting import PromptStyle

    profile = build_profile(record, skin_tone=skin_tone, upper_layer=upper_layer)
    return {
        "person_id": profile.person_id,
        "prompt_default": profile.to_prompt(),
        "prompt_templated": profile.get_prompt(PromptStyle.TEMPLATED),
        "prompt_templated_neg": profile.get_prompt(PromptStyle.TEMPLATED_NEG),
        "attributes": profile.filled_attributes(),
    }


def load_nomad_metadata(path: str | Path) -> Dict[str, dict]:
    """``annotations/metadata.json`` -> ``{"Actor001": record, ...}``."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(records, dict):                # tolerate a keyed variant
        records = list(records.values())
    return {actor_dir_name(r["id"]): r for r in records if "id" in r}


__all__ = ["load_nomad_metadata", "build_profile", "build_actor_captions",
           "actor_dir_name", "height_phrase", "build_phrase",
           "SHOE_TYPES", "PATTERNS", "GLASSES", "RACE_TO_SKIN_TONE"]
