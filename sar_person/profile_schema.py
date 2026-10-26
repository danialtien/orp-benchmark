"""PersonProfileSchema + CLIP-prompt construction mechanism.

One row of the "Missing Person Profile" sheet -> a structured PersonProfileSchema
-> a complete CLIP prompt built field-by-field, skipping meaningless values
(none / no / empty / n/a / unknown / solid pattern).

This is the prompt source used during training (instead of the Excel AD column),
so prompts always reflect the current field values.
"""
from __future__ import annotations
from dataclasses import dataclass, field as _dc_field, fields as _dc_fields
import re

# ---- column map (Excel letter -> schema field) ------------------------------
COLUMN = {
    "full_name": "B", "age": "C", "gender": "D", "nationality": "E", "height": "F",
    "build": "G", "hair_color": "H", "hair_style": "I", "skin_tone": "J",
    "facial_hair": "K", "upper_type": "L", "upper_color": "M", "upper_pattern": "N",
    "lower_type": "O", "lower_color": "P", "footwear": "Q", "hat": "R",
    "high_vis": "S", "backpack": "T", "bag": "U", "trekking_poles": "V",
    "umbrella": "W", "other_carried": "X", "glasses": "Y", "tattoo": "Z",
    "scar": "AA", "other_feature": "AB",
}

# ---- "no information" values that must never enter a prompt ------------------
NULL_TOKENS = {"", "none", "no", "no.", "n/a", "na", "nan", "unknown", "-", "—"}


def _nn(v) -> str:
    """Normalised non-null string ('' if the value is meaningless)."""
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in NULL_TOKENS else s


def _yes(v) -> bool:
    return _nn(v).lower().startswith("y")


# ---- the important attributes, grouped (used for the prompt + documentation) -
ATTRIBUTE_GROUPS = {
    "Identity":      ["full_name", "age", "gender", "nationality"],
    "Physical":      ["height", "build", "hair_color", "hair_style", "skin_tone", "facial_hair"],
    "Clothing":      ["upper_type", "upper_color", "upper_pattern", "lower_type",
                      "lower_color", "footwear", "hat"],
    "Accessories":   ["high_vis", "backpack", "bag", "trekking_poles", "umbrella",
                      "other_carried", "glasses"],
    "Distinguishing": ["tattoo", "scar", "other_feature"],
}

# Order + phrasing of clauses in the prompt. Each fn(value)->phrase|None.
# (full_name / nationality are NOT visual -> excluded from the prompt.)
PROMPT_TEMPLATES = [
    ("age",            lambda v: f"who appears to be around {v.lower()} years old"),
    ("height",         lambda v: f"about {v.lower()} tall"),
    ("build",          lambda v: f"with a {v.lower()} build"),
    ("hair_color",     lambda v: f"with {v.lower()} hair"),
    ("hair_style",     lambda v: f"in a {v.lower()} hairstyle"),
    ("skin_tone",      lambda v: f"with {v.lower()} skin tone"),
    ("facial_hair",    lambda v: f"with {v.lower()}"),
    ("upper_type",     lambda v: f"wearing a {v.lower()}"),
    ("upper_color",    lambda v: f"in a {v.lower()} top"),
    ("upper_pattern",  lambda v: (None if v.lower() == "solid" else f"with a {v.lower()} pattern")),
    ("lower_type",     lambda v: f"wearing {v.lower()}"),
    ("lower_color",    lambda v: f"in {v.lower()} colored bottoms"),
    ("footwear",       lambda v: f"wearing {v.lower()}"),
    ("hat",            lambda v: f"wearing a {v.lower()}"),
    ("high_vis",       lambda v: ("wearing a high-visibility vest" if _yes(v) else None)),
    ("backpack",       lambda v: f"carrying a {v.lower()}"),
    ("bag",            lambda v: f"carrying a {v.lower()}"),
    ("trekking_poles", lambda v: ("holding trekking poles" if _yes(v) else None)),
    ("umbrella",       lambda v: ("holding an umbrella" if v.lower() in ("yes", "y")
                                  else f"holding a {v.lower()} umbrella")),
    ("other_carried",  lambda v: f"carrying {v.lower()}"),
    ("glasses",        lambda v: f"wearing {v.lower()}"),
    ("tattoo",         lambda v: f"with a visible tattoo on the {v.lower()}"),
    ("scar",           lambda v: f"with a visible scar on the {v.lower()}"),
    ("other_feature",  lambda v: f"{v.lower()}"),
]


@dataclass
class PersonProfileSchema:
    person_id: str = ""
    row_index: int = 0
    full_name: str = ""
    age: str = ""
    gender: str = ""
    nationality: str = ""
    height: str = ""
    build: str = ""
    hair_color: str = ""
    hair_style: str = ""
    skin_tone: str = ""
    facial_hair: str = ""
    upper_type: str = ""
    upper_color: str = ""
    upper_pattern: str = ""
    lower_type: str = ""
    lower_color: str = ""
    footwear: str = ""
    hat: str = ""
    high_vis: str = ""
    backpack: str = ""
    bag: str = ""
    trekking_poles: str = ""
    umbrella: str = ""
    other_carried: str = ""
    glasses: str = ""
    tattoo: str = ""
    scar: str = ""
    other_feature: str = ""
    # raw (stripped, un-normalised) cell values captured at ingest. Needed for
    # tri-state semantics: an investigator's explicit "No" is CONFIRMED-ABSENT,
    # a blank cell is UNKNOWN — but _nn() folds both into "". Fields above stay
    # normalised exactly as before, so to_prompt() output is unchanged.
    raw: dict = _dc_field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_excel(cls, ws, row: int) -> "PersonProfileSchema":
        kw = {"person_id": str(ws[f"A{row}"].value or "").strip(), "row_index": row,
              "raw": {}}
        for field, col in COLUMN.items():
            v = ws[f"{col}{row}"].value
            kw[field] = _nn(v)
            kw["raw"][field] = "" if v is None else str(v).strip()
        return cls(**kw)

    # ---- tri-state access (APTM negative templates) --------------------------
    def raw_value(self, name: str) -> str:
        """Raw ingest value for `name` (falls back to the normalised field)."""
        v = self.raw.get(name, "")
        return v if v else getattr(self, name, "")

    def presence(self, name: str) -> str:
        """'yes' / 'no' (confirmed absent) / 'unknown' for a presence-like field."""
        from .prompting.templates import tri_state
        return tri_state(self.raw_value(name))

    # ---- prompt-style entry point (R7) ---------------------------------------
    def get_prompt(self, style=None) -> str:
        """Build the CLIP query for a given PromptStyle.

        No argument (or PromptStyle.DEFAULT) -> byte-identical to to_prompt(),
        the current concatenation builder. TEMPLATED / TEMPLATED_NEG -> APTM
        attribute templates (arXiv:2306.02898 Table 9). Invalid styles raise
        ValueError listing the valid choices.
        """
        from .prompting import PromptStyle, coerce_style
        from .prompting.templates import build_templated_query
        style = PromptStyle.DEFAULT if style is None else coerce_style(style)
        if style is PromptStyle.DEFAULT:
            return self.to_prompt()
        return build_templated_query(
            self, include_negatives=(style is PromptStyle.TEMPLATED_NEG))

    def _gender_head(self) -> str:
        g = self.gender.lower()
        if g.startswith("m"):
            return "a man"
        if g.startswith("f"):
            return "a woman"
        return "a person"

    def to_prompt(self) -> str:
        """Build the full CLIP prompt, skipping meaningless values."""
        parts = [self._gender_head()]
        for field, fn in PROMPT_TEMPLATES:
            v = _nn(getattr(self, field))
            if not v:
                continue
            phrase = fn(v)
            if phrase:
                parts.append(phrase)
        # de-dup while preserving order
        seen, out = set(), []
        for p in parts:
            if p.lower() not in seen:
                seen.add(p.lower()); out.append(p)
        return ", ".join(out)

    def filled_attributes(self) -> dict:
        return {f.name: getattr(self, f.name) for f in _dc_fields(self)
                if f.name not in ("person_id", "row_index", "raw") and _nn(getattr(self, f.name))}

