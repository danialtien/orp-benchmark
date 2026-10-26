"""Prompt-style selection for SAR_Person.

PromptStyle
    DEFAULT       -> existing concatenation builder (PersonProfileSchema.to_prompt,
                     unchanged behaviour, still the auto default everywhere)
    TEMPLATED     -> APTM attribute-prompt templates, positives only
    TEMPLATED_NEG -> APTM templates, positives + negatives ("without" / "does not wear")

Template design follows APTM (arXiv:2306.02898), Supplementary B, Table 9.

Resolution precedence (resolve_prompt_styles): CLI > env SAR_PROMPT_STYLE > config
> default. Invalid style strings fail fast with the list of valid choices.
"""
from __future__ import annotations

import logging
import os

try:                                    # Python >= 3.11
    from enum import StrEnum as _StrEnum
except ImportError:                     # Python <= 3.10
    from enum import Enum

    class _StrEnum(str, Enum):
        def __str__(self) -> str:       # so f"{style}" == style.value
            return str(self.value)

log = logging.getLogger("sar_person.prompting")

ENV_VAR = "SAR_PROMPT_STYLE"


class PromptStyle(_StrEnum):
    DEFAULT = "default"
    TEMPLATED = "templated"
    TEMPLATED_NEG = "templated_neg"


def _valid() -> list[str]:
    return [s.value for s in PromptStyle]


def coerce_style(value: "PromptStyle | str") -> PromptStyle:
    """Parse a style string; raise ValueError listing valid choices on failure."""
    if isinstance(value, PromptStyle):
        return value
    try:
        return PromptStyle(str(value).strip().lower())
    except ValueError:
        raise ValueError(
            f"Invalid prompt style {value!r}. Valid choices: {', '.join(_valid())}"
        ) from None


def resolve_prompt_styles(
    cli: "list[str] | str | None" = None,
    env: "str | None" = None,
    config: "list[str] | str | None" = None,
) -> list[PromptStyle]:
    """Resolve prompt style(s) with precedence CLI > env > config > default.

    `env` defaults to os.environ[SAR_PROMPT_STYLE]; it may hold one style or a
    comma/space-separated list. Returns a non-empty list (ablation may evaluate
    several styles in one run). Never silently falls back on invalid input.
    """
    if env is None:
        env = os.environ.get(ENV_VAR) or None

    source, raw = "default", [PromptStyle.DEFAULT.value]
    if cli:
        source, raw = "cli", [cli] if isinstance(cli, str) else list(cli)
    elif env:
        source, raw = f"env {ENV_VAR}", [t for t in env.replace(",", " ").split() if t]
    elif config:
        source, raw = "config", [config] if isinstance(config, str) else list(config)

    styles = [coerce_style(v) for v in raw]
    log.info("prompt style(s) resolved to %s (from %s)",
             [s.value for s in styles], source)
    return styles


def resolve_prompt_style(cli=None, env=None, config=None) -> PromptStyle:
    """Single-style convenience wrapper around resolve_prompt_styles."""
    return resolve_prompt_styles(cli, env, config)[0]


from .templates import (  # noqa: E402  (re-export)
    AttributeTemplate,
    build_templated_prompts,
    build_templated_query,
)

__all__ = [
    "PromptStyle", "coerce_style", "resolve_prompt_style", "resolve_prompt_styles",
    "AttributeTemplate", "build_templated_prompts", "build_templated_query",
    "ENV_VAR",
]
