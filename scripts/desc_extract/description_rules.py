"""
Logic- or family-specific rules for building the final description in extract_descriptions.

Define overrides and modifiers here. The extract script applies these after the default
description (or placeholder) is computed.
"""

import json
import re
from pathlib import Path

_DESC_EXTRACT_DIR = Path(__file__).resolve().parent
with open(_DESC_EXTRACT_DIR / "ultimate_svcomp2019_prefixes.json", encoding="utf-8") as f:
    _ULTIMATE_SVCOMP2019_PREFIXES: dict[str, str] = json.load(f)
with open(_DESC_EXTRACT_DIR / "ultimate_svcomp2023_prefixes.json", encoding="utf-8") as f:
    _ULTIMATE_SVCOMP2023_PREFIXES: dict[str, str] = json.load(f)

_ULTIMATE_SVCOMP2019_INSERT_AFTER = "unsatisfiable cores and interpolants."
_ULTIMATE_SVCOMP2023_INSERT_AFTER = "unsatisfiable cores, and interpolants."


def _ultimate_svcomp_modify(
    prefixes: dict[str, str],
    insert_after: str,
    smtlib_path: str,
    description: str,
) -> str | None:
    """Insert a per-program-type sentence after the given anchor in the description."""
    filename = smtlib_path.rsplit("/", 1)[-1]
    prefix = re.sub(r"_\d+\.smt2$", "", filename)
    sentence = prefixes.get(prefix)
    if sentence is None:
        return None
    idx = description.find(insert_after)
    if idx == -1:
        return sentence + " " + description
    insert_pos = idx + len(insert_after)
    return description[:insert_pos] + "\n" + sentence + description[insert_pos:]


def _ultimate_svcomp2019_modify(smtlib_path: str, description: str) -> str | None:
    return _ultimate_svcomp_modify(
        _ULTIMATE_SVCOMP2019_PREFIXES,
        _ULTIMATE_SVCOMP2019_INSERT_AFTER,
        smtlib_path,
        description,
    )


def _ultimate_svcomp2023_modify(smtlib_path: str, description: str) -> str | None:
    return _ultimate_svcomp_modify(
        _ULTIMATE_SVCOMP2023_PREFIXES,
        _ULTIMATE_SVCOMP2023_INSERT_AFTER,
        smtlib_path,
        description,
    )


# ---------------------------------------------------------------------------
# UFNIA / Preiner
# ---------------------------------------------------------------------------
# Path structure for UFNIA Preiner: UFNIA/2019-Preiner/<subfolder>/<name>.smt2
# - name starts with f2 -> "Depth-2 formula rewrite", t3 -> "Depth-3 term rewrite"
# - subfolder: combined, full, partial, qf -> "combined", "full", "partial", "quantifier-free"
_PREINER_DEPTH = {"f2": "Depth-2 formula rewrite", "t3": "Depth-3 term rewrite"}
_PREINER_AXIOM = {"combined": "combined", "full": "full", "partial": "partial", "qf": "quantifier-free"}


def _preiner_prefix(smtlib_path: str) -> str | None:
    """Return prefix line for UFNIA Preiner from path, or None to skip."""
    parts = smtlib_path.split("/")
    if len(parts) < 4:
        return None
    subfolder = parts[2]
    filename = parts[-1]
    base = filename.rsplit(".", 1)[0] if "." in filename else filename
    prefix = base[:2]
    depth = _PREINER_DEPTH.get(prefix)
    axiom = _PREINER_AXIOM.get(subfolder)
    if depth is None or axiom is None:
        return None
    return f"{depth} benchmark using {axiom} axiomatization."


def _preiner_modify(smtlib_path: str, description: str) -> str | None:
    """Prepend depth/axiomatization info for UFNIA Preiner."""
    prefix = _preiner_prefix(smtlib_path)
    if prefix is None:
        return None
    return prefix + " " + description


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# (logic, family) -> (smtlib_path: str, description: str) -> str | None.
# If not None, the returned string replaces the description.
DESCRIPTION_RULES: dict[tuple[str, str], object] = {
    ("UFNIA", "Preiner"): _preiner_modify,
    ("ABV", "UltimateAutomizerSvcomp2019"): _ultimate_svcomp2019_modify,
    ("ABV", "UltimateAutomizerSvcomp2023"): _ultimate_svcomp2023_modify,
}


def apply_description_rule(logic: str, family: str, description: str, smtlib_path: str = "") -> str:
    """Apply logic/family-specific rules. Returns the (possibly modified) description."""
    key = (logic, family)
    if key in DESCRIPTION_RULES:
        result = DESCRIPTION_RULES[key](smtlib_path, description)
        if result is not None:
            return result
    return description
