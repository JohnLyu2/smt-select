"""
Logic- or family-specific rules for building the final description in extract_descriptions.

Define overrides and modifiers here. The extract script applies these after the default
description (or placeholder) is computed.
"""

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


# (logic, family) -> (smtlib_path: str) -> str | None. If not None, prefixed to description.
PREFIX_RULES: dict[tuple[str, str], object] = {
    ("UFNIA", "Preiner"): _preiner_prefix,
}


def apply_description_rule(logic: str, family: str, description: str, smtlib_path: str = "") -> str:
    """Apply logic/family-specific rules. Returns the (possibly modified) description."""
    key = (logic, family)
    if key in PREFIX_RULES:
        prefix = PREFIX_RULES[key](smtlib_path)
        if prefix is not None:
            return prefix + " " + description
    return description
