"""
Logic- or family-specific rules for building the final description in extract_descriptions.

Define overrides and modifiers here. The extract script applies these after the default
description (or placeholder) is computed.
"""

import re

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


# ---------------------------------------------------------------------------
# ABV / UltimateAutomizerSvcomp2019
# ---------------------------------------------------------------------------
# Benchmark names follow the pattern <program>_<number>.smt2.  The prefix
# (everything before the trailing _<digits>.smt2) identifies the SV-COMP
# verification task.
_ULTIMATE_SVCOMP2019_PREFIXES: dict[str, str] = {
    "alternating_list_true-unreach-call_true-valid-memsafety.i":
        "The original SV-COMP benchmark checks unreachability and memory safety"
        " in an alternating list program.",
    "cs_szymanski_true-unreach-call.i":
        "The original SV-COMP benchmark checks unreachability in Szymanski's"
        " mutual exclusion algorithm.",
    "cs_time_var_mutex_true-unreach-call.i":
        "The original SV-COMP benchmark checks unreachability in a mutual"
        " exclusion algorithm.",
    "dancing_true-unreach-call_false-valid-memtrack.i":
        "The original SV-COMP benchmark checks unreachability and memory tracking"
        " in a dancing links program.",
    "dll-circular_false-unreach-call_false-valid-memcleanup.i":
        "The original SV-COMP benchmark checks unreachability and memory cleanup"
        " in a circular doubly-linked list program.",
    "dll_of_dll_false-unreach-call_false-valid-memcleanup.i":
        "The original SV-COMP benchmark checks unreachability and memory cleanup"
        " in a doubly-linked list of doubly-linked lists program.",
    "dll_of_dll_true-unreach-call_true-valid-memsafety.i":
        "The original SV-COMP benchmark checks unreachability and memory safety"
        " in a doubly-linked list of doubly-linked lists program.",
    "ex3_forlist_true-termination.c_true-unreach-call.i":
        "The original SV-COMP benchmark checks termination and unreachability"
        " in a pointer state-tracking program.",
    "list-ext_flag_false-unreach-call_false-valid-deref.i":
        "The original SV-COMP benchmark checks unreachability and memory safety"
        " in an extended list program where each node stores its own flag.",
    "list_flag_true-unreach-call_false-valid-memtrack.i":
        "The original SV-COMP benchmark checks unreachability and memory tracking"
        " in a flag-controlled list program.",
    "list_search_false-unreach-call_false-valid-memcleanup.i":
        "The original SV-COMP benchmark checks unreachability and memory cleanup"
        " in a linked-list search program.",
    "list_search_true-unreach-call_false-valid-memcleanup.i":
        "The original SV-COMP benchmark checks unreachability and memory cleanup"
        " in a linked-list search program.",
    "list_true-unreach-call_false-valid-memtrack.i":
        "The original SV-COMP benchmark checks unreachability and memory tracking"
        " in an integer list program.",
    "s3_clnt.blast.01_false-unreach-call.i.cil.c":
        "The original SV-COMP benchmark checks unreachability in an SSL v3"
        " client state-machine program (BLAST variant 01).",
    "s3_clnt.blast.04_false-unreach-call.i.cil.c":
        "The original SV-COMP benchmark checks unreachability in an SSL v3"
        " client state-machine program (BLAST variant 04).",
    "sep10_true-unreach-call.i":
        "The original SV-COMP benchmark checks unreachability in a"
        " reducer commutativity program (array size 10).",
    "sep60_true-unreach-call.i":
        "The original SV-COMP benchmark checks unreachability in a"
        " reducer commutativity program (array size 60).",
    "simple_true-unreach-call_false-valid-memtrack.i":
        "The original SV-COMP benchmark checks unreachability and memory tracking"
        " in a simple list program.",
    "sll-rb-sentinel_false-unreach-call_false-valid-memcleanup.i":
        "The original SV-COMP benchmark checks unreachability and memory cleanup"
        " in a singly-linked list program with red-black colored nodes"
        " and a sentinel node.",
    "sll-sorted_false-unreach-call_false-valid-memcleanup.i":
        "The original SV-COMP benchmark checks unreachability and memory cleanup"
        " in a sorted singly-linked list program.",
    "sll_to_dll_rev_false-unreach-call_false-valid-memcleanup.i":
        "The original SV-COMP benchmark checks unreachability and memory cleanup"
        " in a program that builds a singly-linked list, converts it to a"
        " doubly-linked list, and reverses it.",
    "splice_true-unreach-call_false-valid-memtrack.i":
        "The original SV-COMP benchmark checks unreachability and memory tracking"
        " in a list-splitting program.",
    "standard_running_true-unreach-call.i":
        "The original SV-COMP benchmark checks unreachability in a standard"
        " running example program.",
}


_ULTIMATE_SVCOMP2019_INSERT_AFTER = "unsatisfiable cores and interpolants."


def _ultimate_svcomp2019_modify(smtlib_path: str, description: str) -> str | None:
    """Insert a per-program-type sentence after the interpolants line."""
    filename = smtlib_path.rsplit("/", 1)[-1]
    prefix = re.sub(r"_\d+\.smt2$", "", filename)
    sentence = _ULTIMATE_SVCOMP2019_PREFIXES.get(prefix)
    if sentence is None:
        return None
    idx = description.find(_ULTIMATE_SVCOMP2019_INSERT_AFTER)
    if idx == -1:
        return sentence + " " + description
    insert_pos = idx + len(_ULTIMATE_SVCOMP2019_INSERT_AFTER)
    return description[:insert_pos] + "\n" + sentence + description[insert_pos:]


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
}


def apply_description_rule(logic: str, family: str, description: str, smtlib_path: str = "") -> str:
    """Apply logic/family-specific rules. Returns the (possibly modified) description."""
    key = (logic, family)
    if key in DESCRIPTION_RULES:
        result = DESCRIPTION_RULES[key](smtlib_path, description)
        if result is not None:
            return result
    return description
