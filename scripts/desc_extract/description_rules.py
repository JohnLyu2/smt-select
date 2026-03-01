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
        "This SV-COMP benchmark checks unreachability of an error state"
        " and memory safety in a program that builds a list with alternating"
        " data values and checks its structure.",
    "cs_szymanski_true-unreach-call.i":
        "This SV-COMP benchmark checks unreachability of an error state"
        " in Szymanski's mutual exclusion algorithm.",
    "cs_time_var_mutex_true-unreach-call.i":
        "This SV-COMP benchmark checks unreachability of an error state"
        " in a time-variable mutual exclusion algorithm.",
    "dancing_true-unreach-call_false-valid-memtrack.i":
        "This SV-COMP benchmark checks unreachability of an error state"
        " in a dancing links program that removes and re-inserts a node"
        " in a doubly-linked list, where memory tracking is violated.",
    "dll-circular_false-unreach-call_false-valid-memcleanup.i":
        "This SV-COMP benchmark checks a circular doubly-linked list"
        " program where the error state is reachable and memory cleanup"
        " is violated.",
    "dll_of_dll_false-unreach-call_false-valid-memcleanup.i":
        "This SV-COMP benchmark checks a doubly-linked list of"
        " doubly-linked lists program where the error state is reachable"
        " and memory cleanup is violated.",
    "dll_of_dll_true-unreach-call_true-valid-memsafety.i":
        "This SV-COMP benchmark checks unreachability of an error state"
        " and memory safety in a doubly-linked list of doubly-linked"
        " lists program.",
    "ex3_forlist_true-termination.c_true-unreach-call.i":
        "This SV-COMP benchmark checks termination and unreachability"
        " of an error state in a for-loop list traversal program.",
    "list-ext_flag_false-unreach-call_false-valid-deref.i":
        "This SV-COMP benchmark checks a list program with an external"
        " flag where the error state is reachable and an invalid"
        " dereference exists.",
    "list_flag_true-unreach-call_false-valid-memtrack.i":
        "This SV-COMP benchmark checks unreachability of an error state"
        " in a program that builds a list whose data values depend on a"
        " flag and checks its structure, where memory tracking is violated.",
    "list_search_false-unreach-call_false-valid-memcleanup.i":
        "This SV-COMP benchmark checks a linked-list insert-and-search"
        " program where the error state is reachable and memory cleanup"
        " is violated.",
    "list_search_true-unreach-call_false-valid-memcleanup.i":
        "This SV-COMP benchmark checks unreachability of an error state"
        " in a linked-list insert-and-search program, where memory cleanup"
        " is violated.",
    "list_true-unreach-call_false-valid-memtrack.i":
        "This SV-COMP benchmark checks unreachability of an error state"
        " in a program that builds an integer list with a repeating"
        " pattern and checks its structure, where memory tracking"
        " is violated.",
    "s3_clnt.blast.01_false-unreach-call.i.cil.c":
        "This SV-COMP benchmark checks an SSL v3 client state-machine"
        " program (BLAST variant 01) where the error state is reachable.",
    "s3_clnt.blast.04_false-unreach-call.i.cil.c":
        "This SV-COMP benchmark checks an SSL v3 client state-machine"
        " program (BLAST variant 04) where the error state is reachable.",
    "sep10_true-unreach-call.i":
        "This SV-COMP benchmark checks unreachability of an error state"
        " in a pointer separation program (variant sep10).",
    "sep60_true-unreach-call.i":
        "This SV-COMP benchmark checks unreachability of an error state"
        " in a pointer separation program (variant sep60).",
    "simple_true-unreach-call_false-valid-memtrack.i":
        "This SV-COMP benchmark checks unreachability of an error state"
        " in a program that builds a simple integer list and traverses it"
        " to check its structure, where memory tracking is violated.",
    "sll-rb-sentinel_false-unreach-call_false-valid-memcleanup.i":
        "This SV-COMP benchmark checks a singly-linked list with sentinel"
        " nodes where the error state is reachable and memory cleanup is"
        " violated.",
    "sll-sorted_false-unreach-call_false-valid-memcleanup.i":
        "This SV-COMP benchmark checks a sorted singly-linked list"
        " program where the error state is reachable and memory cleanup"
        " is violated.",
    "sll_to_dll_rev_false-unreach-call_false-valid-memcleanup.i":
        "This SV-COMP benchmark checks a program that converts a"
        " singly-linked list to a doubly-linked list and reverses it,"
        " where the error state is reachable and memory cleanup"
        " is violated.",
    "splice_true-unreach-call_false-valid-memtrack.i":
        "This SV-COMP benchmark checks unreachability of an error state"
        " in a program that splits a list into two sublists by position"
        " and checks each, where memory tracking is violated.",
    "standard_running_true-unreach-call.i":
        "This SV-COMP benchmark checks unreachability of an error state"
        " in a standard running example program.",
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
