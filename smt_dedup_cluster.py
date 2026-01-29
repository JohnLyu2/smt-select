#!/usr/bin/env python3
"""
smt_reduce_v2.py — Scalable SMT-LIB v2 summarizer / reducer (schema + unitization)

Design goals
------------
- Works across diverse SMT2 styles:
  * many small asserts
  * few large asserts
  * one gigantic assert (hundreds of MB)
  * heavy let usage / bit-blasting / CNF-ish
  * UF / arithmetic / BV / strings / quantifiers / unknown logics
- Produces a human/LLM-friendly summary (NOT equisatisfiable).
- Uses Z3 opportunistically (size-capped + time-budgeted); otherwise a streaming token backend.

Key idea
--------
1) Extract blocks textually (assert / declare-fun / define-fun + structural).
2) Unitize assertions:
   - small assert => one Unit
   - huge assert => split top-level (and ...) / (or ...) / let-body if safe
   - otherwise => fallback to sampled windows across the assert text
3) Extract a schema fingerprint per Unit:
   - CoreKey: sort profile + operator profile + quant profile + arith profile + bv profile + size profile
   - Tags: name buckets (grounding abstraction), literal hints, parse reasons
4) Cluster by CoreKey and print representatives under a budget.

Usage
-----
  python smt_reduce_v2.py input.smt2 -o reduced_v2.smt2
  python smt_reduce_v2.py input.smt2 -o reduced_v2.smt2 --z3-seconds 5 --assert-budget 40
  python smt_reduce_v2.py input.smt2 -o reduced_v2.smt2 --huge-assert-mb 5 --z3-unit-kb 200

Notes
-----
- Output is meant for inspection/summarization; do not feed back to solvers expecting equivalence.
"""

import argparse
import math
import re
import time
from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Iterable, Any

import z3


# ============================================================
# Defaults
# ============================================================

DEFAULT_Z3_SECONDS = 10.0
DEFAULT_ASSERT_BUDGET = 20
DEFAULT_KEEP_RARE_TOTAL = 5
DEFAULT_RARE_THRESHOLD = 2
DEFAULT_REPS_PER_CLUSTER = 1

DEFAULT_HUGE_ASSERT_MB = 5          # > this => attempt unitization/splitting/windows
DEFAULT_Z3_UNIT_KB = 200            # parse only units <= this size in KB
DEFAULT_MAX_UNITS = 50_000          # hard cap for safety
DEFAULT_WINDOW_COUNT = 5            # fallback windows per huge assert
DEFAULT_WINDOW_BYTES = 200_000      # bytes per window when sampling (fallback)


# ============================================================
# Robust block extraction (textual, balanced by parentheses)
# ============================================================

_HEAD_RE_CACHE: Dict[str, re.Pattern] = {}

def _head_pat(head: str) -> re.Pattern:
    if head not in _HEAD_RE_CACHE:
        _HEAD_RE_CACHE[head] = re.compile(r"\(\s*" + re.escape(head) + r"(\s|\)|$)")
    return _HEAD_RE_CACHE[head]


def split_into_sessions(text: str) -> List[str]:
    parts = re.split(r"\(\s*exit\s*\)\s*", text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if p.strip()]


def extract_blocks_from_text(text: str, head: str) -> Tuple[List[str], List[str]]:
    pat = _head_pat(head)
    structural_lines: List[str] = []
    blocks: List[str] = []

    buf: List[str] = []
    depth = 0
    in_block = False

    for line in text.splitlines(keepends=True):
        lstrip = line.lstrip()

        if not in_block and pat.search(lstrip):
            in_block = True
            buf = [line]
            depth = lstrip.count("(") - lstrip.count(")")
            if depth <= 0:
                blocks.append("".join(buf))
                buf, depth, in_block = [], 0, False
            continue

        if in_block:
            buf.append(line)
            depth += lstrip.count("(") - lstrip.count(")")
            if depth <= 0:
                blocks.append("".join(buf))
                buf, depth, in_block = [], 0, False
            continue

        structural_lines.append(line)

    if in_block and buf:
        structural_lines.extend(buf)

    return structural_lines, blocks


def extract_session_components(session_text: str) -> dict:
    structural_after_asserts, asserts = extract_blocks_from_text(session_text, "assert")
    structural_after_decl, decls = extract_blocks_from_text("".join(structural_after_asserts), "declare-fun")
    structural_after_def, defs = extract_blocks_from_text("".join(structural_after_decl), "define-fun")
    structural_text = "".join(structural_after_def)
    return {
        "structural_text": structural_text,
        "decls": decls,
        "defs": defs,
        "asserts": asserts,
    }


def choose_best_session(full_text: str) -> str:
    sessions = split_into_sessions(full_text)
    best = sessions[0] if sessions else full_text
    best_n = -1
    for s in sessions:
        _, asserts = extract_blocks_from_text(s, "assert")
        n = len(asserts)
        if n >= best_n:
            best_n = n
            best = s
    return best


def strip_trailing_commands(structural_text: str) -> str:
    lines = structural_text.splitlines(keepends=True)
    out = []
    for ln in lines:
        l = ln.strip().lower()
        if l.startswith("(check-sat") or l.startswith("(exit"):
            continue
        out.append(ln)
    return "".join(out)


def compress_blank_lines(text: str, max_consecutive: int = 2) -> str:
    out = []
    blank_run = 0
    for ln in text.splitlines(keepends=True):
        if ln.strip() == "":
            blank_run += 1
            if blank_run <= max_consecutive:
                out.append(ln)
        else:
            blank_run = 0
            out.append(ln)
    return "".join(out)


# ============================================================
# Grounding abstraction / name bucketing (extend as needed)
# ============================================================

_OCCURS_RE = re.compile(r"^occurs\(.+,\s*\d+\)\s*$")
_HOLDS_RE = re.compile(r"^holds\(.+,\s*\d+\)\s*$")
_HOLD_S_RE = re.compile(r"^hold_s\(.+,\s*\d+\)\s*$")
_STEP_RE = re.compile(r"^step\(.+,\s*\d+\)\s*$")
_INSTOCCURS_RE = re.compile(r"^instoccurs\(.+,\s*\d+\)\s*$")
_REQUIRED_RE = re.compile(r"^required\(.+\)\s*$")
_CSPVAR_RE = re.compile(r"^cspvar\(.+\)\s*$")

def normalize_symbol(sym: str) -> str:
    if sym.startswith("|") and sym.endswith("|") and len(sym) >= 2:
        sym = sym[1:-1]
    s = sym.strip()
    if s.isdigit():
        return "AUX"
    if _OCCURS_RE.match(s) or s.startswith("occurs("):
        return "OCCURS@t"
    if _HOLDS_RE.match(s) or s.startswith("holds("):
        return "HOLDS@t"
    if _HOLD_S_RE.match(s) or s.startswith("hold_s("):
        return "HOLD_S@t"
    if _STEP_RE.match(s) or s.startswith("step("):
        return "STEP@t"
    if _INSTOCCURS_RE.match(s) or s.startswith("instoccurs("):
        return "INSTOCCURS@t"
    if _REQUIRED_RE.match(s) or s.startswith("required("):
        return "REQUIRED"
    if _CSPVAR_RE.match(s) or s.startswith("cspvar("):
        return "CSPVAR"
    if s.startswith("ezcsp__"):
        return "CSP_OP"
    return s


# ============================================================
# Declarations summarization
# ============================================================

def parse_declare_fun_signature(block: str) -> Tuple:
    s = block.strip()
    m = re.search(r"\(\s*declare-fun\s+([^\s()]+)", s)
    if not m:
        return ("UNKNOWN",)

    raw_name = m.group(1)
    name_bucket = normalize_symbol(raw_name)

    m2 = re.search(r"\(\s*declare-fun\s+[^\s()]+\s*\((.*?)\)\s*([^\s()]+)\s*\)", s, re.DOTALL)
    if not m2:
        return ("UNKNOWN", name_bucket)

    args_blob = m2.group(1).strip()
    ret_sort = m2.group(2).strip()
    arg_sorts = tuple(a for a in args_blob.split() if a)
    arity = len(arg_sorts)
    return (arity, arg_sorts, ret_sort, name_bucket)


# ============================================================
# Unitization
# ============================================================

@dataclass(frozen=True)
class Unit:
    origin_assert_index: int
    unit_index_within_assert: int
    unit_kind: str  # "ASSERT" | "SPLIT_CHILD" | "WINDOW"
    unit_text: str
    notes: str = ""


def _skip_ws(s: str, i: int) -> int:
    n = len(s)
    while i < n and s[i].isspace():
        i += 1
    return i


def _read_symbol(s: str, i: int) -> Tuple[str, int]:
    """Read an SMT symbol starting at i (assumes i at non-ws)."""
    n = len(s)
    if i >= n:
        return "", i
    if s[i] == "|":
        j = i + 1
        while j < n and s[j] != "|":
            j += 1
        return s[i:j+1] if j < n else s[i:], min(n, j+1)
    # regular symbol
    j = i
    while j < n and (not s[j].isspace()) and s[j] not in "()":
        j += 1
    return s[i:j], j


def extract_assert_body(assert_text: str) -> Optional[str]:
    """
    Given a balanced '(assert ...)' block, return the body '...' as a string.
    Returns None on failure.
    """
    s = assert_text.strip()
    if not s.startswith("("):
        return None
    # find "(assert"
    m = re.match(r"\(\s*assert\b", s, flags=re.IGNORECASE)
    if not m:
        return None
    i = m.end()
    i = _skip_ws(s, i)
    # body extends until the final ')' of the assert expression
    # because s is trimmed and should end with ')'
    if not s.endswith(")"):
        return None
    body = s[i:-1].strip()
    return body if body else None


def peel_bang_wrapper(expr: str) -> Tuple[str, bool]:
    """
    If expr is of form '(! phi ...)' return phi and flag True.
    Otherwise return expr and False.
    """
    t = expr.strip()
    if not t.startswith("("):
        return expr, False
    # head symbol
    i = 1
    i = _skip_ws(t, i)
    sym, j = _read_symbol(t, i)
    if sym == "!":
        # very conservative: extract first argument 'phi'
        # after '(!' read next expr at same depth
        k = _skip_ws(t, j)
        phi, end = extract_first_child_expr(t, k)
        if phi:
            return phi.strip(), True
    return expr, False


def peek_head_symbol(expr: str) -> Optional[str]:
    t = expr.strip()
    if not t.startswith("("):
        return None
    i = 1
    i = _skip_ws(t, i)
    sym, _ = _read_symbol(t, i)
    return sym.lower() if sym else None


def extract_first_child_expr(s: str, i: int) -> Tuple[Optional[str], int]:
    """
    Extract the first S-expression or atom starting at i.
    Returns (text, new_index).
    """
    n = len(s)
    i = _skip_ws(s, i)
    if i >= n:
        return None, i
    if s[i] == "(":
        depth = 0
        in_str = False
        j = i
        while j < n:
            c = s[j]
            if c == '"' and (j == 0 or s[j-1] != "\\"):
                in_str = not in_str
            if not in_str:
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                    if depth == 0:
                        return s[i:j+1], j+1
            j += 1
        return s[i:], n
    # atom
    sym, j = _read_symbol(s, i)
    return sym, j


def split_top_level_children(expr: str, head: str) -> Optional[List[str]]:
    """
    If expr is '(head child1 child2 ...)', return children (as strings),
    where children are immediate arguments of the head.
    Returns None if expr does not match.
    """
    t = expr.strip()
    if not t.startswith("("):
        return None
    h = peek_head_symbol(t)
    if h != head.lower():
        return None

    # scan after "(head"
    i = 1
    i = _skip_ws(t, i)
    _, i = _read_symbol(t, i)  # consume head
    children: List[str] = []
    n = len(t)

    while True:
        i = _skip_ws(t, i)
        if i >= n:
            break
        if t[i] == ")":
            break
        child, i2 = extract_first_child_expr(t, i)
        if child is None:
            break
        children.append(child)
        i = i2

    return children


def sample_evenly(items: List[str], k: int) -> List[str]:
    if k <= 0 or not items:
        return []
    if len(items) <= k:
        return items
    # pick evenly spaced indices including near ends
    idxs = []
    for j in range(k):
        idx = round(j * (len(items) - 1) / (k - 1)) if k > 1 else 0
        idxs.append(int(idx))
    # dedupe while preserving order
    seen = set()
    out = []
    for idx in idxs:
        if idx not in seen:
            seen.add(idx)
            out.append(items[idx])
    return out


def make_units_from_assert(
    assert_text: str,
    assert_index: int,
    huge_assert_bytes: int,
    max_units: int,
    window_count: int,
    window_bytes: int,
) -> List[Unit]:
    units: List[Unit] = []

    if len(assert_text.encode("utf-8", errors="ignore")) <= huge_assert_bytes:
        units.append(Unit(assert_index, 0, "ASSERT", assert_text.strip() + "\n"))
        return units

    body = extract_assert_body(assert_text)
    if body is None:
        return make_window_units(assert_text, assert_index, window_count, window_bytes, note="no_body")

    body2, had_bang = peel_bang_wrapper(body)
    head = peek_head_symbol(body2)

    # Rule S1: top-level (and ...)
    if head == "and":
        kids = split_top_level_children(body2, "and") or []
        if not kids:
            return make_window_units(assert_text, assert_index, window_count, window_bytes, note="and_no_kids")
        # cap and sample if needed
        chosen = kids[:max_units] if len(kids) <= max_units else sample_evenly(kids, min(max_units, 3000))
        for j, ktxt in enumerate(chosen):
            wrapped = f"(assert {ktxt})\n"
            note = "split_and" + (" bang_peeled" if had_bang else "")
            units.append(Unit(assert_index, j, "SPLIT_CHILD", wrapped, notes=note))
        return units

    # Rule S2: top-level (or ...)
    if head == "or":
        kids = split_top_level_children(body2, "or") or []
        if not kids:
            return make_window_units(assert_text, assert_index, window_count, window_bytes, note="or_no_kids")
        # disjunctions can be huge; sample more aggressively
        chosen = kids[:max_units] if len(kids) <= max_units else sample_evenly(kids, min(max_units, 1000))
        for j, ktxt in enumerate(chosen):
            wrapped = f"(assert {ktxt})\n"
            note = "split_or" + (" bang_peeled" if had_bang else "")
            units.append(Unit(assert_index, j, "SPLIT_CHILD", wrapped, notes=note))
        return units

    # Rule S3: top-level (let ...)
    if head == "let":
        # Conservative let handling:
        # If body is (let (...) BODY), try to split BODY if it is (and/or ...)
        # We do not attempt to parse bindings deeply; we just locate the last child (BODY).
        let_body = extract_let_body(body2)
        if let_body:
            let_body2, had_bang2 = peel_bang_wrapper(let_body)
            h2 = peek_head_symbol(let_body2)
            if h2 in ("and", "or"):
                kids = split_top_level_children(let_body2, h2) or []
                if kids:
                    chosen = kids[:max_units] if len(kids) <= max_units else sample_evenly(kids, min(max_units, 3000 if h2=="and" else 1000))
                    for j, ktxt in enumerate(chosen):
                        # wrap each kid back under original let (bindings kept intact)
                        wrapped = f"(assert (let {extract_let_bindings_blob(body2)} {ktxt}))\n"
                        note = f"split_let_{h2}" + (" bang_peeled" if had_bang or had_bang2 else "")
                        units.append(Unit(assert_index, j, "SPLIT_CHILD", wrapped, notes=note))
                    return units

        # If let is huge but can't be safely split, window it.
        return make_window_units(assert_text, assert_index, window_count, window_bytes, note="let_fallback")

    # Otherwise: fallback windows
    return make_window_units(assert_text, assert_index, window_count, window_bytes, note=f"fallback_head={head}")


def extract_let_body(expr: str) -> Optional[str]:
    """
    Very conservative: for '(let (bindings) body)', return 'body' substring.
    """
    t = expr.strip()
    if peek_head_symbol(t) != "let":
        return None
    # consume "(let"
    i = 1
    i = _skip_ws(t, i)
    _, i = _read_symbol(t, i)
    i = _skip_ws(t, i)
    # next should be bindings S-expr
    bindings, i2 = extract_first_child_expr(t, i)
    if not bindings:
        return None
    i = _skip_ws(t, i2)
    body, _ = extract_first_child_expr(t, i)
    return body.strip() if body else None


def extract_let_bindings_blob(expr: str) -> str:
    """
    Return the bindings blob of a let: for '(let BINDINGS BODY)' return 'BINDINGS'
    If unavailable, return '()' as safe placeholder (keeps syntax valid-ish for printing).
    """
    t = expr.strip()
    if peek_head_symbol(t) != "let":
        return "()"
    i = 1
    i = _skip_ws(t, i)
    _, i = _read_symbol(t, i)  # let
    i = _skip_ws(t, i)
    bindings, _ = extract_first_child_expr(t, i)
    return bindings.strip() if bindings else "()"


def make_window_units(assert_text: str, assert_index: int, window_count: int, window_bytes: int, note: str) -> List[Unit]:
    b = assert_text.encode("utf-8", errors="ignore")
    if not b:
        return [Unit(assert_index, 0, "WINDOW", assert_text.strip() + "\n", notes=f"window_empty {note}")]

    n = len(b)
    k = max(1, window_count)
    win = min(window_bytes, n)

    # start positions: 0, 25%, 50%, 75%, end-win
    starts = []
    for j in range(k):
        if k == 1:
            pos = 0
        else:
            frac = j / (k - 1)
            pos = int(frac * max(0, n - win))
        starts.append(pos)

    # dedupe
    starts = list(dict.fromkeys(starts))

    out = []
    for j, st in enumerate(starts):
        chunk = b[st: st + win].decode("utf-8", errors="ignore")
        # Keep a small wrapper comment and raw chunk (not necessarily valid SMT)
        out.append(Unit(assert_index, j, "WINDOW", chunk, notes=f"window {note}"))
    return out


# ============================================================
# Feature extraction backends
# ============================================================

SORT_FAMILY_ORDER = ["BOOL", "INT", "REAL", "BV", "STRING", "DATATYPE", "ARRAY", "FP", "UNINTERP", "OTHER"]

def bytes_bucket(nbytes: int) -> int:
    # 0..7 representing ranges
    # <1KB, <10KB, <100KB, <1MB, <5MB, <20MB, <100MB, >=100MB
    limits = [1_000, 10_000, 100_000, 1_000_000, 5_000_000, 20_000_000, 100_000_000]
    for i, lim in enumerate(limits):
        if nbytes < lim:
            return i
    return len(limits)

def int_bucket(cnt: int, step: int = 5, cap: int = 50) -> int:
    return min(cap, (cnt // step) * step)

def node_bucket(nodes: int, step: int = 100, cap: int = 2000) -> int:
    return min(cap, (nodes // step) * step)

def depth_bucket(d: int, cap: int = 50) -> int:
    return min(cap, d)


# Operator normalization vocab (expand as needed)
OP_MAP = {
    "and": "AND", "or": "OR", "not": "NOT", "=>": "IMPLIES", "implies": "IMPLIES", "xor": "XOR", "ite": "ITE",
    "=": "EQ", "distinct": "DISTINCT",
    "<": "LT", "<=": "LE", ">": "GT", ">=": "GE",
    "+": "ADD", "-": "SUB", "*": "MUL", "div": "DIV", "mod": "MOD",
    "bvand": "BVAND", "bvor": "BVOR", "bvxor": "BVXOR", "bvnot": "BVNOT",
    "bvadd": "BVADD", "bvsub": "BVSUB", "bvmul": "BVMUL",
    "bvshl": "BVSHL", "bvlshr": "BVLSHR", "bvashr": "BVASHR",
    "concat": "BVCONCAT",  # used both in strings & bv; we disambiguate via tags sometimes
    "extract": "BVEXTRACT",
    "zero_extend": "BVZEROEXT", "sign_extend": "BVSIGNEXT",
    "bvult": "BVULT", "bvule": "BVULE", "bvugt": "BVUGT", "bvuge": "BVUGE",
    "bvslt": "BVSLT", "bvsle": "BVSLE", "bvsgt": "BVSGT", "bvsge": "BVSGE",
    "forall": "FORALL", "exists": "EXISTS",
    # strings (SMT-LIB uses str. prefix)
    "str.len": "STR.LEN", "str.++": "STR.CONCAT", "str.contains": "STR.CONTAINS", "str.prefixof": "STR.PREFIXOF",
    "str.suffixof": "STR.SUFFIXOF", "str.in.re": "STR.IN.RE", "str.to.re": "STR.TO.RE",
}

BOOL_OPS = {"AND","OR","NOT","IMPLIES","ITE","XOR"}
REL_OPS  = {"EQ","DISTINCT","LT","LE","GT","GE"}
ARITH_OPS= {"ADD","SUB","MUL","DIV","MOD"}
BV_OPS   = {o for o in OP_MAP.values() if o.startswith("BV")}
STR_OPS  = {o for o in OP_MAP.values() if o.startswith("STR.")}

def normalize_op(sym: str) -> str:
    s = sym.strip()
    if s.startswith("|") and s.endswith("|"):
        s = s[1:-1]
    s = s.lower()
    return OP_MAP.get(s, "APPLY")


@dataclass
class Features:
    # CoreKey parts
    sort_families: Tuple[str, ...]
    bv_widths: Tuple[int, ...]
    has_arrays: bool
    has_strings: bool

    top_ops: Tuple[str, ...]
    op_buckets: Tuple[Tuple[str, int], ...]
    has_let: bool

    has_quantifiers: bool
    quant_kinds: Tuple[str, ...]
    max_quant_nesting: int
    has_patterns: bool

    uses_int: bool
    uses_real: bool
    is_nonlinear: str  # "NO"|"YES"|"MAYBE"
    has_div_or_mod: bool

    has_bv: bool
    has_extract: bool
    has_concat: bool
    has_shifts: bool
    has_bv_arith: bool
    has_bv_rel: bool

    bytes_bucket: int
    node_bucket: int
    max_depth_bucket: int

    # Tags (not in CoreKey)
    name_buckets: Counter
    literal_profile: Counter
    backend: str
    backend_note: str

    def core_key(self) -> Tuple:
        sort_profile = (
            tuple(sorted(set(self.sort_families), key=lambda x: SORT_FAMILY_ORDER.index(x) if x in SORT_FAMILY_ORDER else 999)),
            tuple(sorted(set(self.bv_widths))),
            self.has_arrays,
            self.has_strings,
        )
        op_profile = (self.top_ops, self.op_buckets, self.has_let)
        quant_profile = (
            self.has_quantifiers,
            self.quant_kinds,
            depth_bucket(self.max_quant_nesting, cap=10),
            self.has_patterns,
        )
        arith_profile = (self.uses_int, self.uses_real, self.is_nonlinear, self.has_div_or_mod)
        bv_profile = (self.has_bv, tuple(sorted(set(self.bv_widths))), self.has_extract, self.has_concat, self.has_shifts, self.has_bv_arith, self.has_bv_rel)
        size_profile = (self.bytes_bucket, self.node_bucket, self.max_depth_bucket)
        return ("S2", sort_profile, op_profile, quant_profile, arith_profile, bv_profile, size_profile)


# ----------------------------
# Token backend (stream-ish)
# ----------------------------

BV_WIDTH_RE = re.compile(r"\(\s*_\s*BitVec\s+(\d+)\s*\)", re.IGNORECASE)
SORT_TOKENS = {
    "bool": "BOOL",
    "int": "INT",
    "real": "REAL",
    "string": "STRING",
    "array": "ARRAY",
}

def token_backend_features(text: str, unit_kind: str, notes: str) -> Features:
    b = text.encode("utf-8", errors="ignore")
    nbytes = len(b)
    bb = bytes_bucket(nbytes)

    # quick scans
    has_let = bool(re.search(r"\(\s*let\b", text, flags=re.IGNORECASE))
    has_patterns = ":pattern" in text
    has_quant = bool(re.search(r"\(\s*(forall|exists)\b", text, flags=re.IGNORECASE))

    # sort families
    families = set()
    has_arrays = False
    has_strings = False
    for tok, fam in SORT_TOKENS.items():
        if re.search(r"\b" + re.escape(tok) + r"\b", text, flags=re.IGNORECASE):
            families.add(fam)
            if fam == "ARRAY":
                has_arrays = True
            if fam == "STRING":
                has_strings = True

    bv_widths = set(int(m.group(1)) for m in BV_WIDTH_RE.finditer(text))
    if bv_widths:
        families.add("BV")

    # operator counts (very approximate): count occurrences of "(sym"
    # This is robust enough for schema clustering across diverse styles.
    op_counts = Counter()
    # Match "(<symbol>" where symbol can be |...| or typical token
    for m in re.finditer(r"\(\s*([^\s()]+)", text):
        sym = m.group(1)
        op = normalize_op(sym)
        op_counts[op] += 1

    # name bucket counts: bar symbols are often grounding-heavy, but also count raw tokens that look like occurs(...).
    name_buckets = Counter()
    for m in re.finditer(r"\|[^|]*\|", text):
        name_buckets[normalize_symbol(m.group(0))] += 1
    # some encodings don't use bars; still catch obvious ones
    for m in re.finditer(r"\b(occurs|holds|hold_s|step|instoccurs|required|cspvar)\(", text):
        name_buckets[normalize_symbol(m.group(0)[:-1] + ")")] += 1

    # literal profile (coarse)
    literal_profile = Counter()
    literal_profile["NUM"] += len(re.findall(r"\b\d+\b", text))
    literal_profile["STR"] += len(re.findall(r"\"(?:\\.|[^\"])*\"", text))
    literal_profile["ZERO"] += len(re.findall(r"\b0\b", text))
    literal_profile["ONE"] += len(re.findall(r"\b1\b", text))

    # arithmetic / BV hints
    uses_int = "INT" in families
    uses_real = "REAL" in families

    has_div_or_mod = op_counts["DIV"] > 0 or op_counts["MOD"] > 0
    # nonlinear: MAYBE if MUL appears (token backend can't reliably tell const-vs-var)
    is_nonlinear = "MAYBE" if op_counts["MUL"] > 0 else "NO"

    has_bv = "BV" in families or any(o in BV_OPS for o in op_counts)
    has_extract = op_counts["BVEXTRACT"] > 0
    has_concat = op_counts["BVCONCAT"] > 0 or op_counts["STR.CONCAT"] > 0
    has_shifts = op_counts["BVSHL"] > 0 or op_counts["BVLSHR"] > 0 or op_counts["BVASHR"] > 0
    has_bv_arith = any(op_counts[o] > 0 for o in ("BVADD","BVSUB","BVMUL"))
    has_bv_rel = any(op_counts[o] > 0 for o in ("BVULT","BVULE","BVUGT","BVUGE","BVSLT","BVSLE","BVSGT","BVSGE"))

    # quant nesting approximation: count max paren depth between forall/exists; too expensive precisely.
    max_quant_nesting = 1 if has_quant else 0
    quant_kinds = []
    if re.search(r"\(\s*forall\b", text, flags=re.IGNORECASE):
        quant_kinds.append("FORALL")
    if re.search(r"\(\s*exists\b", text, flags=re.IGNORECASE):
        quant_kinds.append("EXISTS")
    quant_kinds = tuple(sorted(set(quant_kinds)))

    # size/depth approx via paren depth scan (fast)
    depth = 0
    max_depth = 0
    in_str = False
    for ch in text:
        if ch == '"' :
            in_str = not in_str
        if in_str:
            continue
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth = max(0, depth - 1)
    max_depth_b = depth_bucket(max_depth, cap=50)

    # node estimate: proportional to number of '('
    nodes_est = text.count("(")
    nb = node_bucket(nodes_est, step=100, cap=2000)

    # operator buckets by category
    cat_counts = {
        "BOOL_OPS": sum(op_counts[o] for o in BOOL_OPS),
        "REL_OPS":  sum(op_counts[o] for o in REL_OPS),
        "ARITH_OPS":sum(op_counts[o] for o in ARITH_OPS),
        "BV_OPS":   sum(op_counts[o] for o in BV_OPS),
        "STR_OPS":  sum(op_counts[o] for o in STR_OPS),
        "UF_OPS":   op_counts["APPLY"],
        "QUANT_OPS":op_counts["FORALL"] + op_counts["EXISTS"],
    }
    op_buckets = tuple(sorted((k, int_bucket(v)) for k, v in cat_counts.items()))

    top_ops = tuple(o for o, _ in op_counts.most_common(8))

    if not families:
        families.add("OTHER")

    return Features(
        sort_families=tuple(sorted(families)),
        bv_widths=tuple(sorted(bv_widths)),
        has_arrays=has_arrays,
        has_strings=has_strings,

        top_ops=top_ops,
        op_buckets=op_buckets,
        has_let=has_let,

        has_quantifiers=has_quant,
        quant_kinds=quant_kinds,
        max_quant_nesting=max_quant_nesting,
        has_patterns=has_patterns,

        uses_int=uses_int,
        uses_real=uses_real,
        is_nonlinear=is_nonlinear,
        has_div_or_mod=has_div_or_mod,

        has_bv=has_bv,
        has_extract=has_extract,
        has_concat=has_concat,
        has_shifts=has_shifts,
        has_bv_arith=has_bv_arith,
        has_bv_rel=has_bv_rel,

        bytes_bucket=bb,
        node_bucket=nb,
        max_depth_bucket=max_depth_b,

        name_buckets=name_buckets,
        literal_profile=literal_profile,
        backend="TOKEN",
        backend_note=f"{unit_kind} {notes}".strip(),
    )


# ----------------------------
# Z3 backend (opportunistic)
# ----------------------------

def try_parse_with_z3(assert_text: str, deadline: float) -> Optional[List[z3.ExprRef]]:
    if time.time() > deadline:
        return None
    try:
        exprs = z3.parse_smt2_string(assert_text)
        return list(exprs) if exprs is not None else None
    except Exception:
        return None

def z3_walk(expr: z3.ExprRef, counts: dict, depth: int = 0):
    counts["nodes"] += 1
    counts["max_depth"] = max(counts["max_depth"], depth)

    name = normalize_symbol(expr.decl().name())
    counts["decl_names"][name] += 1

    op = normalize_op(expr.decl().name())
    counts["ops"][op] += 1

    # sort family
    srt = expr.sort()
    fam = "OTHER"
    try:
        if srt.kind() == z3.Z3_BOOL_SORT:
            fam = "BOOL"
        elif srt.kind() == z3.Z3_INT_SORT:
            fam = "INT"
        elif srt.kind() == z3.Z3_REAL_SORT:
            fam = "REAL"
        elif srt.kind() == z3.Z3_BV_SORT:
            fam = "BV"
            counts["bv_widths"].add(srt.size())
        elif str(srt).lower() == "string":
            fam = "STRING"
        elif srt.kind() == z3.Z3_ARRAY_SORT:
            fam = "ARRAY"
        else:
            fam = "OTHER"
    except Exception:
        fam = "OTHER"
    counts["sort_families"].add(fam)

    # literal heuristics
    if z3.is_int_value(expr) or z3.is_rational_value(expr):
        counts["literal"]["NUM"] += 1
        try:
            v = expr.as_long()
            if v == 0:
                counts["literal"]["ZERO"] += 1
            if v == 1:
                counts["literal"]["ONE"] += 1
        except Exception:
            pass

    # nonlinear heuristic: MUL with non-numeral children
    if normalize_op(expr.decl().name()) == "MUL":
        nonconst = 0
        for c in expr.children():
            if not (z3.is_int_value(c) or z3.is_rational_value(c)):
                nonconst += 1
        if nonconst >= 2:
            counts["nonlinear"] = True

    for c in expr.children():
        z3_walk(c, counts, depth + 1)


def z3_backend_features(assert_text: str, unit_kind: str, notes: str) -> Features:
    # We assume parsing already succeeded and we got exprs; caller decides.
    exprs = z3.parse_smt2_string(assert_text)
    exprs = list(exprs) if exprs is not None else []
    nbytes = len(assert_text.encode("utf-8", errors="ignore"))
    bb = bytes_bucket(nbytes)

    counts = {
        "nodes": 0,
        "max_depth": 0,
        "ops": Counter(),
        "decl_names": Counter(),
        "sort_families": set(),
        "bv_widths": set(),
        "literal": Counter(),
        "nonlinear": False,
    }

    for e in exprs:
        z3_walk(e, counts, 0)

    families = counts["sort_families"] or {"OTHER"}
    bv_widths = counts["bv_widths"]

    op_counts = counts["ops"]
    has_let = False  # Z3 AST doesn't preserve 'let'; detect from text
    if re.search(r"\(\s*let\b", assert_text, flags=re.IGNORECASE):
        has_let = True

    has_patterns = ":pattern" in assert_text

    # quantifiers
    has_quant = bool(re.search(r"\(\s*(forall|exists)\b", assert_text, flags=re.IGNORECASE))
    qk = []
    if re.search(r"\(\s*forall\b", assert_text, flags=re.IGNORECASE):
        qk.append("FORALL")
    if re.search(r"\(\s*exists\b", assert_text, flags=re.IGNORECASE):
        qk.append("EXISTS")
    quant_kinds = tuple(sorted(set(qk)))
    max_quant_nesting = 1 if has_quant else 0

    uses_int = "INT" in families
    uses_real = "REAL" in families
    has_div_or_mod = op_counts["DIV"] > 0 or op_counts["MOD"] > 0
    is_nonlinear = "YES" if counts["nonlinear"] else ("MAYBE" if op_counts["MUL"] > 0 else "NO")

    has_bv = "BV" in families or any(o in BV_OPS for o in op_counts)
    has_extract = op_counts["BVEXTRACT"] > 0
    has_concat = op_counts["BVCONCAT"] > 0 or op_counts["STR.CONCAT"] > 0
    has_shifts = op_counts["BVSHL"] > 0 or op_counts["BVLSHR"] > 0 or op_counts["BVASHR"] > 0
    has_bv_arith = any(op_counts[o] > 0 for o in ("BVADD","BVSUB","BVMUL"))
    has_bv_rel = any(op_counts[o] > 0 for o in ("BVULT","BVULE","BVUGT","BVUGE","BVSLT","BVSLE","BVSGT","BVSGE"))

    has_arrays = "ARRAY" in families
    has_strings = "STRING" in families

    # name buckets from decl names (already normalized)
    name_buckets = Counter()
    for k, v in counts["decl_names"].items():
        if k in ("OCCURS@t","HOLDS@t","HOLD_S@t","STEP@t","INSTOCCURS@t","REQUIRED","CSPVAR","CSP_OP","AUX"):
            name_buckets[k] += v

    literal_profile = counts["literal"]

    cat_counts = {
        "BOOL_OPS": sum(op_counts[o] for o in BOOL_OPS),
        "REL_OPS":  sum(op_counts[o] for o in REL_OPS),
        "ARITH_OPS":sum(op_counts[o] for o in ARITH_OPS),
        "BV_OPS":   sum(op_counts[o] for o in BV_OPS),
        "STR_OPS":  sum(op_counts[o] for o in STR_OPS),
        "UF_OPS":   op_counts["APPLY"],
        "QUANT_OPS":op_counts["FORALL"] + op_counts["EXISTS"],
    }
    op_buckets = tuple(sorted((k, int_bucket(v)) for k, v in cat_counts.items()))
    top_ops = tuple(o for o, _ in op_counts.most_common(8))

    nb = node_bucket(counts["nodes"], step=100, cap=2000)
    md = depth_bucket(counts["max_depth"], cap=50)

    return Features(
        sort_families=tuple(sorted(families)),
        bv_widths=tuple(sorted(bv_widths)),
        has_arrays=has_arrays,
        has_strings=has_strings,

        top_ops=top_ops,
        op_buckets=op_buckets,
        has_let=has_let,

        has_quantifiers=has_quant,
        quant_kinds=quant_kinds,
        max_quant_nesting=max_quant_nesting,
        has_patterns=has_patterns,

        uses_int=uses_int,
        uses_real=uses_real,
        is_nonlinear=is_nonlinear,
        has_div_or_mod=has_div_or_mod,

        has_bv=has_bv,
        has_extract=has_extract,
        has_concat=has_concat,
        has_shifts=has_shifts,
        has_bv_arith=has_bv_arith,
        has_bv_rel=has_bv_rel,

        bytes_bucket=bb,
        node_bucket=nb,
        max_depth_bucket=md,

        name_buckets=name_buckets,
        literal_profile=literal_profile,
        backend="Z3",
        backend_note=f"{unit_kind} {notes}".strip(),
    )


# ============================================================
# Clustering + reporting
# ============================================================

@dataclass
class ClusterInfo:
    key: Tuple
    count: int
    reps: List[str]
    backends: Counter
    notes: Counter
    name_buckets: Counter
    literal_profile: Counter

def render_core_key(key: Tuple) -> str:
    # readable rendering for comments; keep deterministic but compact
    # key structure: ("S2", sort_profile, op_profile, quant_profile, arith_profile, bv_profile, size_profile)
    _, sort_p, op_p, q_p, a_p, bv_p, sz_p = key
    return (
        f"S2 sort={sort_p} "
        f"ops={op_p} "
        f"quant={q_p} "
        f"arith={a_p} "
        f"bv={bv_p} "
        f"size={sz_p}"
    )

def cluster_units(
    units_with_feats: List[Tuple[Unit, Features]],
    reps_per_cluster: int,
) -> Dict[Tuple, ClusterInfo]:
    clusters: Dict[Tuple, ClusterInfo] = {}

    for unit, feats in units_with_feats:
        k = feats.core_key()
        if k not in clusters:
            clusters[k] = ClusterInfo(
                key=k, count=0, reps=[], backends=Counter(), notes=Counter(),
                name_buckets=Counter(), literal_profile=Counter()
            )
        c = clusters[k]
        c.count += 1
        c.backends[feats.backend] += 1
        c.notes[feats.backend_note] += 1
        c.name_buckets.update(feats.name_buckets)
        c.literal_profile.update(feats.literal_profile)

        if len(c.reps) < reps_per_cluster:
            # Keep representative snippet; for WINDOW, trim aggressively
            txt = unit.unit_text.strip()
            if unit.unit_kind == "WINDOW":
                txt = txt[:2000] + ("..." if len(txt) > 2000 else "")
                txt = f"; [WINDOW SNIPPET]\n; notes: {unit.notes}\n; ---\n{txt}\n; ---\n"
            c.reps.append(txt + ("\n" if not txt.endswith("\n") else ""))

    return clusters

def select_representative_clusters(
    clusters: Dict[Tuple, ClusterInfo],
    assert_budget: int,
    rare_threshold: int,
    keep_rare_total: int,
) -> List[ClusterInfo]:
    # sort by cluster size desc
    ordered = sorted(clusters.values(), key=lambda c: c.count, reverse=True)
    reps: List[ClusterInfo] = []
    kept = 0
    rare_kept = 0

    for c in ordered:
        if kept >= assert_budget:
            break
        if c.count > rare_threshold:
            reps.append(c); kept += 1
        else:
            if rare_kept < keep_rare_total:
                reps.append(c); kept += 1; rare_kept += 1
    return reps


# ============================================================
# Main reducer (V2)
# ============================================================

def reduce_smt_v2(
    in_path: str,
    z3_seconds: float,
    assert_budget: int,
    keep_rare_total: int,
    rare_threshold: int,
    reps_per_cluster: int,
    huge_assert_mb: int,
    z3_unit_kb: int,
    max_units: int,
    window_count: int,
    window_bytes: int,
    decl_sig_budget: int,
    decl_examples_per_sig: int,
) -> Tuple[str, dict]:
    with open(in_path, "r", encoding="utf-8", errors="replace") as f:
        full_text = f.read()

    best_session = choose_best_session(full_text)
    comp = extract_session_components(best_session)

    structural_text = strip_trailing_commands(comp["structural_text"])
    structural_text = compress_blank_lines(structural_text, max_consecutive=2)

    decls = comp["decls"]
    defs = comp["defs"]
    asserts = comp["asserts"]

    # summarize declare-fun
    decl_sig_counts = Counter()
    decl_sig_examples: Dict[Tuple, List[str]] = defaultdict(list)
    name_bucket_counts = Counter()

    for d in decls:
        sig = parse_declare_fun_signature(d)
        decl_sig_counts[sig] += 1
        if len(decl_sig_examples[sig]) < decl_examples_per_sig:
            dd = d.strip() + ("\n" if not d.endswith("\n") else "")
            decl_sig_examples[sig].append(dd)

    for sig, cnt in decl_sig_counts.items():
        if len(sig) >= 4:
            name_bucket_counts[sig[3]] += cnt
        else:
            name_bucket_counts["UNKNOWN"] += cnt

    # unitize assertions
    huge_assert_bytes = huge_assert_mb * 1_000_000
    units: List[Unit] = []
    for i, a in enumerate(asserts):
        units.extend(make_units_from_assert(
            assert_text=a,
            assert_index=i,
            huge_assert_bytes=huge_assert_bytes,
            max_units=max_units,
            window_count=window_count,
            window_bytes=window_bytes,
        ))

    # feature extraction
    deadline = time.time() + max(0.0, z3_seconds)
    z3_unit_max_bytes = z3_unit_kb * 1000
    units_with_feats: List[Tuple[Unit, Features]] = []

    z3_ok = 0
    z3_fail = 0
    z3_skip_size = 0
    token_only = 0

    for u in units:
        ub = len(u.unit_text.encode("utf-8", errors="ignore"))
        use_z3 = (z3_seconds > 0) and (u.unit_kind != "WINDOW") and (ub <= z3_unit_max_bytes) and (time.time() <= deadline)

        if use_z3:
            parsed = try_parse_with_z3(u.unit_text, deadline)
            if parsed:
                z3_ok += 1
                feats = z3_backend_features(u.unit_text, u.unit_kind, u.notes)
            else:
                z3_fail += 1
                feats = token_backend_features(u.unit_text, u.unit_kind, u.notes + " z3_fail_or_timeout")
        else:
            if u.unit_kind != "WINDOW" and ub > z3_unit_max_bytes:
                z3_skip_size += 1
            token_only += 1
            feats = token_backend_features(u.unit_text, u.unit_kind, u.notes + (" z3_skipped" if z3_seconds > 0 else " z3_disabled"))

        units_with_feats.append((u, feats))

    # cluster
    clusters = cluster_units(units_with_feats, reps_per_cluster=reps_per_cluster)
    selected = select_representative_clusters(clusters, assert_budget, rare_threshold, keep_rare_total)

    # assemble output
    out_lines: List[str] = []
    out_lines.append(structural_text.rstrip() + "\n")

    if defs:
        out_lines.append("\n; ===== Kept define-fun blocks =====\n\n")
        for d in defs:
            dd = d.strip()
            out_lines.append(dd + ("\n" if not dd.endswith("\n") else ""))
            out_lines.append("\n")

    out_lines.append("\n; ===== Declaration summary (declare-fun) =====\n")
    out_lines.append(f"; Total declare-fun: {sum(decl_sig_counts.values())}\n")
    out_lines.append("; By name bucket (grounding abstraction):\n")
    for bucket, cnt in name_bucket_counts.most_common():
        out_lines.append(f";   {bucket}: {cnt}\n")

    out_lines.append("\n; Top declare-fun signatures:\n")
    for sig, cnt in decl_sig_counts.most_common(decl_sig_budget):
        out_lines.append(f"; {cnt}× signature: {sig}\n")
        for ex in decl_sig_examples.get(sig, [])[:decl_examples_per_sig]:
            out_lines.append(ex)
        out_lines.append("\n")

    # assertion/unit summary
    out_lines.append("\n; ===== Assertion/Unit schema summary (V2) =====\n")
    out_lines.append(f"; Total asserts found: {len(asserts)}\n")
    out_lines.append(f"; Total units produced: {len(units)}\n")
    out_lines.append(f"; Unique schemas: {len(clusters)}\n")
    out_lines.append(f"; Z3 parsed units: {z3_ok}\n")
    out_lines.append(f"; Z3 failed/timeout units: {z3_fail}\n")
    out_lines.append(f"; Z3 skipped due to size: {z3_skip_size}\n")
    out_lines.append(f"; Token backend units (incl. windows): {token_only}\n")
    out_lines.append(f"; Printed cluster representatives: {len(selected)} (budget={assert_budget})\n")
    out_lines.append(f"; Notes: output is NOT equisatisfiable; intended for inspection only.\n\n")

    for c in selected:
        out_lines.append(f"; Cluster size: {c.count}\n")
        out_lines.append(f"; Backends: {dict(c.backends)}\n")
        # print top few notes
        top_notes = c.notes.most_common(3)
        if top_notes:
            out_lines.append(f"; Common notes: {top_notes}\n")
        if c.name_buckets:
            out_lines.append(f"; Name buckets (top): {c.name_buckets.most_common(5)}\n")
        if c.literal_profile:
            out_lines.append(f"; Literals (top): {c.literal_profile.most_common(5)}\n")
        out_lines.append(f"; Schema: {render_core_key(c.key)}\n")
        for rep in c.reps:
            out_lines.append(rep.strip() + "\n")
        out_lines.append("\n")

    out_lines.append("\n; ===== Footer =====\n")
    out_lines.append("(check-sat)\n(exit)\n")

    stats = {
        "asserts_total": len(asserts),
        "units_total": len(units),
        "schemas_total": len(clusters),
        "clusters_printed": len(selected),
        "decl_total": sum(decl_sig_counts.values()),
        "decl_sig_groups": len(decl_sig_counts),
        "z3_ok_units": z3_ok,
        "z3_fail_units": z3_fail,
        "z3_skip_size_units": z3_skip_size,
        "token_units": token_only,
        "huge_assert_mb": huge_assert_mb,
        "z3_unit_kb": z3_unit_kb,
    }

    return "".join(out_lines), stats


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="SMT2 reducer/summarizer V2 (schema + unitization, Z3 opportunistic)")
    ap.add_argument("input", help="Input SMT2 file")
    ap.add_argument("-o", "--output", default="reduced_v2.smt2", help="Output file")

    ap.add_argument("--z3-seconds", type=float, default=DEFAULT_Z3_SECONDS,
                    help="Total time budget for Z3 parsing (seconds). 0 disables Z3 parsing.")
    ap.add_argument("--assert-budget", type=int, default=DEFAULT_ASSERT_BUDGET,
                    help="Max number of cluster representatives to print.")
    ap.add_argument("--keep-rare-total", type=int, default=DEFAULT_KEEP_RARE_TOTAL,
                    help="Max number of rare clusters to print total.")
    ap.add_argument("--rare-threshold", type=int, default=DEFAULT_RARE_THRESHOLD,
                    help="Clusters with size <= this are considered rare.")
    ap.add_argument("--reps-per-cluster", type=int, default=DEFAULT_REPS_PER_CLUSTER,
                    help="Representative snippets per cluster.")

    ap.add_argument("--huge-assert-mb", type=int, default=DEFAULT_HUGE_ASSERT_MB,
                    help="If an assert block exceeds this size, attempt splitting/windows.")
    ap.add_argument("--z3-unit-kb", type=int, default=DEFAULT_Z3_UNIT_KB,
                    help="Only attempt Z3 parsing for units up to this size (KB).")
    ap.add_argument("--max-units", type=int, default=DEFAULT_MAX_UNITS,
                    help="Hard cap on units produced per file (safety).")
    ap.add_argument("--window-count", type=int, default=DEFAULT_WINDOW_COUNT,
                    help="Fallback windows per huge assert when splitting fails.")
    ap.add_argument("--window-bytes", type=int, default=DEFAULT_WINDOW_BYTES,
                    help="Bytes per fallback window (approx).")

    ap.add_argument("--decl-sig-budget", type=int, default=5,
                    help="Max number of declare-fun signature groups printed.")
    ap.add_argument("--decl-examples-per-sig", type=int, default=1,
                    help="Example declare-fun blocks per signature group.")

    args = ap.parse_args()

    out_text, stats = reduce_smt_v2(
        in_path=args.input,
        z3_seconds=args.z3_seconds,
        assert_budget=args.assert_budget,
        keep_rare_total=args.keep_rare_total,
        rare_threshold=args.rare_threshold,
        reps_per_cluster=args.reps_per_cluster,
        huge_assert_mb=args.huge_assert_mb,
        z3_unit_kb=args.z3_unit_kb,
        max_units=args.max_units,
        window_count=args.window_count,
        window_bytes=args.window_bytes,
        decl_sig_budget=args.decl_sig_budget,
        decl_examples_per_sig=args.decl_examples_per_sig,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(out_text)

    print(f"Wrote {args.output}")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
