#!/usr/bin/env python3
"""
Extract syntactic feature CSV from meta_info JSON.

Reads a JSON file (e.g. data/meta_info_24/ABV.json), and writes a CSV under
data/features/syntactic/catalog_all/ with:
  - Rows: one per benchmark (smtlib_path)
  - Columns: size, asserts_count, declare_fun_count, declare_const_count,
    declare_sort_count, define_fun_count, define_fun_rec_count,
    constant_fun_count, define_sort_count, declare_datatype_count,
    max_term_depth, then one column per symbol (count or 0).
"""

import argparse
import csv
import json
from pathlib import Path

FIXED_COLUMNS = [
    "size",
    "asserts_count",
    "declare_fun_count",
    "declare_const_count",
    "declare_sort_count",
    "define_fun_count",
    "define_fun_rec_count",
    "constant_fun_count",
    "define_sort_count",
    "declare_datatype_count",
    "max_term_depth",
]

# Symbol columns (same order as data/symbols.txt)
SYMBOLS = [
    "true", "false", "Bool", "ite", "not", "or", "and", "=>", "xor", "=",
    "distinct", "const", "forall", "exists", "let", "Int", "div", "mod",
    "divisible", "iand", "int.pow2", "div_total", "mod_total", "Real", "/",
    "/_total", "+", "-", "*", "<", "<=", ">", ">=", "to_real", "to_int",
    "is_int", "abs", "^", "-", "real.pi", "exp", "sin", "cos", "tan", "csc", "sec",
    "cot", "arcsin", "arccos", "arctan", "arccsc", "arcsec", "arccot", "sqrt",
    "BitVec", "bvempty", "concat", "extract", "repeat", "bvnot", "bvand",
    "bvor", "bvnand", "bvnor", "bvxor", "bvxnor", "bvcomp", "bvneg", "bvadd",
    "bvmul", "bvudiv", "bvurem", "bvsub", "bvsdiv", "bvsrem", "bvsmod",
    "bvult", "bvule", "bvugt", "bvuge", "bvslt", "bvsle", "bvsgt", "bvsge",
    "bvshl", "bvlshr", "bvashr", "zero_extend", "sign_extend", "rotate_left",
    "rotate_right", "reduce_and", "reduce_or", "reduce_xor", "bvite", "bv1ult",
    "bitOf", "bvuaddo", "bvsaddo", "bvumulo", "bvsmulo", "bvusubo", "bvssubo",
    "bvsdivo", "bvultbv", "bvsltbv", "bvredand", "bvredor", "int2bv", "bv2nat",
    "bvsize", "Array", "select", "store", "store_all", "eqrange", "FloatingPoint",
    "RoundingMode", "fp", "fp.add", "fp.sub", "fp.mul", "fp.div", "fp.fma",
    "fp.sqrt", "fp.rem", "fp.roundToIntegral", "fp.min", "fp.max", "fp.abs",
    "fp.neg", "fp.leq", "fp.lt", "fp.geq", "fp.gt", "fp.eq", "fp.isNormal",
    "fp.isSubnormal", "fp.isZero", "fp.isInfinite", "fp.isNaN", "fp.isPositive",
    "fp.isNegative", "roundNearestTiesToEven", "roundNearestTiesToAway",
    "roundTowardPositive", "roundTowardNegative", "roundTowardZero", "fp.to_ubv",
    "fp.to_ubv_total", "fp.to_sbv", "fp.to_sbv_total", "fp.to_real", "to_fp",
    "to_fp_unsigned", "to_fp_bv", "String", "Char", "RegLan", "str.len", "str.++",
    "str.substr", "str.contains", "str.replace", "str.indexof", "str.at",
    "str.prefixof", "str.suffixof", "str.rev", "str.unit", "str.update",
    "str.to_lower", "str.to_upper", "str.to_code", "str.from_code", "str.is_digit",
    "str.to_int", "str.from_int", "str.<", "str.<=", "str.replace_all",
    "str.replace_re", "str.replace_re_all", "str.indexof_re", "re.allchar",
    "re.none", "re.all", "re.empty", "str.to_re", "re.*", "re.+", "re.opt",
    "re.comp", "re.range", "re.++", "re.inter", "re.union", "re.diff", "re.loop",
    "str.in_re", "seq.empty", "seq.unit", "seq.nth", "seq.len",
]

DEFAULT_OUTPUT_DIR = "data/features/syntactic/catalog_all"
EXPECTED_SYMBOLS_SIZE = 204


def extract_csv(input_json: Path, output_csv: Path) -> tuple[int, int]:
    assert len(SYMBOLS) == EXPECTED_SYMBOLS_SIZE, (
        f"SYMBOLS size {len(SYMBOLS)} != expected {EXPECTED_SYMBOLS_SIZE}"
    )
    with input_json.open() as f:
        entries = json.load(f)

    header = ["path"] + FIXED_COLUMNS + SYMBOLS

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for entry in entries:
            row = [entry["smtlib_path"]]
            for col in FIXED_COLUMNS:
                row.append(entry.get(col, 0))
            symbol_counts = entry.get("symbol_counts") or {}
            for sym in SYMBOLS:
                row.append(symbol_counts.get(sym, 0))
            writer.writerow(row)
    return len(entries), len(header)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract syntactic feature CSV from meta_info JSON",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Single input JSON path",
    )
    parser.add_argument(
        "--meta-dir",
        type=Path,
        default=None,
        help="Process all .json files in this directory (e.g. data/meta_info_24)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (only with --input; default: data/features/syntactic/catalog_all/<stem>.csv)",
    )
    args = parser.parse_args()

    if args.meta_dir is not None:
        meta_dir = args.meta_dir.resolve()
        if not meta_dir.is_dir():
            raise FileNotFoundError(f"Meta dir not found: {meta_dir}")
        output_dir = Path(DEFAULT_OUTPUT_DIR).resolve()
        json_files = sorted(meta_dir.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No .json files in {meta_dir}")
        for input_json in json_files:
            output_csv = output_dir / f"{input_json.stem}.csv"
            n_rows, n_cols = extract_csv(input_json, output_csv)
            print(f"Wrote {n_cols} columns, {n_rows} rows -> {output_csv}")
        return

    # Single-file mode
    input_json = args.input.resolve() if args.input is not None else Path("data/meta_info_24/ABV.json").resolve()
    if not input_json.is_file():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    if args.output is not None:
        output_csv = args.output.resolve()
    else:
        output_dir = Path(DEFAULT_OUTPUT_DIR).resolve()
        stem = input_json.stem
        output_csv = output_dir / f"{stem}.csv"

    n_rows, n_cols = extract_csv(input_json, output_csv)
    print(f"Wrote {n_cols} columns, {n_rows} rows -> {output_csv}")


if __name__ == "__main__":
    main()
