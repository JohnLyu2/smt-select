#!/usr/bin/env python3
"""
Extract descriptions from meta-info JSON files and write per-logic description JSONs.

Input: per-logic meta-info JSONs (default data/raw_data/meta_info/, e.g. ABV.json)
  with benchmark entries containing a description field.
Output: one JSON per logic under --out-dir (default data/descriptions/).
  Top-level keys are smt-lib paths; each value has raw_description and description.
  encode_desc_logics.py and train_setfit_splits.py read from that output path.
"""

import argparse
import json
import sys
from pathlib import Path

from description_rules import apply_description_rule


def description_for_benchmark(benchmark: dict) -> tuple[str, str]:
    """Return (raw_description, description) for a benchmark.

    Mirrors missing/empty handling used when encoding (see encode_all_desc_from_descriptions_file):
    - raw: original value of 'description' or '' if missing.
    - description: if missing or empty/whitespace, use placeholder
      'This is a {logic} benchmark from the family {family}'; else stripped.
    """
    raw = benchmark.get("description", "")
    if not raw or not raw.strip():
        logic = benchmark.get("logic", "unknown")
        family = benchmark.get("family", "unknown")
        description = f"This is a {logic} benchmark from the family {family}"
    else:
        description = raw.strip()
    return (raw, description)


def _prepend_application_category(description: str, benchmark: dict) -> str:
    """Prepend 'Application: XXX' and 'Category: XXX' to description when available."""
    parts: list[str] = []
    app = benchmark.get("application", "").strip()
    if app:
        parts.append(f"Application: {app}")
    cat = benchmark.get("category", "").strip()
    if cat:
        parts.append(f"Category: {cat}")
    if not parts:
        return description
    return "\n".join(parts) + "\n\n" + description


def extract_descriptions_from_json(json_path: Path) -> dict[str, dict]:
    """Read a meta-info JSON and return a dict path -> {raw_description, description}."""
    with open(json_path, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)
    out: dict[str, dict] = {}
    for benchmark in benchmarks:
        smtlib_path = benchmark.get("smtlib_path", "")
        if not smtlib_path:
            continue
        raw, description = description_for_benchmark(benchmark)
        logic = benchmark.get("logic", "")
        family = benchmark.get("family", "")
        description = apply_description_rule(logic, family, description, smtlib_path)
        description = _prepend_application_category(description, benchmark)
        out[smtlib_path] = {"raw_description": raw, "description": description}
    return out


# Output default: downstream scripts (encode_desc_logics, train_setfit_splits) expect this path.
DEFAULT_OUT_DIR = Path("data") / "descriptions"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract descriptions from meta-info JSONs into description JSONs (default out: data/descriptions/)."
    )
    parser.add_argument(
        "--meta-dir",
        type=Path,
        default=Path("data") / "raw_data" / "meta_info",
        help="Directory containing per-logic meta-info JSON files (default: data/raw_data/meta_info)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=f"Output directory for description JSONs (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--logic",
        type=str,
        default=None,
        metavar="LOGIC",
        help="Process only this logic (e.g. QF_NRA). Default: all JSON files in meta-dir.",
    )
    args = parser.parse_args()

    meta_dir = args.meta_dir.resolve()
    out_dir = (args.out_dir or DEFAULT_OUT_DIR).resolve()

    if not meta_dir.is_dir():
        print(f"Error: meta-info directory is not a directory: {meta_dir}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.logic:
        json_path = meta_dir / f"{args.logic}.json"
        if not json_path.is_file():
            print(f"Error: no such file {json_path}", file=sys.stderr)
            return 1
        json_files = [json_path]
    else:
        json_files = sorted(meta_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {meta_dir}", file=sys.stderr)
        return 1

    for json_path in json_files:
        logic = json_path.stem
        try:
            path_to_desc = extract_descriptions_from_json(json_path)
            out_path = out_dir / f"{logic}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(path_to_desc, f, indent=2, ensure_ascii=False)
            print(f"  {logic}: {len(path_to_desc)} instances -> {out_path}")
        except Exception as e:
            print(f"  Error processing {logic}: {e}", file=sys.stderr)
            return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
