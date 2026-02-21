#!/usr/bin/env python3
"""
Extract descriptions from meta-info JSON files (e.g. data/meta_info_24/ABV.json)
and write per-logic JSON files under data/meta_info_24/descriptions/.

Output format: one JSON per logic; top-level keys are smt-lib paths; each value has
  - raw_description: the original description (unchanged, may be empty)
  - description: same as used by encode_all_logic_desc: placeholder if missing/empty,
    otherwise stripped text.
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
        out[smtlib_path] = {"raw_description": raw, "description": description}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract descriptions from meta-info JSONs into data/meta_info_24/descriptions/."
    )
    parser.add_argument(
        "--meta-dir",
        type=Path,
        default=Path("data") / "meta_info_24",
        help="Directory containing per-logic meta-info JSON files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <meta-dir>/descriptions)",
    )
    args = parser.parse_args()

    meta_dir = args.meta_dir.resolve()
    out_dir = (args.out_dir or meta_dir / "descriptions").resolve()

    if not meta_dir.is_dir():
        print(f"Error: meta-info directory is not a directory: {meta_dir}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
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
