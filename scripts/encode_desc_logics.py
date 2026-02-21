#!/usr/bin/env python3
"""
Encode benchmark descriptions from description JSONs in data/meta_info_24/descriptions/
(e.g. ABV.json: path -> {raw_description, description}) and save to CSV.
Requires --output-dir (unless --trunc-stats). Use --trunc-stats to only show truncation statistics.
"""

import argparse
import os
import sys
from pathlib import Path

# Avoid tokenizers fork warnings by disabling parallelism.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Add src to path to import desc_encoder
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.desc_encoder import encode_all_desc_from_descriptions_file

# Input: description JSONs (path -> {raw_description, description}), same as scripts/desc_extract output
DESCRIPTIONS_DIR = Path("data") / "meta_info_24" / "descriptions"
# Logics to skip (large or otherwise excluded)
EXCLUDED_LOGICS = {"AUFBVDTNIRA", "QF_NIA"}


def main():
    """Process all description JSONs in data/meta_info_24/descriptions/ and create CSV files."""
    parser = argparse.ArgumentParser(
        description="Encode benchmark descriptions from description JSONs (data/meta_info_24/descriptions/) and save to CSV."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for CSV files (required unless --trunc-stats is used)",
    )
    parser.add_argument(
        "--trunc-stats",
        action="store_true",
        help="Show truncation statistics",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Text embedding model name (default: Qwen/Qwen3-Embedding-0.6B)",
    )

    args = parser.parse_args()

    # Output-dir is required unless --trunc-stats is used
    if not args.trunc_stats and not args.output_dir:
        parser.error("--output-dir is required unless --trunc-stats is used")

    project_root = Path(__file__).parent.parent
    descriptions_dir = project_root / DESCRIPTIONS_DIR
    if not descriptions_dir.is_dir():
        parser.error(f"Descriptions directory is not a directory or does not exist: {descriptions_dir}")

    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Find all description JSON files, excluding specified logics
    all_json = sorted(descriptions_dir.glob("*.json"))
    json_files = [f for f in all_json if f.stem not in EXCLUDED_LOGICS]

    if not json_files:
        print(f"No description JSON files found in {descriptions_dir}")
        return 1

    print(f"Descriptions directory: {descriptions_dir}")
    print(f"Found {len(json_files)} JSON file(s) to process")
    if output_dir is not None:
        print(f"Output directory: {output_dir}\n")
    else:
        print("Showing truncation statistics only\n")

    failed = 0
    # Process each JSON file
    for json_file in json_files:
        logic = json_file.stem  # e.g., "ABV" from "ABV.json"

        print(f"Processing {logic}...")
        print(f"  Input:  {json_file}")

        # Only set output_csv if we're writing (not trunc-stats only)
        output_csv = (output_dir / f"{logic}.csv") if output_dir is not None else None
        if output_csv is not None:
            print(f"  Output: {output_csv}")

        try:
            csv_path = encode_all_desc_from_descriptions_file(
                json_path=str(json_file),
                output_csv_path=str(output_csv) if output_csv else None,
                model_name=args.model_name,
                normalize=False,
                batch_size=8,
                show_progress=True,
                show_trunc_stats=args.trunc_stats,
            )
            if csv_path:
                print(f"  Success! Saved to {csv_path}\n")
            else:
                print("  Truncation statistics shown (no encoding performed)\n")
        except Exception as e:
            print(f"  Error processing {logic}: {e}\n", file=sys.stderr)
            failed += 1
            continue

    print("All processing complete!" if failed == 0 else f"Complete with {failed} failure(s).")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())