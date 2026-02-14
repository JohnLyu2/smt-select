#!/usr/bin/env python3
"""
Encode all benchmark descriptions from meta-info JSON files in data/meta_info_24/
and save to CSV. Requires --output-dir (unless --trunc-stats).
Use --trunc-stats to only show truncation statistics without writing CSVs.
"""

import argparse
import os
import sys
from pathlib import Path

# Avoid tokenizers fork warnings by disabling parallelism.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Add src to path to import desc_encoder
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.desc_encoder import encode_all_desc

# Meta-info folder (relative to project root)
META_INFO_DIR = Path("data") / "meta_info_24"
# Logics to skip (large or otherwise excluded)
EXCLUDED_LOGICS = {"AUFBVDTNIRA", "QF_NIA"}


def main():
    """Process all meta-info JSON files in data/meta_info_24/ and create CSV files."""
    parser = argparse.ArgumentParser(
        description="Encode all benchmark descriptions from meta-info JSON files and save to CSV."
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
        default="sentence-transformers/all-mpnet-base-v2",
        help="Sentence-transformer model name to use",
    )

    args = parser.parse_args()

    # Validate: output-dir is required unless --trunc-stats is used
    if not args.trunc_stats and not args.output_dir:
        parser.error("--output-dir is required unless --trunc-stats is used")

    project_root = Path(__file__).parent.parent
    meta_info_dir = project_root / META_INFO_DIR
    if not meta_info_dir.is_dir():
        parser.error(f"Meta-info directory is not a directory or does not exist: {meta_info_dir}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

    # Find all JSON files, excluding specified logics
    all_json = sorted(meta_info_dir.glob("*.json"))
    json_files = [f for f in all_json if f.stem not in EXCLUDED_LOGICS]

    if not json_files:
        print(f"No meta-info JSON files found in {meta_info_dir}")
        return 1

    print(f"Meta-info directory: {meta_info_dir}")
    print(f"Found {len(json_files)} JSON file(s) to process")
    if args.output_dir:
        print(f"Output directory: {output_dir}\n")
    else:
        print("Showing truncation statistics only\n")

    failed = 0
    # Process each JSON file
    for json_file in json_files:
        logic = json_file.stem  # e.g., "ABV" from "ABV.json"

        print(f"Processing {logic}...")
        print(f"  Input:  {json_file}")

        # Only set output_csv if output_dir is provided
        output_csv = None
        if args.output_dir:
            output_csv = output_dir / f"{logic}.csv"
            print(f"  Output: {output_csv}")

        try:
            csv_path = encode_all_desc(
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