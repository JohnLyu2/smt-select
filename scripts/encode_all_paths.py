#!/usr/bin/env python3
"""
Script to encode all benchmark paths from JSON files in data/raw_jsons/
and save them to data/features/path_emb/all_mpnet_base_v2/{LOGIC}.csv
"""

import argparse
import sys
from pathlib import Path

# Add src to path to import path_encoder
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.path_encoder import encode_all_paths


def main():
    """Process all JSON files in data/raw_jsons/ and create CSV files."""
    parser = argparse.ArgumentParser(
        description="Encode all benchmark paths from JSON files and save to CSV."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for CSV files",
    )

    args = parser.parse_args()

    # Define paths
    project_root = Path(__file__).parent.parent
    raw_jsons_dir = project_root / "data" / "raw_jsons"
    output_dir = Path(args.output_dir)

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all JSON files
    json_files = sorted(raw_jsons_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {raw_jsons_dir}")
        return 1

    print(f"Found {len(json_files)} JSON file(s) to process")
    print(f"Output directory: {output_dir}\n")

    # Process each JSON file
    success_count = 0
    error_count = 0

    for json_file in json_files:
        logic = json_file.stem  # e.g., "ABV" from "ABV.json"

        print(f"Processing {logic}...")
        print(f"  Input:  {json_file}")

        output_csv = output_dir / f"{logic}.csv"
        print(f"  Output: {output_csv}")

        try:
            csv_path = encode_all_paths(
                json_path=str(json_file),
                output_csv_path=str(output_csv),
                model_name="sentence-transformers/all-mpnet-base-v2",
                normalize=False,
                batch_size=32,
                show_progress=True,
            )
            print(f"  Success! Saved to {csv_path}\n")
            success_count += 1
        except Exception as e:
            print(f"  Error processing {logic}: {e}\n", file=sys.stderr)
            error_count += 1
            continue

    print("=" * 60)
    print("Processing complete!")
    print(f"  Successful: {success_count}/{len(json_files)}")
    if error_count > 0:
        print(f"  Errors: {error_count}/{len(json_files)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
