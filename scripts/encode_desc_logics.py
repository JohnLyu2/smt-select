#!/usr/bin/env python3
"""
Encode benchmark descriptions from description JSONs in data/descriptions/
(e.g. ABV.json: path -> {raw_description, description}) and save to CSV.

Output layout (like data/features/syntactic): one folder per logic, each with
- features.csv: path + embedding columns (emb_0, emb_1, ...)
- extraction_times.csv: path, time_sec, failed (failed is always 0).
Output goes to --output-dir (default: data/features/desc_emb). Use --trunc-stats to only show truncation statistics.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Avoid tokenizers fork warnings by disabling parallelism.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Add src to path to import desc_encoder
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.desc_encoder import encode_all_desc_from_descriptions_file

# Fixed input directory for description JSONs (path -> {raw_description, description})
DESCRIPTIONS_DIR = "data/descriptions"
# Default output base (one folder per logic: <output_dir>/<logic>/features.csv, extraction_times.csv)
DEFAULT_OUTPUT_DIR = "data/features/desc/all-mpnet-base-v2"
# When using --setfit-dir, seeds to use per logic (all seeds per logic = Option B)
SETFIT_SEEDS = [0, 10, 20, 30, 40]


def main():
    """Process all description JSONs in data/descriptions/ and create CSV files."""
    parser = argparse.ArgumentParser(
        description="Encode benchmark descriptions from description JSONs (data/descriptions/) and save to CSV."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output base directory: one folder per logic with features.csv and extraction_times.csv (default: {DEFAULT_OUTPUT_DIR})",
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
        help="Text embedding model name (default: sentence-transformers/all-mpnet-base-v2)",
    )
    parser.add_argument(
        "--logic",
        type=str,
        nargs="*",
        default=None,
        metavar="LOGIC",
        help="Process only these logics (e.g. --logic BV). Default: all logics in descriptions dir.",
    )
    parser.add_argument(
        "--setfit",
        action="store_true",
        help="Use a SetFit model: --model-name is path to saved SetFit dir (uses backbone for encoding).",
    )
    parser.add_argument(
        "--setfit-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Use SetFit models per logic/seed: PATH must contain <logic>/seed<N> dirs (e.g. models/setfit_mpnet/setfit_mpnet). Encodes each logic with all seeds; output to <output-dir>/<logic>/seed<N>/. Implies --setfit.",
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    descriptions_dir = (project_root / DESCRIPTIONS_DIR).resolve()
    if not descriptions_dir.is_dir():
        parser.error(f"Descriptions directory is not a directory or does not exist: {descriptions_dir}")

    setfit_dir: Path | None = None
    if args.setfit_dir:
        setfit_dir = (project_root / args.setfit_dir).resolve()
        if not setfit_dir.is_dir():
            parser.error(f"SetFit directory is not a directory or does not exist: {setfit_dir}")

    output_dir: Path | None = None
    if not args.trunc_stats:
        default_out = "data/features/desc/setfit_mpnet" if setfit_dir else DEFAULT_OUTPUT_DIR
        output_dir = (project_root / (args.output_dir or default_out)).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    # Find all description JSON files
    json_files = sorted(descriptions_dir.glob("*.json"))
    if args.logic:
        requested = set(args.logic)
        json_files = [f for f in json_files if f.stem in requested]
        missing = requested - {f.stem for f in json_files}
        if missing:
            parser.error(f"Logic(s) not found in {descriptions_dir}: {sorted(missing)}")

    if not json_files:
        print(f"No description JSON files found in {descriptions_dir}")
        return 1

    print(f"Descriptions directory: {descriptions_dir}")
    print(f"Found {len(json_files)} JSON file(s) to process")
    if setfit_dir is not None:
        print(f"SetFit directory: {setfit_dir} (all seeds per logic: {SETFIT_SEEDS})")
    if output_dir is not None:
        print(f"Output directory: {output_dir}\n")
    else:
        print("Showing truncation statistics only\n")

    # --- SetFit per (logic, seed) mode: encode each logic with each seed model ---
    if setfit_dir is not None and output_dir is not None:
        for json_file in json_files:
            logic = json_file.stem
            for seed in SETFIT_SEEDS:
                model_path = setfit_dir / logic / f"seed{seed}"
                if not model_path.is_dir():
                    raise FileNotFoundError(
                        f"SetFit model directory not found: {model_path}"
                    )
            for seed in SETFIT_SEEDS:
                model_path = setfit_dir / logic / f"seed{seed}"
                logic_dir = output_dir / logic / f"seed{seed}"
                logic_dir.mkdir(parents=True, exist_ok=True)
                features_csv = logic_dir / "features.csv"
                times_csv = logic_dir / "extraction_times.csv"
                print(f"Processing {logic} seed{seed}...")
                print(f"  Model:  {model_path}")
                print(f"  Output: {logic_dir}/")
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    paths = [p for p in data if isinstance(data.get(p), dict) and "description" in data[p]]
                    n = len(paths)
                    t0 = time.perf_counter()
                    csv_path = encode_all_desc_from_descriptions_file(
                        json_path=str(json_file),
                        output_csv_path=str(features_csv),
                        model_name=str(model_path),
                        normalize=False,
                        batch_size=8,
                        show_progress=True,
                        show_trunc_stats=args.trunc_stats,
                        is_setfit=True,
                    )
                    elapsed = time.perf_counter() - t0
                    if csv_path is None:
                        print("  Truncation statistics shown (no encoding performed)\n")
                        continue
                    time_per_path = elapsed / n if n else 0.0
                    with open(times_csv, "w", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(["path", "time_sec", "failed"])
                        for p in paths:
                            writer.writerow([p, time_per_path, 0])
                    print(f"  Success! {n} instances in {elapsed:.2f}s\n")
                except Exception as e:
                    print(f"  Error: {e}\n", file=sys.stderr)
                    return 1
        print("All processing complete!")
        return 0

    # Process each JSON file: one logic folder with features.csv and extraction_times.csv
    for json_file in json_files:
        logic = json_file.stem  # e.g., "ABV" from "ABV.json"

        print(f"Processing {logic}...")
        print(f"  Input:  {json_file}")

        if output_dir is None:
            print("  Truncation statistics only (no output)\n")
            try:
                encode_all_desc_from_descriptions_file(
                    json_path=str(json_file),
                    output_csv_path=None,
                    model_name=args.model_name,
                    normalize=False,
                    batch_size=8,
                    show_progress=True,
                    show_trunc_stats=True,
                    is_setfit=args.setfit,
                )
            except Exception as e:
                print(f"  Error: {e}\n", file=sys.stderr)
            continue

        logic_dir = output_dir / logic
        logic_dir.mkdir(parents=True, exist_ok=True)
        features_csv = logic_dir / "features.csv"
        times_csv = logic_dir / "extraction_times.csv"
        print(f"  Output: {logic_dir}/ (features.csv, extraction_times.csv)")

        try:
            # Get paths for extraction_times (same order as encoder will use)
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            paths = [p for p in data if isinstance(data.get(p), dict) and "description" in data[p]]
            n = len(paths)

            t0 = time.perf_counter()
            csv_path = encode_all_desc_from_descriptions_file(
                json_path=str(json_file),
                output_csv_path=str(features_csv),
                model_name=args.model_name,
                normalize=False,
                batch_size=8,
                show_progress=True,
                show_trunc_stats=args.trunc_stats,
                is_setfit=args.setfit,
            )
            elapsed = time.perf_counter() - t0

            if csv_path is None:
                print("  Truncation statistics shown (no encoding performed)\n")
                continue

            # Write extraction_times.csv: path, time_sec (distribute total time), failed=0
            time_per_path = elapsed / n if n else 0.0
            with open(times_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["path", "time_sec", "failed"])
                for p in paths:
                    writer.writerow([p, time_per_path, 0])

            print(f"  Success! {n} instances in {elapsed:.2f}s -> {logic_dir}\n")
        except Exception as e:
            print(f"  Error processing {logic}: {e}\n", file=sys.stderr)
            return 1

    print("All processing complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())