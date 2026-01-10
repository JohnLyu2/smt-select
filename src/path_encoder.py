"""Text embedding function for SMT-LIB benchmark paths."""

import argparse
import json
import csv
import sys
from pathlib import Path
import numpy as np

# Import reusable functions from desc_encoder
from src.desc_encoder import encode_text, get_embedding_model


def encode_all_paths(
    json_path: str | Path,
    output_csv_path: str | Path | None = None,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    normalize: bool = False,
    batch_size: int = 32,
    show_progress: bool = True,
) -> str:
    """
    Encode all benchmark paths from a JSON file and write to CSV.

    This function extracts the smtlib_path field from each benchmark in the JSON file
    and creates text embeddings using a sentence transformer model. The embeddings
    capture semantic information encoded in the path structure (logic, family, benchmark name).

    Args:
        json_path: Path to JSON file containing benchmark data (e.g., data/raw_jsons/ABV.json)
        output_csv_path: Path to output CSV file. If None, defaults to same directory as JSON
                         with .csv extension (e.g., data/raw_jsons/ABV.csv)
        model_name: Name of the sentence transformer model to use
        normalize: Whether to normalize embeddings to unit length
        batch_size: Batch size for processing multiple texts
        show_progress: Whether to show progress bar when encoding

    Returns:
        Path to the output CSV file as a string

    Examples:
        >>> csv_path = encode_all_paths("data/raw_jsons/ABV.json")
        >>> # Output will be written to data/raw_jsons/ABV.csv

        >>> csv_path = encode_all_paths(
        ...     "data/raw_jsons/ABV.json",
        ...     output_csv_path="data/features/path_emb/all_mpnet_base_v2/ABV.csv"
        ... )
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # Determine output path
    if output_csv_path is None:
        output_csv_path = json_path.with_suffix(".csv")
    else:
        output_csv_path = Path(output_csv_path)

    # Load JSON file
    with open(json_path, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    if not benchmarks:
        raise ValueError(f"JSON file is empty or contains no benchmarks: {json_path}")

    # Extract paths
    paths = []
    for benchmark in benchmarks:
        smtlib_path = benchmark.get("smtlib_path", "")

        # Use placeholder path if missing or empty (similar to description handling)
        if not smtlib_path or not smtlib_path.strip():
            logic = benchmark.get("logic", "unknown")
            family = benchmark.get("family") or "unknown"  # Handle None explicitly
            benchmark_name = benchmark.get("benchmark_name", "unknown.smt2")
            # Create placeholder path for embedding
            smtlib_path = f"{logic}/{family}/{benchmark_name}"

        paths.append(smtlib_path.strip())

    if not paths:
        raise ValueError(f"No valid paths found in JSON file: {json_path}")

    # Encode all paths in batch
    embeddings = encode_text(
        paths,
        model_name=model_name,
        normalize=normalize,
        batch_size=batch_size,
        show_progress=show_progress,
    )

    # Get embedding dimension (embeddings is always 2D when encoding multiple texts)
    embedding_dim = embeddings.shape[1]

    # Write to CSV
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        # Create column names: path, emb_0, emb_1, ..., emb_{dim-1}
        fieldnames = ["path"] + [f"emb_{i}" for i in range(embedding_dim)]
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)

        # Write each benchmark's embedding
        for path, embedding in zip(paths, embeddings):
            row = [path] + embedding.tolist()
            writer.writerow(row)

    return str(output_csv_path)


def main():
    """Command-line interface for encode_all_paths."""
    parser = argparse.ArgumentParser(
        description="Encode all benchmark paths from a JSON file and write to CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - output will be saved to data/raw_jsons/ABV.csv
  python -m src.path_encoder data/raw_jsons/ABV.json

  # Specify custom output path
  python -m src.path_encoder data/raw_jsons/ABV.json -o data/features/path_emb/all_mpnet_base_v2/ABV.csv

  # Use different model and normalize embeddings
  python -m src.path_encoder data/raw_jsons/ABV.json --model all-MiniLM-L6-v2 --normalize

  # Adjust batch size and disable progress bar
  python -m src.path_encoder data/raw_jsons/ABV.json --batch-size 64 --no-progress
        """,
    )
    parser.add_argument(
        "json_path",
        type=str,
        help="Path to JSON file containing benchmark data (e.g., data/raw_jsons/ABV.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to output CSV file. If not specified, defaults to same directory as JSON with .csv extension",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Name of the sentence transformer model to use (default: sentence-transformers/all-mpnet-base-v2)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize embeddings to unit length",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing multiple texts (default: 32)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar when encoding",
    )

    args = parser.parse_args()

    try:
        print(f"Loading JSON file: {args.json_path}")
        csv_path = encode_all_paths(
            json_path=args.json_path,
            output_csv_path=args.output,
            model_name=args.model,
            normalize=args.normalize,
            batch_size=args.batch_size,
            show_progress=not args.no_progress,
        )
        print(f"Success! Path embeddings saved to: {csv_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
