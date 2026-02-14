"""Text embedding function for SMT descriptions."""

import argparse
import json
import csv
import sys
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

# Global model cache
_model_cache = {}


def get_embedding_model(
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    is_setfit: bool = False,
) -> SentenceTransformer:
    """
    Get or load an embedding model (cached for efficiency).

    Args:
        model_name: Name of the sentence transformer model to use
        is_setfit: Whether the model is a SetFit model (extracts backbone)

    Returns:
        SentenceTransformer model instance
    """
    global _model_cache
    cache_key = f"{model_name}:setfit={is_setfit}"
    if cache_key not in _model_cache:
        if is_setfit:
            from setfit import SetFitModel
            setfit_model = SetFitModel.from_pretrained(model_name)
            _model_cache[cache_key] = setfit_model.model_body
        else:
            _model_cache[cache_key] = SentenceTransformer(model_name)
    return _model_cache[cache_key]


def encode_text(
    text: str | list[str],
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    normalize: bool = False,
    batch_size: int = 32,
    show_progress: bool = False,
    is_setfit: bool = False,
) -> np.ndarray:
    """
    Transform text description(s) into embedding vector(s).

    Args:
        text: Single text string or list of text strings to encode
        model_name: Name of the sentence transformer model to use
        normalize: Whether to normalize embeddings to unit length
        batch_size: Batch size for processing multiple texts
        show_progress: Whether to show progress bar when encoding multiple texts
        is_setfit: Whether the model is a SetFit model (extracts backbone)

    Returns:
        numpy array of embeddings:
        - For single text: shape (embedding_dim,)
        - For multiple texts: shape (num_texts, embedding_dim)

    Examples:
        >>> # Single text
        >>> embedding = encode_text("This is a description")
        >>> print(embedding.shape)  # (768,)

        >>> # Multiple texts
        >>> embeddings = encode_text(["Text 1", "Text 2", "Text 3"])
        >>> print(embeddings.shape)  # (3, 768)
    """
    model = get_embedding_model(model_name, is_setfit=is_setfit)

    # Handle single text vs list of texts
    is_single = isinstance(text, str)
    texts = [text] if is_single else text

    # Encode texts
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
        batch_size=batch_size,
        normalize_embeddings=normalize,
    )

    # Return single embedding for single text, array for multiple
    if is_single:
        return embeddings[0]
    return embeddings


def encode_all_desc(
    json_path: str | Path,
    output_csv_path: str | Path | None = None,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    normalize: bool = False,
    batch_size: int = 32,
    show_progress: bool = True,
    show_trunc_stats: bool = False,
    is_setfit: bool = False,
) -> str | None:
    """
    Encode all benchmark descriptions from a JSON file and write to CSV.

    Args:
        json_path: Path to JSON file containing benchmark data (e.g., data/raw_jsons/ABV.json)
        output_csv_path: Path to output CSV file. If None, defaults to same directory as JSON
                         with .csv extension (e.g., data/raw_jsons/ABV.csv)
        model_name: Name of the sentence transformer model to use
        normalize: Whether to normalize embeddings to unit length
        batch_size: Batch size for processing multiple texts
        show_progress: Whether to show progress bar when encoding
        show_trunc_stats: Whether to show truncation statistics only (no encoding performed) (default: False)
        is_setfit: Whether the model is a SetFit model (extracts backbone)

    Returns:
        Path to the output CSV file, or None if show_trunc_stats is True (only statistics shown, no encoding performed)

    Examples:
        >>> csv_path = encode_all_desc("data/raw_jsons/ABV.json")
        >>> # Output will be written to data/raw_jsons/ABV.csv
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

    # Extract descriptions and smtlib paths
    paths = []
    descriptions = []
    for benchmark in benchmarks:
        smtlib_path = benchmark.get("smtlib_path", "")
        description = benchmark.get("description", "")

        # Use placeholder description if missing or empty
        if not description or not description.strip():
            logic = benchmark.get("logic", "unknown")
            family = benchmark.get("family", "unknown")
            description = f"This is a {logic} benchmark from the family {family}"

        paths.append(smtlib_path)
        descriptions.append(description.strip())

    if not descriptions:
        raise ValueError(f"No valid descriptions found in JSON file: {json_path}")

    # Check for truncation before encoding
    model = get_embedding_model(model_name, is_setfit=is_setfit)
    if model is None:
        raise ValueError(f"Failed to load model: {model_name}")

    # Get tokenizer from the model (standard way: first module's tokenizer)
    # model[0] is typically the transformer/encoder module which contains the tokenizer
    try:
        # Try standard location first: model[0].tokenizer
        if (
            hasattr(model, "__len__")
            and len(model) > 0
            and hasattr(model[0], "tokenizer")
        ):
            tokenizer = model[0].tokenizer
        # Fallback: try direct access
        elif hasattr(model, "tokenizer"):
            tokenizer = model.tokenizer
        else:
            raise AttributeError(
                "Tokenizer not found in model. "
                "Expected tokenizer at model[0].tokenizer or model.tokenizer"
            )
    except (AttributeError, KeyError, IndexError, TypeError) as e:
        raise AttributeError(f"Failed to access tokenizer from model: {e}") from e

    truncated_count = 0
    total_tokens = 0
    truncated_tokens = 0

    # Verify tokenizer is callable
    if not hasattr(tokenizer, "__call__"):
        raise TypeError("Tokenizer is not callable")

    # Use native batch tokenization (more efficient than looping)
    # Tokenize all descriptions at once without truncation
    tokenized = tokenizer(
        descriptions,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_tensors=None,  # Return Python lists, not tensors
    )

    # Verify tokenized output has expected structure
    # BatchEncoding objects have input_ids as an attribute
    if not hasattr(tokenized, "input_ids"):
        actual_type = type(tokenized).__name__
        raise ValueError(
            f"Tokenizer output missing 'input_ids' attribute. Got type: {actual_type}"
        )

    input_ids = tokenized.input_ids

    # Get max sequence length from tokenizer
    if hasattr(tokenizer, "model_max_length"):
        max_seq_length = tokenizer.model_max_length
    else:
        raise AttributeError("Tokenizer does not have 'model_max_length' attribute. ")

    # Extract token lengths from input_ids (native tokenizer output)
    token_lengths = [len(ids) for ids in input_ids]

    # Calculate statistics
    for token_count in token_lengths:
        total_tokens += token_count
        if token_count > max_seq_length:
            truncated_count += 1
            truncated_tokens += token_count - max_seq_length

    total_descriptions = len(descriptions)
    truncation_percentage = (
        (truncated_count / total_descriptions * 100) if total_descriptions > 0 else 0
    )
    avg_truncated_tokens = (
        (truncated_tokens / truncated_count) if truncated_count > 0 else 0
    )
    avg_token_length = (
        total_tokens / total_descriptions if total_descriptions > 0 else 0
    )
    min_token_length = min(token_lengths) if token_lengths else 0
    max_token_length = max(token_lengths) if token_lengths else 0
    quantiles = np.quantile(token_lengths, [0.25, 0.5, 0.75]) if token_lengths else []

    # Print truncation statistics if requested
    if show_trunc_stats:
        print("  Truncation Statistics:")
        print(f"    Model max sequence length: {max_seq_length} tokens")
        print(f"    Total descriptions: {total_descriptions}")
        print(
            f"    Truncated descriptions: {truncated_count} ({truncation_percentage:.2f}%)"
        )
        print(
            f"    Token length min/median/max: {min_token_length}/"
            f"{(quantiles[1] if len(quantiles) > 1 else 0):.1f}/{max_token_length}"
        )
        if len(quantiles) > 0:
            print(
                f"    Token length quantiles (25/50/75%): "
                f"{quantiles[0]:.1f}/{quantiles[1]:.1f}/{quantiles[2]:.1f}"
            )
        print(f"    Average token length: {avg_token_length:.1f} tokens")
        if truncated_count > 0:
            print(
                f"    Average tokens truncated: {avg_truncated_tokens:.1f} tokens per truncated text"
            )
            print(f"    Total tokens truncated: {truncated_tokens} tokens")
        # Skip embedding when only showing truncation statistics
        return None

    # Encode all descriptions in batch
    embeddings = encode_text(
        descriptions,
        model_name=model_name,
        normalize=normalize,
        batch_size=batch_size,
        show_progress=show_progress,
        is_setfit=is_setfit,
    )

    # Get embedding dimension (embeddings is always 2D when encoding multiple texts)
    embedding_dim = embeddings.shape[1]

    # Write to CSV
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
    """Command-line interface for encode_all_desc."""
    parser = argparse.ArgumentParser(
        description="Encode all benchmark descriptions from a JSON file and write to CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - output will be saved to data/raw_jsons/ABV.csv
  python -m src.desc_encoder data/raw_jsons/ABV.json

  # Specify custom output path
  python -m src.desc_encoder data/raw_jsons/ABV.json -o data/embeddings/ABV.csv

  # Use different model and normalize embeddings
  python -m src.desc_encoder data/raw_jsons/ABV.json --model all-MiniLM-L6-v2 --normalize

  # Adjust batch size and disable progress bar
  python -m src.desc_encoder data/raw_jsons/ABV.json --batch-size 64 --no-progress
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
        default=8,
        help="Batch size for processing multiple texts (default: 8)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar when encoding",
    )
    parser.add_argument(
        "--trunc-stats",
        action="store_true",
        help="Show truncation statistics and embedding is skipped",
    )
    parser.add_argument(
        "--setfit",
        action="store_true",
        help="Model is a SetFit model (extracts sentence transformer backbone)",
    )

    args = parser.parse_args()

    try:
        print(f"Loading JSON file: {args.json_path}")
        csv_path = encode_all_desc(
            json_path=args.json_path,
            output_csv_path=args.output,
            model_name=args.model,
            normalize=args.normalize,
            batch_size=args.batch_size,
            show_progress=not args.no_progress,
            show_trunc_stats=args.trunc_stats,
            is_setfit=args.setfit,
        )
        if csv_path is not None:
            print(f"Success! Embeddings saved to: {csv_path}")
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
