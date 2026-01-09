import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch import nn
from torch.utils.data import DataLoader
import json
import argparse
import json
import csv
import sys
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Tuple
from sentence_transformers import models

# Global model cache
_model_cache = {}


def get_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
) -> SentenceTransformer:
    """
    Get or load an embedding model (cached for efficiency).

    Args:
        model_name: Name of the sentence transformer model to use

    Returns:
        SentenceTransformer model instance
    """
    global _model_cache
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def encode_text(
    text: str | list[str],
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    normalize: bool = False,
    batch_size: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Transform text description(s) into embedding vector(s).

    Args:
        text: Single text string or list of text strings to encode
        model_name: Name of the sentence transformer model to use
        normalize: Whether to normalize embeddings to unit length
        batch_size: Batch size for processing multiple texts
        show_progress: Whether to show progress bar when encoding multiple texts

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
    model = get_embedding_model(model_name)

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
            description = f"This a {logic} benchmark from the family {family}"

        paths.append(smtlib_path)
        descriptions.append(description.strip())

    if not descriptions:
        raise ValueError(f"No valid descriptions found in JSON file: {json_path}")

    # Check for truncation before encoding
    model = get_embedding_model(model_name)
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

    # Calculate statistics
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

    # Print truncation statistics if requested
    if show_trunc_stats:
        print("  Truncation Statistics:")
        print(f"    Model max sequence length: {max_seq_length} tokens")
        print(f"    Total descriptions: {total_descriptions}")
        print(
            f"    Truncated descriptions: {truncated_count} ({truncation_percentage:.2f}%)"
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




def load_descriptions(csv_path: str | Path) -> dict[str, str]:
    """
    Load path → description mapping.
    """
    df = pd.read_csv(csv_path)
    if "path" not in df.columns or "description" not in df.columns:
        raise ValueError("Description CSV must have columns: path, description")

    return dict(zip(df["path"], df["description"]))


def select_fastest_solver(
    results: list[tuple[int, float]],
    solver_dict: dict[int, str],
) -> str | None:
    """
    Select the fastest solver that solved the instance.

    Returns:
        solver name, or None if no solver solved it
    """
    best_solver = None
    best_time = float("inf")

    for solver_id, (is_solved, runtime) in enumerate(results):
        if is_solved == 1 and runtime > 0:
            if runtime < best_time:
                best_time = runtime
                best_solver = solver_dict[solver_id]

    return best_solver

def build_best_solver_labels(
    multi_perf_dict: dict[str, list[tuple[int, float]]],
    solver_dict: dict[int, str],
) -> dict[str, str]:
    labels = {}

    for path, results in multi_perf_dict.items():
        best_solver = select_fastest_solver(results, solver_dict)
        if best_solver is not None:
            labels[path] = best_solver

    return labels

def load_best_solver_labels(csv_path: str | Path) -> dict[str, str]:
    multi_perf_dict: Dict[str, List[Tuple[int, float]]] = {}
    solver_dict: Dict[int, str] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        # Read header rows
        header_row0 = next(reader) # path,OpenSMT,,SMTInterpol,,...
        print(header_row0)
        next(reader)  # Skip second header row: ,solved,runtime,solved,runtime,...

        # Extract solver names from header_row0 (skip empty columns)
        solver_names = []
        for i, cell in enumerate(header_row0):
            if cell and cell != "path":  # Skip empty cells and 'path' column
                solver_names.append(cell)

        # Build solver_dict: id -> solver name
        for idx, solver_name in enumerate(solver_names):
            solver_dict[idx] = solver_name

        # Calculate column indices for each solver's solved and runtime
        # Solver columns start at index 1, then every other column (1, 3, 5, 7, 9, ...)
        # Result is at solver_col, runtime is at solver_col + 1
        num_solvers = len(solver_names)
        solver_cols = [1 + 2 * i for i in range(num_solvers)]


        for row in reader:
            if not row or not row[0]:  # Skip empty rows
                continue

            path = row[0]
            results = []

            # Extract (is_solved, wc_time) pairs for each solver
            for solver_col in solver_cols:
                if solver_col < len(row) and solver_col + 1 < len(row):
                    try:
                        is_solved = int(row[solver_col])
                        wc_time = float(row[solver_col + 1])
                        results.append((is_solved, wc_time))
                    except (ValueError, IndexError):
                        # Handle missing or invalid data
                        results.append((0, 0.0))
                else:
                    # Handle missing columns
                    results.append((0, 0.0))

            multi_perf_dict[path] = results
    return build_best_solver_labels(multi_perf_dict, solver_dict)

def get_embedding_model(
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
) -> SentenceTransformer:
    """
    Get or load an embedding model (cached for efficiency).

    Args:
        model_name: Name of the sentence transformer model to use

    Returns:
        SentenceTransformer model instance
    """
    global _model_cache
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def finetune(
    json_path: str | Path,
    solver_csv: str | Path,
    output_csv_path: str | Path | None = None,
    model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
    normalize: bool = False,
    batch_size: int = 32,
    show_progress: bool = True,
    show_trunc_stats: bool = False,
    epochs: int = 5,
    lr: float = 2e-5,
):
    """
    Fine-tune a SentenceTransformer to predict the best solver.
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
    desc_map = {}
    descriptions = []
    paths = []
    for benchmark in benchmarks:
        smtlib_path = benchmark.get("smtlib_path", "")
        description = benchmark.get("description", "")

        # Use placeholder description if missing or empty
        if not description or not description.strip():
            logic = benchmark.get("logic", "unknown")
            family = benchmark.get("family", "unknown")
            description = f"This a {logic} benchmark from the family {family}"

        descriptions.append(description.strip())
        paths.append(smtlib_path)
        desc_map[smtlib_path] = description.strip()


    solver_map = load_best_solver_labels(solver_csv)

    # Join on path
    texts = []
    solvers = []

    for path, solver in solver_map.items():
        desc = desc_map.get(path)
        if desc and desc.strip():
            texts.append(desc.strip())
            solvers.append(solver)

    if not texts:
        raise ValueError("No overlapping (description, solver) pairs found")

    # Encode solver labels
    solver_to_id = {s: i for i, s in enumerate(sorted(set(solvers)))}
    id_to_solver = {i: s for s, i in solver_to_id.items()}

    labels = [solver_to_id[s] for s in solvers]

    print("Solver label mapping:")
    for s, i in solver_to_id.items():
        print(f"  {s} → {i}")

    # Check for truncation before encoding
    model = get_embedding_model(model_name)
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

        # Calculate statistics
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

    # Print truncation statistics if requested
    if show_trunc_stats:
        print("  Truncation Statistics:")
        print(f"    Model max sequence length: {max_seq_length} tokens")
        print(f"    Total descriptions: {total_descriptions}")
        print(
                f"    Truncated descriptions: {truncated_count} ({truncation_percentage:.2f}%)"
            )
        print(f"    Average token length: {avg_token_length:.1f} tokens")
        if truncated_count > 0:
            print(
                    f"    Average tokens truncated: {avg_truncated_tokens:.1f} tokens per truncated text"
                )
            print(f"    Total tokens truncated: {truncated_tokens} tokens")
            # Skip embedding when only showing truncation statistics
        return None


    # finetuning of the model
    loss = losses.SoftmaxLoss(
            model=model,
            sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
            num_labels=len(solver_to_id),
        )

    train_examples = [
            InputExample(texts=[text, text], label=label)
            for text, label in zip(texts, labels)
        ]

    train_loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size, )
    warmup_steps = int(len(train_loader) * epochs * 0.1)

    model.fit(
            train_objectives=[(train_loader, loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": lr},
            show_progress_bar=True,
    )


    # actual embedding
    # Encode all descriptions in batch
    # Encode descriptions
    embeddings = model.encode(
        descriptions,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
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
        "--json_path",
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
        "--finetune",
        action="store_true",
        help="Fine-tune the model for solver classification",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of fine-tuning epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate for fine-tuning",
    )

    parser.add_argument(
        "--solver_csv",
        type=str,
        help="Path to csv solver data",
    )


    args = parser.parse_args()

    try:
        if args.finetune:
            csv_path = finetune(
                json_path=args.json_path,
                solver_csv=args.solver_csv,
                model_name=args.model,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
            )
        else:
            print(f"Loading JSON file: {args.json_path}")
            csv_path = encode_all_desc(
                json_path=args.json_path,
                output_csv_path=args.output,
                model_name=args.model,
                normalize=args.normalize,
                batch_size=args.batch_size,
                show_progress=not args.no_progress,
                show_trunc_stats=args.trunc_stats,
            )
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
