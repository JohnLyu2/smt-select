import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from itertools import permutations
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, models


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def normalize_path(p: str) -> str:
    return p.strip().replace("\\", "/")


def load_descriptions(csv_path: str | Path) -> Dict[str, str]:
    desc = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            desc[normalize_path(row["path"])] = row["description"].strip()
    return desc


def load_solver_results(csv_path: str | Path):
    """
    Returns:
      solver_names: list[str]
      perf[path] = [(solved, runtime), ...]
    """
    perf = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        header_row0 = next(reader)
        next(reader)

        solver_names = [c for c in header_row0 if c and c != "path"]
        solver_cols = [1 + 2 * i for i in range(len(solver_names))]

        for row in reader:
            if not row or not row[0]:
                continue

            path = normalize_path(row[0])
            results = []

            for col in solver_cols:
                try:
                    solved = int(row[col])
                    runtime = float(row[col + 1])
                    results.append((solved, runtime))
                except Exception:
                    results.append((0, float("inf")))

            perf[path] = results

    return solver_names, perf


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
def build_pairwise_model(base_model_name: str) -> SentenceTransformer:
    transformer = models.Transformer(base_model_name)
    pooling = models.Pooling(transformer.get_word_embedding_dimension())

    emb_dim = transformer.get_word_embedding_dimension()

    classifier = models.Dense(
        in_features=emb_dim,
        out_features=2,
        activation_function=None,
    )

    return SentenceTransformer(modules=[transformer, pooling, classifier])


# ------------------------------------------------------------
# Dataset construction
# ------------------------------------------------------------

def build_pairwise_dataset(
    descriptions: Dict[str, str],
    solver_names: List[str],
    perf: Dict[str, List[Tuple[int, float]]],
):
    examples = []

    for path, desc in descriptions.items():
        if path not in perf:
            continue

        results = perf[path]

        for i, j in permutations(range(len(solver_names)), 2):
            solved_i, time_i = results[i]
            solved_j, time_j = results[j]

            if solved_i == 1 and solved_j == 1:
                label = int(time_i < time_j)

                text = (
                    f"SMT description: {desc} "
                    f"Solver A: {solver_names[i]} "
                    f"Solver B: {solver_names[j]}"
                )

                examples.append((text, label))

    return examples


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------


def collate_fn(batch):
    texts, labels = zip(*batch)
    return list(texts), torch.tensor(labels, dtype=torch.long)


def finetune_pairwise(
    desc_csv: str | Path,
    solver_train_csv: str | Path,
    solver_test_csv: str | Path,
    model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
    output_csv_path: str | Path | None = None,
    batch_size: int = 16,
    epochs: int = 5,
    lr: float = 2e-5,
    show_trunc_stats: bool = False):


    desc_dict = load_descriptions(desc_csv)
    solver_names, train_perf = load_solver_results(solver_train_csv)
    _, test_perf = load_solver_results(solver_test_csv)


    train_dataset = build_pairwise_dataset(
        desc_dict, solver_names, train_perf
    )

    if not train_dataset:
        raise ValueError("No pairwise training data created")

    print(f"Pairwise training examples: {len(train_dataset)}")

    model = build_pairwise_model(model_name)

    descriptions = list(desc_dict.values())
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

    model.train()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # ---------- training loop ----------
    for epoch in range(epochs):
        total_loss = 0.0

        for texts, labels in train_loader:
            optimizer.zero_grad()

            features = model.tokenize(texts)
            output = model(features)
            logits = output["sentence_embedding"]
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - loss: {total_loss / len(train_loader):.4f}")

    # ---------- inference on test ----------
    model.eval()

    predictions = []

    for path, desc in desc_dict.items():
        if path not in test_perf:
            continue

        wins = Counter()

        for i, j in permutations(range(len(solver_names)), 2):
            text = (
                f"SMT description: {desc} "
                f"Solver A: {solver_names[i]} "
                f"Solver B: {solver_names[j]}"
            )

            with torch.no_grad():
                logits = model.encode(text, convert_to_tensor=True)
                pred = logits.argmax().item()

            if pred == 1:  # A faster than B
                wins[solver_names[i]] += 1
            else:
                wins[solver_names[j]] += 1

        if wins:
            predicted_solver = wins.most_common(1)[0][0]
            predictions.append((path, predicted_solver))


# ---------- write predictions ----------
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "predicted_solver"])
        writer.writerows(predictions)

    print(f"Saved predictions to {output_csv_path}")


def main():
    parser = argparse.ArgumentParser(
            description="Train a pairwise SMT solver selection model and run inference on a test set.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

    parser.add_argument(
            "--desc_csv",
            type=str,
            required=True,
            help="CSV file containing instance descriptions (pathToFile, description)",
        )

    parser.add_argument(
            "--solver_train_csv",
            type=str,
            required=True,
            help="CSV file containing solver results for training instances",
        )

    parser.add_argument(
            "--solver_test_csv",
            type=str,
            required=True,
            help="CSV file containing solver results for test instances",
        )

    parser.add_argument(
            "--model",
            type=str,
            default="sentence-transformers/all-MiniLM-L12-v2",
            help="Base SentenceTransformer model",
        )

    parser.add_argument(
            "--epochs",
            type=int,
            default=5,
            help="Number of training epochs",
        )

    parser.add_argument(
            "--batch_size",
            type=int,
            default=16,
            help="Training batch size",
        )

    parser.add_argument(
            "--lr",
            type=float,
            default=2e-5,
            help="Learning rate",
        )


    parser.add_argument(
            "--output_predictions",
            type=str,
            default="pairwise_predictions.csv",
            help="Output CSV file with predictions (path, predicted_solver)",
        )

    args = parser.parse_args()

    #try:
    finetune_pairwise(
                desc_csv=args.desc_csv,
                solver_train_csv=args.solver_train_csv,
                solver_test_csv=args.solver_test_csv,
                model_name=args.model,
                output_csv_path=args.output_predictions,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
            )
    #except Exception as e:
    #        print(f"Error: {e}", file=sys.stderr)
    #        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
