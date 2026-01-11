#!/usr/bin/env python3
"""
LoRA + downstream tasks for SMT solver selection from natural-language descriptions.

Supports tasks:
  - pairwise: train binary classifier "Solver A faster than Solver B" using only instances
              where both solvers solved in TRAIN. Predict solver on TEST via tournament voting.
  - single:   train multi-class classifier to predict best solver (fastest among solved) on TRAIN.
              Predict best solver class on TEST.
  - regression: train multi-output regression to predict per-solver runtime (log1p by default) on TRAIN.
                Predict solver with min predicted runtime on TEST.

Inputs:
  --desc_csv: CSV with columns: path, description   (or pathToFile, description)
  --solver_train_csv: solver results CSV with 2-row header (as in your data)
  --solver_test_csv:  solver results CSV with 2-row header (as in your data)

Output:
  --output_predictions: CSV with columns: path, predicted_solver

Notes:
  - For pairwise training, we only create pairs for instances where BOTH solvers solved.
  - For single/regression, we require at least one solver solved (to form a label/target).
  - SentenceTransformer forward() returns a dict; logits are in output["sentence_embedding"].
  - DO NOT use .encode() during training; use tokenize()+forward to keep gradients.
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, models

# Optional LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType  # type: ignore
    _PEFT_AVAILABLE = True
except Exception:
    _PEFT_AVAILABLE = False


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def normalize_path(p: str) -> str:
    return p.strip().replace("\\", "/")


def load_descriptions(csv_path: str | Path) -> Dict[str, str]:
    """
    Accepts either:
      - columns: path, description
      - columns: pathToFile, description
    """
    csv_path = str(csv_path)
    desc: Dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        if "description" not in cols:
            raise ValueError("Description CSV must contain a 'description' column")

        if "path" in cols:
            path_col = "path"
        elif "pathToFile" in cols:
            path_col = "pathToFile"
        else:
            raise ValueError("Description CSV must contain 'path' or 'pathToFile' column")

        for row in reader:
            path = normalize_path(row[path_col])
            desc[path] = (row["description"] or "").strip()
    return desc


def load_solver_results(csv_path: str | Path) -> Tuple[List[str], Dict[str, List[Tuple[int, float]]]]:
    """
    Solver CSV format (two-row header):
      row0: path, Solver1, , Solver2, , ...
      row1:      result,runtime,result,runtime,...

    Returns:
      solver_names: list[str]  in the order they appear
      perf[path] = [(solved, runtime), ...] aligned with solver_names
    """
    perf: Dict[str, List[Tuple[int, float]]] = {}
    csv_path = str(csv_path)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        header_row0 = next(reader)  # e.g. ['path','Bitwuzla','','SMTInterpol','',...]
        _ = next(reader)           # skip row1

        solver_names = [c for c in header_row0 if c and c != "path"]
        num_solvers = len(solver_names)
        solver_cols = [1 + 2 * i for i in range(num_solvers)]

        for row in reader:
            if not row or not row[0]:
                continue

            path = normalize_path(row[0])
            results: List[Tuple[int, float]] = []
            for col in solver_cols:
                try:
                    solved = int(row[col])
                    runtime = float(row[col + 1])
                    results.append((solved, runtime))
                except Exception:
                    results.append((0, float("inf")))
            perf[path] = results

    return solver_names, perf


def intersect_paths(desc: Dict[str, str], perf: Dict[str, List[Tuple[int, float]]]) -> List[str]:
    return [p for p in desc.keys() if p in perf]


# ------------------------------------------------------------
# Labels / targets
# ------------------------------------------------------------

def best_solver_from_row(
    solver_names: List[str],
    row: List[Tuple[int, float]],
    require_runtime_positive: bool = True,
) -> Optional[int]:
    """
    Returns index of fastest solver among those that solved.
    If none solved, returns None.
    """
    best_idx = None
    best_time = float("inf")
    for i, (solved, t) in enumerate(row):
        if solved != 1:
            continue
        if require_runtime_positive and not (t > 0 and math.isfinite(t)):
            continue
        if t < best_time:
            best_time = t
            best_idx = i
    return best_idx


def runtimes_target(
    row: List[Tuple[int, float]],
    log1p: bool = True,
    timeout_fill: float = 1e6,
) -> torch.Tensor:
    """
    Returns per-solver runtime target vector.
    For unsolved, fill with timeout_fill.
    Optionally apply log1p transform.
    """
    vals: List[float] = []
    for solved, t in row:
        if solved == 1 and t > 0 and math.isfinite(t):
            v = t
        else:
            v = timeout_fill
        vals.append(math.log1p(v) if log1p else v)
    return torch.tensor(vals, dtype=torch.float32)


# ------------------------------------------------------------
# Model + LoRA
# ------------------------------------------------------------

def apply_lora_to_sentence_transformer(
    st_model: SentenceTransformer,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> SentenceTransformer:
    """
    Inject LoRA adapters into the HuggingFace transformer backbone used by SentenceTransformer.

    Default target_modules works for many BERT-like models:
      ["query", "key", "value"]

    If PEFT isn't installed, raises a clear error.
    """
    if not _PEFT_AVAILABLE:
        raise RuntimeError("peft is not installed. Install with: pip install peft")

    if target_modules is None:
        target_modules = ["query", "key", "value"]

    # SentenceTransformers' first module is usually models.Transformer, exposing auto_model
    hf_model = st_model[0].auto_model

    lora_cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules,
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    hf_model = get_peft_model(hf_model, lora_cfg)
    st_model[0].auto_model = hf_model
    return st_model


def build_model_with_head(
    base_model_name: str,
    out_dim: int,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> SentenceTransformer:
    transformer = models.Transformer(base_model_name)
    pooling = models.Pooling(transformer.get_word_embedding_dimension())
    emb_dim = transformer.get_word_embedding_dimension()

    head = models.Dense(
        in_features=emb_dim,
        out_features=out_dim,
        activation_function=None,  # logits (or raw regression outputs)
    )

    st = SentenceTransformer(modules=[transformer, pooling, head])

    if use_lora:
        st = apply_lora_to_sentence_transformer(
            st,
            r=lora_r,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
    return st


# ------------------------------------------------------------
# Datasets
# ------------------------------------------------------------

@dataclass
class Example:
    text: str
    y: torch.Tensor  # shape: () for classification OR (K,) for regression


def build_dataset_pairwise(
    desc: Dict[str, str],
    solver_names: List[str],
    perf: Dict[str, List[Tuple[int, float]]],
) -> List[Example]:
    data: List[Example] = []

    for path, d in desc.items():
        if path not in perf:
            continue

        results = perf[path]

        # Skip instances where nobody solved
        if not any(solved == 1 for solved, _ in results):
            continue

        for i, j in permutations(range(len(solver_names)), 2):
            si, _ = results[i]
            sj, _ = results[j]

            # Strong signal: exactly one solver solves
            if si != sj:
                label = 1 if si > sj else 0
                text = (
                    f"SMT description: {d} "
                    f"Solver A: {solver_names[i]} "
                    f"Solver B: {solver_names[j]}"
                )
                data.append(Example(text=text, y=torch.tensor(label, dtype=torch.long)))

    return data



def build_dataset_single(
    desc: Dict[str, str],
    solver_names: List[str],
    perf: Dict[str, List[Tuple[int, float]]],
) -> List[Example]:
    """
    Single-solver classification:
    - keep instance if at least one solver solved
    - label = fastest solver among those that solved
    """
    data: List[Example] = []

    for path, d in desc.items():
        if path not in perf:
            continue

        results = perf[path]

        # Collect solvers that solved
        solved = [
            (i, t) for i, (s, t) in enumerate(results)
            if s == 1 and math.isfinite(t) and t > 0
        ]

        if not solved:
            continue  # nobody solved → drop

        # Pick fastest among solvers that solved
        best_solver, _ = min(solved, key=lambda x: x[1])

        data.append(
            Example(
                text=d,
                y=torch.tensor(best_solver, dtype=torch.long),
            )
        )

    return data



def build_dataset_regression(
    desc: Dict[str, str],
    solver_names: List[str],
    perf: Dict[str, List[Tuple[int, float]]],
    timeout: float = 1200.0,
    log1p: bool = True,
) -> List[Example]:
    data: List[Example] = []

    for path, d in desc.items():
        if path not in perf:
            continue

        results = perf[path]

        # Keep only instances where at least one solver solved
        if not any(s == 1 for s, _ in results):
            continue

        y = []
        for solved, t in results:
            if solved == 1 and math.isfinite(t) and t > 0:
                val = math.log1p(t) if log1p else t
            else:
                val = math.log1p(timeout) if log1p else timeout
            y.append(val)

        data.append(
            Example(
                text=d,
                y=torch.tensor(y, dtype=torch.float),
            )
        )

    return data




def collate_fn(batch: List[Example]):
    texts = [ex.text for ex in batch]
    ys = torch.stack([ex.y if ex.y.ndim > 0 else ex.y.view(1) for ex in batch], dim=0)
    return texts, ys


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train_loop(
    model: SentenceTransformer,
    loader: DataLoader,
    task: str,
    lr: float,
    epochs: int,
    device: str,
):
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    if task in {"pairwise", "single"}:
        loss_fn = nn.CrossEntropyLoss()
    elif task == "regression":
        loss_fn = nn.MSELoss()
    else:
        raise ValueError(f"Unknown task: {task}")

    for epoch in range(epochs):
        total = 0.0
        for texts, y in loader:
            optimizer.zero_grad()

            features = model.tokenize(texts)
            # move tokenized tensors to device
            for k, v in features.items():
                if isinstance(v, torch.Tensor):
                    features[k] = v.to(device)

            out = model(features)
            logits = out["sentence_embedding"]  # Tensor

            if task in {"pairwise", "single"}:
                # y shape is (B,1) from collate_fn, convert to (B,)
                y_cls = y.squeeze(1).to(device)
                loss = loss_fn(logits, y_cls)
            else:
                y_reg = y.to(device)
                loss = loss_fn(logits, y_reg)

            loss.backward()
            optimizer.step()
            total += float(loss.item())

        print(f"Epoch {epoch+1}/{epochs} - loss: {total/len(loader):.4f}")


# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------

@torch.no_grad()
def predict_single(
    model: SentenceTransformer,
    solver_names: List[str],
    desc_map: Dict[str, str],
    test_perf: Dict[str, List[Tuple[int, float]]],
    device: str,
) -> List[Tuple[str, str]]:
    model.to(device)
    model.eval()

    preds: List[Tuple[str, str]] = []
    for path, d in desc_map.items():
        if path not in test_perf:
            continue
        logits = model.encode(d, convert_to_tensor=True, device=device)
        idx = int(torch.argmax(logits).item())
        idx = max(0, min(idx, len(solver_names) - 1))
        preds.append((path, solver_names[idx]))
    return preds


@torch.no_grad()
def predict_regression(
    model: SentenceTransformer,
    solver_names: List[str],
    desc_map: Dict[str, str],
    test_perf: Dict[str, List[Tuple[int, float]]],
    device: str,
) -> List[Tuple[str, str]]:
    model.to(device)
    model.eval()

    preds: List[Tuple[str, str]] = []
    for path, d in desc_map.items():
        if path not in test_perf:
            continue
        yhat = model.encode(d, convert_to_tensor=True, device=device)  # (K,)
        # choose minimal predicted runtime (note: if trained on log1p, monotonic so still OK)
        idx = int(torch.argmin(yhat).item())
        idx = max(0, min(idx, len(solver_names) - 1))
        preds.append((path, solver_names[idx]))
    return preds


@torch.no_grad()
def predict_pairwise_tournament(
    model: SentenceTransformer,
    solver_names: List[str],
    desc_map: Dict[str, str],
    test_perf: Dict[str, List[Tuple[int, float]]],
    device: str,
) -> List[Tuple[str, str]]:
    model.to(device)
    model.eval()

    preds: List[Tuple[str, str]] = []
    for path, d in desc_map.items():
        if path not in test_perf:
            continue

        wins = Counter()
        for i, j in permutations(range(len(solver_names)), 2):
            text = f"SMT description: {d} Solver A: {solver_names[i]} Solver B: {solver_names[j]}"
            logits = model.encode(text, convert_to_tensor=True, device=device)  # (2,)
            pred = int(torch.argmax(logits).item())  # 1 => A faster, 0 => B faster/tie
            if pred == 1:
                wins[solver_names[i]] += 1
            else:
                wins[solver_names[j]] += 1

        if wins:
            preds.append((path, wins.most_common(1)[0][0]))
    return preds


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LoRA downstream tasks for solver selection (pairwise/single/regression).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--task", choices=["pairwise", "single", "regression"], required=True)

    parser.add_argument("--desc_csv", type=str, required=True,
                        help="CSV with columns (path, description) or (pathToFile, description)")
    parser.add_argument("--solver_train_csv", type=str, required=True,
                        help="Solver CSV for training (two-row header)")
    parser.add_argument("--solver_test_csv", type=str, required=True,
                        help="Solver CSV for test/inference (two-row header)")

    parser.add_argument("--model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)

    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")

    parser.add_argument("--use_lora", action="store_true", help="Enable LoRA adapters (requires peft)")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--output_predictions", type=str, default="predictions.csv",
                        help="Output CSV path (path, predicted_solver)")

    # regression options
    parser.add_argument("--reg_log1p", action="store_true", help="Train regression on log1p(runtime)")
    parser.add_argument("--timeout_fill", type=float, default=1e6,
                        help="Runtime fill value for unsolved instances in regression targets")

    args = parser.parse_args()

    if args.use_lora and not _PEFT_AVAILABLE:
        print("Error: --use_lora requires peft. Install with: pip install peft", file=sys.stderr)
        return 1

    # Load descriptions + solver results
    desc_all = load_descriptions(args.desc_csv)

    solver_names_train, train_perf = load_solver_results(args.solver_train_csv)
    solver_names_test, test_perf = load_solver_results(args.solver_test_csv)

    if solver_names_train != solver_names_test:
        print("Warning: solver lists differ between train and test CSVs.", file=sys.stderr)
        print("Train solvers:", solver_names_train, file=sys.stderr)
        print("Test  solvers:", solver_names_test, file=sys.stderr)

    # We'll use train solver order as canonical
    solver_names = solver_names_train

    # Restrict desc dict to only paths present in each split
    train_paths = set(intersect_paths(desc_all, train_perf))
    test_paths = set(intersect_paths(desc_all, test_perf))

    desc_train = {p: desc_all[p] for p in train_paths}
    desc_test = {p: desc_all[p] for p in test_paths}

    if not desc_train:
        print("Error: No overlapping instances between desc_csv and solver_train_csv.", file=sys.stderr)
        return 1
    if not desc_test:
        print("Error: No overlapping instances between desc_csv and solver_test_csv.", file=sys.stderr)
        return 1

    # Build dataset
    if args.task == "pairwise":
        train_data = build_dataset_pairwise(desc_train, solver_names, train_perf)
        out_dim = 2
    elif args.task == "single":
        train_data = build_dataset_single(desc_train, solver_names, train_perf)
        out_dim = len(solver_names)
    else:
        train_data = build_dataset_regression(
            desc_train, solver_names, train_perf,
            log1p=args.reg_log1p,
        )
        out_dim = len(solver_names)

    if not train_data:
        print(f"Error: No training examples created for task={args.task}.", file=sys.stderr)
        return 1

    print(f"Train instances (desc ∩ train_perf): {len(desc_train)}")
    print(f"Test  instances (desc ∩ test_perf):  {len(desc_test)}")
    print(f"Training examples: {len(train_data)}")

    # Build model (+ optional LoRA)
    model = build_model_with_head(
        base_model_name=args.model,
        out_dim=out_dim,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # DataLoader
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Train
    train_loop(
        model=model,
        loader=train_loader,
        task=args.task,
        lr=args.lr,
        epochs=args.epochs,
        device=args.device,
    )

    # Predict on test
    if args.task == "pairwise":
        preds = predict_pairwise_tournament(model, solver_names, desc_test, test_perf, device=args.device)
    elif args.task == "single":
        preds = predict_single(model, solver_names, desc_test, test_perf, device=args.device)
    else:
        preds = predict_regression(model, solver_names, desc_test, test_perf, device=args.device)

    # Write predictions
    out_path = Path(args.output_predictions)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "predicted_solver"])
        writer.writerows(preds)

    print(f"Saved predictions to {out_path}")

    # Save minimal metadata for reproducibility
    meta = {
        "task": args.task,
        "base_model": args.model,
        "use_lora": args.use_lora,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout} if args.use_lora else None,
        "solvers": solver_names,
        "train_instances": len(desc_train),
        "test_instances": len(desc_test),
        "training_examples": len(train_data),
        "reg_log1p": bool(args.reg_log1p),
        "timeout_fill": float(args.timeout_fill),
    }
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved run metadata to {meta_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
