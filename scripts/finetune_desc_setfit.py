#!/usr/bin/env python3
"""
Finetune SetFit on natural-language SMT benchmark descriptions for solver selection.

Builds training data from description JSON and performance JSON (e.g. under
data/cp26/raw_data/smtcomp24_performance), trains a SetFit model, and saves
the model plus solver2id and train_config for evaluation.

Example:
  source .venv/bin/activate
  python scripts/finetune_desc_setfit.py \\
    --desc-json data/meta_info_24/BV.json \\
    --train-perf-json data/cp26/raw_data/smtcomp24_performance/BV.json \\
    --model-dir data/models/setfit_desc/BV

After training, evaluate with:
  python -m src.evaluate --setfit-model <model_dir> --desc-json <desc_json> \\
    --perf-csv <test_perf_csv> [--output-csv out.csv]
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.setfit_model import create_setfit_data, train_setfit_model


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Finetune SetFit on SMT descriptions for solver selection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--desc-json",
        required=True,
        help="Path to meta-info JSON with benchmark descriptions (e.g. data/meta_info_24/LOGIC.json).",
    )
    parser.add_argument(
        "--train-perf-json",
        required=True,
        help="Path to training performance JSON (e.g. data/cp26/raw_data/smtcomp24_performance/LOGIC.json).",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory to save the trained SetFit model, solver2id.json, and train_config.json.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Base sentence-transformers model (default: all-MiniLM-L6-v2).",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of SetFit training epochs (default: 1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="SetFit training batch size (default: 16).",
    )
    parser.add_argument(
        "--multi-label",
        action="store_true",
        help="Use multi-label training (all solvers that solve an instance).",
    )
    parser.add_argument(
        "--include-all-solved",
        action="store_true",
        help="Include instances solved by all solvers in training data.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1200.0,
        help="Timeout in seconds used in performance data (default: 1200.0).",
    )
    parser.add_argument(
        "--train-data-json",
        default=None,
        help="Optional path to write training data as JSON (paths/texts/labels only).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    set_seeds(args.seed)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Train data
    train_data = create_setfit_data(
        args.train_perf_json,
        args.desc_json,
        args.timeout,
        include_all_solved=args.include_all_solved,
        multi_label=args.multi_label,
    )
    solver_id_dict = train_data["solver_id_dict"]
    n_train = len(train_data["texts"])
    logging.info("Train samples: %s | Solvers: %s", n_train, len(solver_id_dict))
    if args.multi_label:
        logging.info("Multi-label: label vector length %s", len(solver_id_dict))
    else:
        logging.info("Unique labels: %s", len(set(train_data["labels"])))

    if args.train_data_json:
        out_data = {
            "texts": train_data["texts"],
            "labels": train_data["labels"],
            "paths": train_data["paths"],
            "solver_id_dict": {str(i): name for i, name in solver_id_dict.items()},
        }
        Path(args.train_data_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.train_data_json, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)
        logging.info("Wrote training data to %s", args.train_data_json)

    train_setfit_model(
        train_data=train_data,
        test_data=None,
        model_name=args.model_name,
        output_dir=str(model_dir),
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        multi_label=args.multi_label,
    )

    # Save solver name -> id for SetfitSelector at eval time
    solver2id = {name: idx for idx, name in solver_id_dict.items()}
    with open(model_dir / "solver2id.json", "w", encoding="utf-8") as f:
        json.dump(solver2id, f, indent=2)
    logging.info("Wrote %s", model_dir / "solver2id.json")

    # Save training config for reproducibility
    train_config = {
        "desc_json": str(Path(args.desc_json).resolve()),
        "train_perf_json": str(Path(args.train_perf_json).resolve()),
        "model_name": args.model_name,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "multi_label": args.multi_label,
        "include_all_solved": args.include_all_solved,
        "timeout": args.timeout,
        "seed": args.seed,
    }
    with open(model_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(train_config, f, indent=2)
    logging.info("Wrote %s", model_dir / "train_config.json")

    logging.info("Training complete. To evaluate:")
    logging.info(
        "  python -m src.evaluate --setfit-model %s --desc-json %s --perf-csv <TEST_CSV> [--output-csv out.csv]",
        model_dir.resolve(),
        Path(args.desc_json).resolve(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
