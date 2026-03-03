#!/usr/bin/env python3
"""
Train SetFit (MPNet) on description embeddings for all seeds of a logic.

For each seed directory under data/cp26/performance_splits/smtcomp24/<LOGIC>/,
trains a SetFit model using benchmark descriptions and saves it.

Usage:
  source .venv/bin/activate
  python scripts/train_setfit_splits.py --logic ABV
  python scripts/train_setfit_splits.py --logic BV --num-epochs 2
  python scripts/train_setfit_splits.py                # all logics
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.setfit_model import create_setfit_data, train_setfit_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_BASE = PROJECT_ROOT / "data" / "cp26" / "performance_splits" / "smtcomp24"
DESC_BASE = PROJECT_ROOT / "data" / "meta_info_24" / "descriptions"
MODEL_BASE = PROJECT_ROOT / "models" / "setfit_mpnet"


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_seed_dirs(splits_dir: Path) -> list[tuple[int, Path]]:
    pattern = re.compile(r"^seed(\d+)$")
    out: list[tuple[int, Path]] = []
    for sub in splits_dir.iterdir():
        if not sub.is_dir():
            continue
        m = pattern.match(sub.name)
        if not m:
            continue
        if (sub / "train.json").exists() and (sub / "test.json").exists():
            out.append((int(m.group(1)), sub))
    return sorted(out, key=lambda x: x[0])


def discover_logics() -> list[str]:
    """Find logics that have both splits and a description JSON."""
    if not SPLITS_BASE.is_dir():
        return []
    return sorted(
        d.name
        for d in SPLITS_BASE.iterdir()
        if d.is_dir()
        and (DESC_BASE / f"{d.name}.json").is_file()
        and discover_seed_dirs(d)
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Train SetFit on descriptions for all seeds of a logic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--logic", default=None, help="Division name (e.g. ABV, BV, QF_LIA). Omit to run all logics.")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Base embedding model (default: sentence-transformers/all-mpnet-base-v2).",
    )
    parser.add_argument("--num-epochs", type=int, default=1, help="Training epochs (default: 1).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--timeout", type=float, default=1200.0, help="Timeout in seconds (default: 1200).")
    parser.add_argument("--no-amp", action="store_true", help="Disable Automatic Mixed Precision (enabled by default).")
    parser.add_argument(
        "--sampling-strategy",
        default="undersampling",
        choices=["oversampling", "undersampling", "unique"],
        help="Contrastive pair sampling strategy (default: undersampling).",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=20,
        help="Number of pair-generation iterations; set -1 to use --sampling-strategy instead (default: 20).",
    )
    args = parser.parse_args()

    if args.logic is not None:
        logics = [args.logic]
        splits_dir = SPLITS_BASE / args.logic
        if not splits_dir.is_dir():
            raise FileNotFoundError(f"Splits directory not found: {splits_dir}")
    else:
        logics = discover_logics()
        if not logics:
            raise ValueError(
                f"No logics found with both splits in {SPLITS_BASE} and descriptions in {DESC_BASE}"
            )
        logging.info("Discovered %d logics: %s", len(logics), logics)

    failed: list[str] = []
    for logic in logics:
        desc_json = DESC_BASE / f"{logic}.json"
        if not desc_json.is_file():
            raise FileNotFoundError(f"Description JSON not found: {desc_json}")

        seed_entries = discover_seed_dirs(SPLITS_BASE / logic)
        if not seed_entries:
            raise ValueError(f"No seed dirs (seedN with train.json/test.json) in {SPLITS_BASE / logic}")

        logging.info("Logic: %s | Seeds: %s | Model: %s", logic, [s for s, _ in seed_entries], args.model_name)

        try:
            _train_logic(logic, seed_entries, desc_json, args)
        except Exception:
            logging.exception("Failed training for logic %s", logic)
            failed.append(logic)

    if failed:
        logging.error("Failed logics: %s", failed)
        return 1

    logging.info("\nAll done. Models saved under %s", MODEL_BASE)
    return 0


def _train_logic(
    logic: str,
    seed_entries: list[tuple[int, Path]],
    desc_json: Path,
    args: argparse.Namespace,
) -> None:
    for seed_val, seed_dir in seed_entries:
        logging.info("\n%s", "=" * 60)
        logging.info("Logic %s — Seed %d", logic, seed_val)
        logging.info("=" * 60)

        set_seeds(seed_val)

        train_perf = str(seed_dir / "train.json")
        model_dir = MODEL_BASE / logic / f"seed{seed_val}"
        model_dir.mkdir(parents=True, exist_ok=True)

        train_data = create_setfit_data(
            train_perf,
            str(desc_json),
            args.timeout,
            include_all_solved=False,
            multi_label=False,
        )
        solver_id_dict = train_data["solver_id_dict"]
        logging.info(
            "Train samples: %d | Solvers: %d | Unique labels: %d",
            len(train_data["texts"]),
            len(solver_id_dict),
            len(set(train_data["labels"])),
        )

        train_setfit_model(
            train_data=train_data,
            test_data=None,
            model_name=args.model_name,
            output_dir=str(model_dir),
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            multi_label=False,
            use_amp=not args.no_amp,
            sampling_strategy=args.sampling_strategy,
            num_iterations=args.num_iterations if args.num_iterations >= 0 else None,
        )

        solver2id = {name: idx for idx, name in solver_id_dict.items()}
        with open(model_dir / "solver2id.json", "w", encoding="utf-8") as f:
            json.dump(solver2id, f, indent=2)

        train_config = {
            "logic": logic,
            "seed": seed_val,
            "desc_json": str(desc_json),
            "train_perf_json": train_perf,
            "model_name": args.model_name,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "timeout": args.timeout,
            "use_amp": not args.no_amp,
            "sampling_strategy": args.sampling_strategy,
            "num_iterations": args.num_iterations if args.num_iterations >= 0 else None,
        }
        with open(model_dir / "train_config.json", "w", encoding="utf-8") as f:
            json.dump(train_config, f, indent=2)

        logging.info("Saved model to %s", model_dir)

    logging.info("Logic %s complete. Models under %s", logic, MODEL_BASE / logic)


if __name__ == "__main__":
    sys.exit(main())
