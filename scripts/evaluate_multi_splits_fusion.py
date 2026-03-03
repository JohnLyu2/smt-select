#!/usr/bin/env python3
"""
Evaluate Fusion-PWC (GIN + text embeddings, project, LayerNorm, projection, PWC heads) over multiple train/test splits.

Expects:
  - Splits: --splits-dir with seed subdirs (train.json, test.json); paths rebased with --benchmark-root.
  - GIN embeddings: --gin-features-base/{division}/seed{N}/features.csv (and extraction_times.csv for failed paths).
  - Text embeddings: --desc-features-dir/{division}/features.csv (e.g. desc/all-mpnet-base-v2/BV/features.csv).

For each seed: build emb_by_path (L2-normalize GIN), train Fusion-PWC, evaluate on train and test, write summary and per-seed CSVs.
Use --eval-only with --models-base to evaluate pre-trained fusion models without training.
"""

import argparse
import csv
import gc
import json
import logging
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.defaults import DEFAULT_BENCHMARK_ROOT
from src.evaluate import as_evaluate, compute_metrics, load_extraction_times_csv
from src.fusion_pwc import (
    build_emb_by_path,
    load_embedding_csv,
    train_fusion_pwc,
    FusionPWCSelector,
)
from src.performance import (
    filter_training_instances,
    parse_as_perf_csv,
    parse_performance_json,
    MultiSolverDataset,
)

DEFAULT_GIN_FEATURES_BASE = Path("data/features/gin_pwc")
DEFAULT_DESC_FEATURES_DIR = Path("data/features/desc/all-mpnet-base-v2")
DEFAULT_LITETEXT_DIR = Path("data/cp26/results/lite+text")
DEFAULT_SPLITS_BASE = Path("data/cp26/performance_splits/smtcomp24")

CSV_HEADER = ["benchmark", "selected", "solved", "runtime", "solver_runtime", "overhead", "feature_fail"]


def discover_seed_dirs(splits_dir: Path) -> list[tuple[int, Path]]:
    """Find seed subdirs under splits_dir that contain train.json and test.json."""
    out: list[tuple[int, Path]] = []
    pattern = re.compile(r"^seed(\d+)$")
    for sub in splits_dir.iterdir():
        if not sub.is_dir():
            continue
        m = pattern.match(sub.name)
        if not m:
            continue
        if (sub / "train.json").exists() and (sub / "test.json").exists():
            out.append((int(m.group(1)), sub))
    return sorted(out, key=lambda x: x[0])


def _rebase_perf_data(multi_perf_data: MultiSolverDataset, benchmark_root: Path) -> MultiSolverDataset:
    """Rebase instance paths with benchmark_root."""
    rebased = {str(benchmark_root / p): multi_perf_data[p] for p in multi_perf_data.keys()}
    return MultiSolverDataset(
        rebased,
        multi_perf_data.get_solver_id_dict(),
        multi_perf_data.get_timeout(),
    )


def load_failed_gin_paths(extraction_times_csv: Path) -> list[str]:
    """Load instance paths with failed=1 from extraction_times CSV (relative paths)."""
    failed = []
    with open(extraction_times_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        path_key = "path"
        for row in reader:
            if (row.get("failed") or "").strip() == "1":
                p = (row.get(path_key) or row.get("benchmark") or "").strip().replace("\\", "/")
                if p:
                    failed.append(p)
    return failed


def _load_litetext_lookup(path: Path) -> dict[str, dict]:
    """Load Lite+Text eval CSV; return benchmark -> row dict."""
    out: dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bench = (row.get("benchmark") or "").strip()
            if bench:
                out[bench] = row
    return out


def _load_eval_csv(path: Path) -> list[dict]:
    """Load eval CSV; return list of row dicts."""
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_eval_csv(path: Path, rows: list[dict]) -> None:
    """Write rows using CSV_HEADER order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for row in rows:
            writer.writerow([row.get(k, "") for k in CSV_HEADER])


def _merge_with_litetext(
    fusion_rows: list[dict],
    litetext_lookup: dict[str, dict],
    timeout: float,
) -> list[dict]:
    """
    Replace feature_fail rows with Lite+Text results.

    For feature_fail rows:
      overhead_out = fusion_overhead + lite_text_overhead
      runtime_out  = lite_text_solver_runtime + overhead_out  (capped at timeout)
    """
    out: list[dict] = []
    for row in fusion_rows:
        bench = (row.get("benchmark") or "").strip()
        try:
            ff = int(row.get("feature_fail", 0) or 0)
        except (ValueError, TypeError):
            ff = 0

        if ff == 0:
            out.append(row)
            continue

        if bench not in litetext_lookup:
            raise ValueError(
                f"Missing Lite+Text row for feature-fail benchmark: {bench!r}"
            )
        lt_row = litetext_lookup[bench]

        raw_fusion_overhead = row.get("overhead", "")
        try:
            fusion_overhead = float(raw_fusion_overhead) if raw_fusion_overhead not in ("", None) else 0.0
        except (TypeError, ValueError):
            fusion_overhead = 0.0

        try:
            lt_solver_runtime = float(lt_row.get("solver_runtime") or 0.0)
        except (TypeError, ValueError):
            lt_solver_runtime = 0.0
        try:
            lt_solved = int(lt_row.get("solved") or 0)
        except (TypeError, ValueError):
            lt_solved = 0
        raw_lt_overhead = lt_row.get("overhead", "")
        try:
            lt_overhead = float(raw_lt_overhead) if raw_lt_overhead not in ("", None) else 0.0
        except (TypeError, ValueError):
            lt_overhead = 0.0

        overhead_out = fusion_overhead + lt_overhead
        runtime_out = lt_solver_runtime + overhead_out

        if runtime_out > timeout:
            solved = 0
            runtime_out = timeout
            overhead_out = max(0.0, timeout - lt_solver_runtime)
        else:
            solved = lt_solved

        out.append({
            "benchmark": bench,
            "selected": lt_row.get("selected", ""),
            "solved": str(solved),
            "runtime": str(runtime_out),
            "solver_runtime": str(lt_solver_runtime),
            "overhead": f"{overhead_out:.6f}",
            "feature_fail": "1",
        })
    return out


def evaluate_multi_splits_fusion(
    splits_dir: Path,
    *,
    gin_features_base: Path | str | None = None,
    desc_features_dir: Path | str | None = None,
    lite_text_dir: Path | str | None = None,
    benchmark_root: Path | str | None = None,
    save_models: bool = False,
    output_dir: Path | None = None,
    models_base: Path | None = None,
    eval_only: bool = False,
    timeout: float = 1200.0,
    d_text_small: int = 64,
    hidden_fused: int = 128,
    num_epochs: int = 500,
    batch_size: int = 64,
    lr: float = 1e-3,
    dropout: float = 0.1,
    val_ratio: float = 0.15,
    patience: int = 50,
    val_split_seed: int = 42,
    min_epochs: int = 100,
    skip_easy_unsolvable: bool = False,
    skip_trivial_under: float = 24.0,
    seeds: list[int] | None = None,
) -> dict:
    """
    Run Fusion-PWC train/eval for each seed under splits_dir.
    When eval_only=True, load models from models_base/fusion_pwc/{division}/seed{N}.
    """
    splits_dir = Path(splits_dir).resolve()
    if not splits_dir.is_dir():
        raise ValueError(f"Splits directory does not exist: {splits_dir}")
    if output_dir is None:
        raise ValueError("output_dir is required (CSVs are needed for Lite+Text fallback merge)")
    if eval_only and models_base is None:
        raise ValueError("models_base is required when eval_only=True")

    root = Path(benchmark_root or DEFAULT_BENCHMARK_ROOT).resolve()
    if not root.is_dir():
        raise ValueError(f"Benchmark root is not a directory: {root}")

    gin_base = Path(gin_features_base or DEFAULT_GIN_FEATURES_BASE).resolve()
    desc_dir = Path(desc_features_dir or DEFAULT_DESC_FEATURES_DIR).resolve()
    lt_dir = Path(lite_text_dir or DEFAULT_LITETEXT_DIR).resolve()
    division = splits_dir.name

    lt_division_dir = lt_dir / division
    if not lt_division_dir.is_dir():
        raise FileNotFoundError(
            f"Lite+Text results dir not found for division {division}: {lt_division_dir}"
        )

    desc_division_dir = desc_dir / division
    desc_csv = desc_division_dir / "features.csv"
    if not desc_csv.is_file():
        raise ValueError(f"Description features CSV not found: {desc_csv}")
    desc_extraction_csv = desc_division_dir / "extraction_times.csv"
    if not desc_extraction_csv.is_file():
        raise FileNotFoundError(f"Description extraction times not found: {desc_extraction_csv}")
    # Map desc extraction times (logic-relative paths) to full paths under benchmark root
    desc_extraction_rel = load_extraction_times_csv(desc_extraction_csv)
    desc_overhead_by_full = {str(root / p): t for p, t in desc_extraction_rel.items()}

    seed_entries = discover_seed_dirs(splits_dir)
    if not seed_entries:
        raise ValueError(
            f"No seed dirs (seedN with train.json and test.json) found in {splits_dir}"
        )
    if seeds is not None:
        seed_set = set(seeds)
        seed_entries = [(s, d) for s, d in seed_entries if s in seed_set]
        if not seed_entries:
            raise ValueError(f"No matching seed dirs for seeds {sorted(seed_set)} in {splits_dir}")

    n_seeds = len(seed_entries)
    logging.info(
        "Starting Fusion-PWC evaluation over %d splits for division %s%s",
        n_seeds,
        division,
        " (eval-only)" if eval_only else "",
    )

    seed_results: list[dict] = []

    for seed_val, seed_dir in seed_entries:
        logging.info("\n%s", "=" * 60)
        logging.info("Seed %d (%s)", seed_val, seed_dir.name)
        logging.info("%s", "=" * 60)

        train_path = seed_dir / "train.json"
        test_path = seed_dir / "test.json"
        train_data = parse_performance_json(str(train_path), timeout)
        test_data = parse_performance_json(str(test_path), timeout)
        train_data = _rebase_perf_data(train_data, root)
        test_data = _rebase_perf_data(test_data, root)

        if skip_easy_unsolvable:
            paths_to_keep, filter_stats = filter_training_instances(
                train_data,
                skip_unsolvable=True,
                skip_trivial_under=skip_trivial_under,
            )
            train_data_for_training = MultiSolverDataset(
                {p: train_data[p] for p in paths_to_keep},
                train_data.get_solver_id_dict(),
                train_data.get_timeout(),
            )
            logging.info(
                "Training on %d instances (dropped %d unsolvable, %d trivial); train eval will use full %d",
                filter_stats["n_kept"],
                filter_stats["n_unsolvable"],
                filter_stats["n_trivial"],
                len(train_data),
            )
        else:
            train_data_for_training = train_data

        gin_csv = gin_base / division / f"seed{seed_val}" / "features.csv"
        extraction_times_csv = gin_base / division / f"seed{seed_val}" / "extraction_times.csv"
        if not gin_csv.is_file():
            raise FileNotFoundError(f"GIN embeddings not found: {gin_csv}")
        if not extraction_times_csv.is_file():
            raise FileNotFoundError(f"GIN extraction times not found: {extraction_times_csv}")

        gin_by_rel = load_embedding_csv(gin_csv)
        text_by_rel = load_embedding_csv(desc_csv)
        if text_by_rel:
            d_text = next(iter(text_by_rel.values())).shape[0]
        else:
            raise ValueError(f"No text embeddings loaded from {desc_csv}")
        paths_full = list(train_data.keys()) + list(test_data.keys())
        emb_by_path = build_emb_by_path(
            gin_by_rel,
            text_by_rel,
            root,
            paths_full,
            normalize_gin=True,
        )

        # GIN failures from extraction_times.csv (logic-relative paths), mapped to full paths.
        failed_rel = load_failed_gin_paths(extraction_times_csv)
        failed_full = [str(root / p) for p in failed_rel]

        # Sanity check: any instance without embeddings must be a known GIN failure.
        missing_emb_paths = {p for p in paths_full if p not in emb_by_path}
        unexpected_missing = sorted(p for p in missing_emb_paths if p not in failed_full)
        if unexpected_missing:
            raise ValueError(
                "Fusion-PWC: found instances without (GIN, text) embeddings "
                "that are not marked failed in GIN extraction_times.csv. "
                f"Examples: {unexpected_missing[:5]} (total {len(unexpected_missing)})."
            )

        # Build overhead map: per-instance overhead = GIN extraction + description extraction.
        gin_extraction_rel = load_extraction_times_csv(extraction_times_csv)
        gin_overhead_by_full = {str(root / p): t for p, t in gin_extraction_rel.items()}
        overhead_by_path = {
            p: gin_overhead_by_full.get(p, 0.0) + desc_overhead_by_full.get(p, 0.0)
            for p in paths_full
        }

        n_with_emb = len(emb_by_path)
        n_train = len(train_data)
        n_test = len(test_data)
        logging.info(
            "Instances with both embeddings: %d (train+test total %d)",
            n_with_emb,
            n_train + n_test,
        )

        if eval_only:
            model_save_dir = models_base / "fusion_pwc" / division / f"seed{seed_val}"
            if not model_save_dir.is_dir() or not (model_save_dir / "config.json").exists():
                raise FileNotFoundError(f"Eval-only: model dir not found: {model_save_dir}")
            logging.info("Loading saved model from %s", model_save_dir)
        else:
            if save_models and models_base is not None:
                model_save_dir = models_base / "fusion_pwc" / division / f"seed{seed_val}"
            elif save_models and output_dir:
                model_save_dir = output_dir / "models" / "fusion_pwc" / division / f"seed{seed_val}"
            else:
                model_save_dir = Path(tempfile.mkdtemp())
            model_save_dir.mkdir(parents=True, exist_ok=True)

            log_handler: logging.FileHandler | None = None
            if output_dir:
                train_log_dir = output_dir / "train_log"
                train_log_dir.mkdir(parents=True, exist_ok=True)
                log_file = train_log_dir / f"seed{seed_val}.log"
                log_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
                log_handler.setLevel(logging.DEBUG)
                log_handler.setFormatter(
                    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
                )
                logging.getLogger().addHandler(log_handler)

            try:
                train_paths_with_emb = [
                    p for p in train_data_for_training.keys() if p in emb_by_path
                ]
                train_fusion_pwc(
                    emb_by_path,
                    train_paths_with_emb,
                    train_data_for_training,
                    str(model_save_dir),
                    d_gin=64,
                    d_text=d_text,
                    d_text_small=d_text_small,
                    hidden_fused=hidden_fused,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    lr=lr,
                    dropout=dropout,
                    val_ratio=val_ratio,
                    patience=patience,
                    val_split_seed=val_split_seed,
                    min_epochs=min_epochs,
                    failed_paths=failed_full,
                )
            finally:
                if log_handler is not None:
                    logging.getLogger().removeHandler(log_handler)
                    log_handler.close()

        selector = FusionPWCSelector.load(
            str(model_save_dir),
            emb_by_path=emb_by_path,
            device="cpu",
        )

        seed_out_dir = output_dir / f"seed{seed_val}"
        seed_out_dir.mkdir(parents=True, exist_ok=True)
        test_output_csv = str(seed_out_dir / "test_eval.csv")

        as_evaluate(
            selector,
            test_data,
            write_csv_path=test_output_csv,
            show_progress=True,
            csv_benchmark_root=root,
            extra_overhead_by_path=overhead_by_path,
        )

        # Merge feature_fail rows with Lite+Text results
        lt_seed_dir = lt_division_dir / f"seed{seed_val}"
        if not lt_seed_dir.is_dir():
            raise FileNotFoundError(f"Lite+Text seed dir not found: {lt_seed_dir}")
        lt_test_csv = lt_seed_dir / "test_eval.csv"
        if not lt_test_csv.is_file():
            raise FileNotFoundError(f"Lite+Text test CSV not found: {lt_test_csv}")

        test_csv_path = Path(test_output_csv)
        fusion_test_rows = _load_eval_csv(test_csv_path)
        lt_test_lookup = _load_litetext_lookup(lt_test_csv)
        merged_test = _merge_with_litetext(fusion_test_rows, lt_test_lookup, timeout)
        _write_eval_csv(test_csv_path, merged_test)
        n_fb_test = sum(1 for r in merged_test if r.get("feature_fail") == "1")
        logging.info("  Merged test: %d/%d rows from Lite+Text fallback", n_fb_test, len(merged_test))

        # Compute metrics from the merged CSV
        test_result = parse_as_perf_csv(test_output_csv, timeout)
        test_metrics = compute_metrics(test_result, test_data)

        seed_results.append({
            "seed": seed_val,
            "train_size": n_train,
            "test_size": n_test,
            "test_metrics": test_metrics,
        })

        test_gap_pct = (test_metrics["gap_cls_par2"] * 100) if test_metrics.get("gap_cls_par2") is not None else 0.0
        logging.info(
            "  Test: solved %d/%d, gap closed (PAR-2): %.2f%%",
            test_metrics["solved"], n_test, test_gap_pct,
        )

        if not save_models and not eval_only and model_save_dir.exists():
            shutil.rmtree(model_save_dir, ignore_errors=True)

        del selector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_metrics_list = [r["test_metrics"] for r in seed_results]
    aggregated = {
        "test": {
            "gap_cls_solved_mean": float(np.mean([m["gap_cls_solved"] for m in test_metrics_list])),
            "gap_cls_solved_std": float(np.std([m["gap_cls_solved"] for m in test_metrics_list])),
            "gap_cls_par2_mean": float(np.mean([m["gap_cls_par2"] for m in test_metrics_list])),
            "gap_cls_par2_std": float(np.std([m["gap_cls_par2"] for m in test_metrics_list])),
        },
    }

    results = {
        "division": division,
        "n_seeds": n_seeds,
        "seed_values": [s for s, _ in seed_entries],
        "model_type": "fusion_pwc",
        "splits_dir": str(splits_dir),
        "benchmark_root": str(root),
        "seeds": seed_results,
        "aggregated": aggregated,
    }

    if output_dir:
        summary_path = Path(output_dir) / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        def to_python(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: to_python(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_python(x) for x in obj]
            return obj

        with open(summary_path, "w") as f:
            json.dump(to_python(results), f, indent=2)
        logging.info("Saved summary to %s", summary_path)

    agg = results["aggregated"]
    logging.info("\n%s", "=" * 60)
    logging.info("Multi-splits summary — %s (Fusion-PWC)", results["division"])
    logging.info("%s", "=" * 60)
    logging.info("Seeds: %s", results["seed_values"])
    t = agg["test"]
    avg_solved = float(np.mean([m["solved"] for m in test_metrics_list]))
    logging.info(
        "Test: avg solved %.1f, gap closed (solved) %.2f%% ± %.2f%%, gap closed (PAR-2) %.2f%% ± %.2f%%",
        avg_solved,
        t["gap_cls_solved_mean"] * 100, t["gap_cls_solved_std"] * 100,
        t["gap_cls_par2_mean"] * 100, t["gap_cls_par2_std"] * 100,
    )
    logging.info("%s", "=" * 60)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Fusion-PWC (GIN + text fusion + PWC) over multiple splits"
    )
    parser.add_argument("--logic", type=str, default=None, help="Division (e.g. BV). Sets splits and output paths.")
    parser.add_argument(
        "--splits-dir",
        type=str,
        default=None,
        help="Directory with seed subdirs (train.json, test.json). If omitted with --logic, auto-set. If both omitted, all logics are discovered.",
    )
    parser.add_argument("--benchmark-root", type=str, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument(
        "--gin-features-base",
        type=str,
        default=None,
        help=f"Base for GIN embeddings (default: {DEFAULT_GIN_FEATURES_BASE})",
    )
    parser.add_argument(
        "--desc-features-dir",
        type=str,
        default=None,
        help=f"Directory for description CSVs (default: {DEFAULT_DESC_FEATURES_DIR})",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--lite-text-dir", type=str, default=None,
        help=f"Lite+Text results root for fallback (default: {DEFAULT_LITETEXT_DIR})",
    )
    parser.add_argument("--eval-only", action="store_true", help="Load models from --models-base, skip training.")
    parser.add_argument(
        "--skip-easy-unsolvable",
        action="store_true",
        help="Exclude train instances where no solver solves or all solve in <= N seconds; saves train time",
    )
    parser.add_argument(
        "--skip-trivial-under",
        type=float,
        default=24.0,
        help="When --skip-easy-unsolvable: exclude train instances where every solver solved with runtime <= this (default 24)",
    )
    parser.add_argument("--models-base", type=str, default=None, help="Base for saved models (required if --eval-only).")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--d-text-small", type=int, default=64, help="Text projection output dim")
    parser.add_argument("--hidden-fused", type=int, default=64, help="Fusion projection output dim")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--val-split-seed", type=int, default=42)
    parser.add_argument("--min-epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="*", default=None, metavar="N")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    if args.logic:
        args.splits_dir = str(DEFAULT_SPLITS_BASE / args.logic)
        args.output_dir = str(Path("data/cp26/results/fusion_pwc") / args.logic)
        args.save_models = True
        if args.models_base is None:
            args.models_base = Path("models")
    if args.eval_only and args.models_base is None:
        parser.error("--models-base is required when --eval-only (or use --logic)")
    if args.eval_only and args.models_base:
        args.models_base = Path(args.models_base)
        if not args.models_base.is_dir():
            parser.error(f"--models-base must be an existing directory: {args.models_base}")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.splits_dir:
        output_dir = Path(args.output_dir) if args.output_dir else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        evaluate_multi_splits_fusion(
            Path(args.splits_dir),
            gin_features_base=args.gin_features_base,
            desc_features_dir=args.desc_features_dir,
            lite_text_dir=args.lite_text_dir,
            benchmark_root=args.benchmark_root,
            save_models=args.save_models,
            output_dir=output_dir,
            models_base=Path(args.models_base) if args.models_base else None,
            eval_only=args.eval_only,
            timeout=args.timeout,
            d_text_small=args.d_text_small,
            hidden_fused=args.hidden_fused,
            num_epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            dropout=0.1,
            val_ratio=args.val_ratio,
            patience=args.patience,
            val_split_seed=args.val_split_seed,
            min_epochs=args.min_epochs,
            skip_easy_unsolvable=args.skip_easy_unsolvable,
            skip_trivial_under=args.skip_trivial_under,
            seeds=args.seeds,
        )
        return

    if not args.output_dir:
        parser.error("--output-dir is required when running all logics (no --logic or --splits-dir)")

    gin_base = Path(args.gin_features_base or DEFAULT_GIN_FEATURES_BASE)
    desc_dir = Path(args.desc_features_dir or DEFAULT_DESC_FEATURES_DIR)
    lt_dir = Path(args.lite_text_dir or DEFAULT_LITETEXT_DIR)
    splits_base = DEFAULT_SPLITS_BASE
    base_output_dir = Path(args.output_dir)

    candidates = sorted(
        sub.name for sub in desc_dir.iterdir()
        if sub.is_dir() and (sub / "features.csv").is_file()
    )
    logics = [
        name for name in candidates
        if (splits_base / name).is_dir()
        and (lt_dir / name).is_dir()
        and (gin_base / name).is_dir()
    ]
    if not logics:
        parser.error(
            f"No logics found with desc features in {desc_dir}, "
            f"splits in {splits_base}, lite+text in {lt_dir}, and GIN in {gin_base}"
        )
    logging.info("Running all %d logics: %s", len(logics), logics)

    for logic in logics:
        logging.info("\n%s Running %s %s", "=" * 20, logic, "=" * 20)
        output_dir = (base_output_dir / logic).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluate_multi_splits_fusion(
            splits_base / logic,
            gin_features_base=args.gin_features_base,
            desc_features_dir=args.desc_features_dir,
            lite_text_dir=args.lite_text_dir,
            benchmark_root=args.benchmark_root,
            save_models=args.save_models,
            output_dir=output_dir,
            models_base=Path(args.models_base) if args.models_base else None,
            eval_only=args.eval_only,
            timeout=args.timeout,
            d_text_small=args.d_text_small,
            hidden_fused=args.hidden_fused,
            num_epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            dropout=0.1,
            val_ratio=args.val_ratio,
            patience=args.patience,
            val_split_seed=args.val_split_seed,
            min_epochs=args.min_epochs,
            skip_easy_unsolvable=args.skip_easy_unsolvable,
            skip_trivial_under=args.skip_trivial_under,
            seeds=args.seeds,
        )


if __name__ == "__main__":
    main()
