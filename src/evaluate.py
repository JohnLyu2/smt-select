import csv
import argparse
import logging
from pathlib import Path

from .performance import SingleSolverDataset
from .performance import parse_performance_csv
from .pwc import PwcSelector
from .setfit_model import SetfitSelector


def as_evaluate(as_model, multi_perf_data, write_csv_path=None, show_progress=True):
    """
    Evaluate the performance of the algorithm selection model based on the provided multi_perf_data;
    If write_csv_path is not None, write the result to the csv file.
    If show_progress is True, show a tqdm progress bar over instances.
    """
    if write_csv_path is not None:
        with Path(write_csv_path).open(mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(
                [
                    "benchmark",
                    "selected",
                    "solved",
                    "runtime",
                ]
            )
    perf_dict = {}
    instance_paths = list(multi_perf_data.keys())
    if show_progress:
        from tqdm import tqdm
        instance_paths = tqdm(instance_paths, desc="Evaluating", unit="instance")
    for instance_path in instance_paths:
        # before = time.perf_counter()
        selected = as_model.algorithm_select(instance_path)
        # after = time.perf_counter()
        # inference_time = after - before
        selected_perf = multi_perf_data.get_performance(instance_path, selected)
        perf_dict[instance_path] = selected_perf
        if write_csv_path is not None:
            with Path(write_csv_path).open(mode="a", newline="") as csv_file:
                csv_writer = csv.writer(csv_file)
                is_solved, time = selected_perf
                selected_solver = multi_perf_data.get_solver_name(selected)
                csv_writer.writerow(
                    [
                        instance_path,
                        selected_solver,
                        is_solved,
                        time,
                    ]
                )
        logging.debug(f"Evaluated {instance_path}: selected solver {selected}")
    return SingleSolverDataset(
        perf_dict,
        "AS",  # TODO: add AS name from as_model
        multi_perf_data.get_timeout(),
    )


def compute_metrics(result_dataset, multi_perf_data) -> dict:
    """Compute AS vs SBS/VBS metrics. Returns dict with solved, avg_par2, sbs_*, vbs_*, gap_cls_*, total_count."""
    total_count = len(result_dataset)
    solved_count = result_dataset.get_solved_count()
    total_par2 = sum(result_dataset.get_par2(p) for p in result_dataset.keys())
    avg_par2 = total_par2 / total_count if total_count > 0 else 0.0

    sbs_dataset = multi_perf_data.get_best_solver_dataset()
    sbs_solved = sbs_dataset.get_solved_count()
    total_par2_sbs = sum(sbs_dataset.get_par2(p) for p in sbs_dataset.keys())
    avg_par2_sbs = total_par2_sbs / total_count if total_count > 0 else 0.0

    vbs_dataset = multi_perf_data.get_virtual_best_solver_dataset()
    vbs_solved = vbs_dataset.get_solved_count()
    total_par2_vbs = sum(vbs_dataset.get_par2(p) for p in vbs_dataset.keys())
    avg_par2_vbs = total_par2_vbs / total_count if total_count > 0 else 0.0

    solved_denom = vbs_solved - sbs_solved
    par2_denom = avg_par2_vbs - avg_par2_sbs
    gap_cls_solved = (
        (solved_count - sbs_solved) / solved_denom
        if solved_denom != 0
        else (1.0 if solved_count == vbs_solved else 0.0)
    )
    gap_cls_par2 = (
        (avg_par2 - avg_par2_sbs) / par2_denom
        if par2_denom != 0
        else (1.0 if avg_par2 == avg_par2_vbs else 0.0)
    )
    return {
        "total_count": total_count,
        "solved": solved_count,
        "avg_par2": avg_par2,
        "sbs_solved": sbs_solved,
        "sbs_avg_par2": avg_par2_sbs,
        "vbs_solved": vbs_solved,
        "vbs_avg_par2": avg_par2_vbs,
        "gap_cls_solved": gap_cls_solved,
        "gap_cls_par2": gap_cls_par2,
    }


def format_evaluation_short(metrics: dict) -> str:
    """Two-line summary: instances/solved/solve rate, then PAR2 and gap_cls_par2."""
    n = metrics["total_count"]
    sr = (metrics["solved"] / n * 100) if n else 0.0
    lines = [
        f"Instances: {n}, Solved: {metrics['solved']}, Solve rate: {sr:.2f}%",
        f"Avg PAR2: {metrics['avg_par2']:.2f}, gap_cls_par2: {metrics['gap_cls_par2']:.4f}",
    ]
    return "\n".join(lines)


def log_evaluation_summary(metrics: dict, multi_perf_data) -> None:
    """Log full evaluation block (AS, SBS, VBS, gap closed)."""
    total_count = metrics["total_count"]
    solve_rate = (metrics["solved"] / total_count * 100) if total_count > 0 else 0.0
    sbs_solve_rate = (metrics["sbs_solved"] / total_count * 100) if total_count > 0 else 0.0
    vbs_solve_rate = (metrics["vbs_solved"] / total_count * 100) if total_count > 0 else 0.0
    sbs_dataset = multi_perf_data.get_best_solver_dataset()
    vbs_dataset = multi_perf_data.get_virtual_best_solver_dataset()

    logging.info("=" * 60)
    logging.info("Evaluation Results:")
    logging.info("  Total instances: %d", total_count)
    logging.info("  Solved: %d", metrics["solved"])
    logging.info("  Solve rate: %.2f%%", solve_rate)
    logging.info("  Average PAR-2: %.2f", metrics["avg_par2"])
    logging.info("  Gap closed (solved): %.4f", metrics["gap_cls_solved"])
    logging.info("  Gap closed (PAR-2): %.4f", metrics["gap_cls_par2"])
    logging.info("")
    logging.info("SBS:")
    logging.info("  Solver: %s", sbs_dataset.get_solver_name())
    logging.info("  Solved: %d", metrics["sbs_solved"])
    logging.info("  Solve rate: %.2f%%", sbs_solve_rate)
    logging.info("  Average PAR-2: %.2f", metrics["sbs_avg_par2"])
    logging.info("")
    logging.info("VBS:")
    logging.info("  Solver: %s", vbs_dataset.get_solver_name())
    logging.info("  Solved: %d", metrics["vbs_solved"])
    logging.info("  Solve rate: %.2f%%", vbs_solve_rate)
    logging.info("  Average PAR-2: %.2f", metrics["vbs_avg_par2"])
    logging.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate algorithm selection model performance"
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--pwc-model",
        type=str,
        help="Path to the trained PWC model file",
    )
    model_group.add_argument(
        "--setfit-model",
        type=str,
        help="SetFit model name",
    )
    parser.add_argument(
        "--desc-json",
        type=str,
        default=None,
        help="Path to descriptions JSON (required for SetfitSelector)",
    )
    parser.add_argument(
        "--perf-csv",
        type=str,
        required=True,
        help="Path to the performance CSV file",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to write evaluation results CSV",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1200.0,
        help="Timeout in seconds for PAR-2 (default: 1200.0)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if args.setfit_model is not None:
        if args.desc_json is None:
            raise ValueError("--desc-json is required when using --setfit-model.")
        logging.info("Using SetfitSelector with model %s", args.setfit_model)
        as_model = SetfitSelector(args.setfit_model, args.desc_json)
    else:
        # Load model
        logging.info("Loading PWC model from %s", args.pwc_model)
        as_model = PwcSelector.load(args.pwc_model)
        logging.info(
            f"Loaded {as_model.model_type} model with {as_model.solver_size} solvers"
        )

    # Load performance data
    logging.info(f"Loading performance data from {args.perf_csv}")
    multi_perf_data = parse_performance_csv(args.perf_csv, args.timeout)
    logging.info(
        f"Loaded {len(multi_perf_data)} instances with {multi_perf_data.num_solvers()} solvers"
    )

    # Evaluate
    logging.info("Starting evaluation...")
    result_dataset = as_evaluate(as_model, multi_perf_data, args.output_csv)
    metrics = compute_metrics(result_dataset, multi_perf_data)
    log_evaluation_summary(metrics, multi_perf_data)

    if args.output_csv:
        logging.info(f"Results written to {args.output_csv}")


if __name__ == "__main__":
    main()
