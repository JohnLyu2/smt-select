import csv
import argparse
import logging
import multiprocessing
from pathlib import Path

from tqdm import tqdm

from .performance import SingleSolverDataset
from .performance import parse_performance_csv
from .pwc import PwcSelector
from .setfit_model import SetfitSelector

# Set by pool initializer in worker processes.
_worker_selector = None


def _init_selector(init_arg: tuple) -> None:
    """Pool initializer: load selector and store in global for _eval_worker."""
    global _worker_selector
    loader_fn, loader_args = init_arg
    _worker_selector = loader_fn(*loader_args)


def _eval_worker(instance_path: str) -> tuple[str, int]:
    """Worker: return (instance_path, selected_solver_id). Uses _worker_selector set by _init_selector."""
    return (instance_path, _worker_selector.algorithm_select(instance_path))


def _load_pwc_selector(model_path: str):
    """Top-level loader for PWC (picklable for multiprocessing)."""
    return PwcSelector.load(model_path)


def _load_setfit_selector(setfit_model: str, desc_json: str):
    """Top-level loader for SetFit (picklable for multiprocessing)."""
    return SetfitSelector(setfit_model, desc_json)


def as_evaluate_parallel(
    instance_paths: list[str],
    loader_fn,
    loader_args: tuple,
    multi_perf_data,
    n_workers: int,
    write_csv_path: str | None = None,
    show_progress: bool = True,
    result_timeout: int = 30,
    fallback_solver_id: int | None = None,
):
    """
    Evaluate algorithm selection in parallel. Each worker loads the selector via loader_fn(*loader_args).
    Returns SingleSolverDataset like as_evaluate.

    result_timeout: max seconds to wait for each worker result; prevents main process blocking
    forever on a stuck worker. On timeout/error the instance is scored with fallback_solver_id.
    fallback_solver_id: solver id to use when a worker times out or fails. If None, use the
    first solver id from multi_perf_data (min of solver id dict keys).
    """
    n_workers = min(n_workers, len(instance_paths))
    if n_workers <= 0:
        return SingleSolverDataset({}, "AS", multi_perf_data.get_timeout())
    if fallback_solver_id is None:
        solver_ids = multi_perf_data.get_solver_id_dict().keys()
        fallback_solver_id = min(solver_ids) if solver_ids else 0
    perf_dict: dict = {}
    path_to_selected: dict[str, int] = {}
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(
        n_workers,
        initializer=_init_selector,
        initargs=((loader_fn, loader_args),),
    ) as pool:
        async_results = [
            (instance_paths[i], pool.apply_async(_eval_worker, (instance_paths[i],)))
            for i in range(len(instance_paths))
        ]
        it = async_results
        if show_progress:
            it = tqdm(it, total=len(async_results), desc="Evaluating", unit="instance")
        for instance_path, ar in it:
            try:
                _, selected = ar.get(timeout=result_timeout)
            except (TimeoutError, multiprocessing.TimeoutError):
                logging.warning(
                    "Evaluation result timeout (%ds) for %s — worker stuck; using fallback solver %s",
                    result_timeout,
                    instance_path,
                    multi_perf_data.get_solver_name(fallback_solver_id),
                )
                selected = fallback_solver_id
            except Exception as e:
                logging.debug("Evaluation failed for %s: %s", instance_path, e)
                selected = fallback_solver_id
            path_to_selected[instance_path] = selected
            perf_dict[instance_path] = multi_perf_data.get_performance(
                instance_path, selected
            )
    if write_csv_path is not None:
        with Path(write_csv_path).open(mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["benchmark", "selected", "solved", "runtime"])
            for path in instance_paths:
                selected = path_to_selected[path]
                is_solved, runtime = perf_dict[path]
                csv_writer.writerow(
                    [
                        path,
                        multi_perf_data.get_solver_name(selected),
                        is_solved,
                        runtime,
                    ]
                )
    return SingleSolverDataset(
        perf_dict,
        "AS",
        multi_perf_data.get_timeout(),
    )


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
        "--jobs",
        type=int,
        default=1,
        help="Parallel workers for evaluation; 1 = sequential (default: 1)",
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

    # Load performance data
    logging.info("Loading performance data from %s", args.perf_csv)
    multi_perf_data = parse_performance_csv(args.perf_csv, args.timeout)
    logging.info(
        "Loaded %d instances with %d solvers",
        len(multi_perf_data),
        multi_perf_data.num_solvers(),
    )
    instance_paths = list(multi_perf_data.keys())

    # Evaluate
    logging.info("Starting evaluation...")
    if args.jobs > 1:
        if args.setfit_model is not None:
            loader_fn, loader_args = _load_setfit_selector, (
                args.setfit_model,
                args.desc_json,
            )
        else:
            loader_fn, loader_args = _load_pwc_selector, (args.pwc_model,)
        result_dataset = as_evaluate_parallel(
            instance_paths,
            loader_fn,
            loader_args,
            multi_perf_data,
            n_workers=args.jobs,
            write_csv_path=args.output_csv,
            show_progress=True,
        )
    else:
        if args.setfit_model is not None:
            if args.desc_json is None:
                raise ValueError(
                    "--desc-json is required when using --setfit-model."
                )
            logging.info("Using SetfitSelector with model %s", args.setfit_model)
            as_model = SetfitSelector(args.setfit_model, args.desc_json)
        else:
            logging.info("Loading PWC model from %s", args.pwc_model)
            as_model = PwcSelector.load(args.pwc_model)
            logging.info(
                "Loaded %s model with %d solvers",
                as_model.model_type,
                as_model.solver_size,
            )
        result_dataset = as_evaluate(
            as_model, multi_perf_data, args.output_csv, show_progress=True
        )
    metrics = compute_metrics(result_dataset, multi_perf_data)
    log_evaluation_summary(metrics, multi_perf_data)

    if args.output_csv:
        logging.info(f"Results written to {args.output_csv}")


if __name__ == "__main__":
    main()
