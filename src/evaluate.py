import csv
import argparse
import logging
from pathlib import Path

from .performance import SingleSolverDataset
from .parser import parse_performance_csv
from .pwc import PwcModel, PairwiseSVM


def as_evaluate(as_model, multi_perf_data, write_csv_path=None):
    """
    Evaluate the performance of the algorithm selection model based on the provided multi_perf_data;
    If write_csv_path is not None, write the result to the csv file
    """
    if write_csv_path is not None:
        with Path(write_csv_path).open(mode="w", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(
                [
                    "benchmark",
                    "selected",
                    "solved",
                    "time",
                ]
            )
    perf_dict = {}
    for instance_path in multi_perf_data.keys():
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate algorithm selection model performance"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model file",
    )
    parser.add_argument(
        "--perf-csv",
        type=str,
        required=True,
        help="Path to the performance CSV file",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1200.0,
        help="Timeout value in seconds (default: 1200.0)",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to write evaluation results CSV",
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

    # Load model
    logging.info(f"Loading model from {args.model}")
    as_model = PwcModel.load(args.model)
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

    # Print statistics
    solved_count = result_dataset.get_solved_count()
    total_count = len(result_dataset)
    solve_rate = (solved_count / total_count * 100) if total_count > 0 else 0.0

    logging.info("=" * 60)
    logging.info("Evaluation Results:")
    logging.info(f"  Total instances: {total_count}")
    logging.info(f"  Solved: {solved_count}")
    logging.info(f"  Solve rate: {solve_rate:.2f}%")
    logging.info("=" * 60)

    if args.output_csv:
        logging.info(f"Results written to {args.output_csv}")


if __name__ == "__main__":
    main()
