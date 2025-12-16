import csv
from typing import Dict, List, Tuple

from .performance import MultiSolverDataset


def parse_performance_csv(csv_path: str, timeout: float) -> MultiSolverDataset:
    """
    Parse a performance CSV file and create a MultiSolverDataset.

    Args:
        csv_path: Path to the CSV file
        timeout: Timeout value in seconds (used for benchmarking)

    Returns:
        A MultiSolverDataset containing:
        - multi_perf_dict: path -> list of (is_solved, wc_time) tuples
        - solver_dict: id -> solver name, where id corresponds to the position in the performance list
        - timeout: timeout value in seconds
    """
    multi_perf_dict: Dict[str, List[Tuple[int, float]]] = {}
    solver_dict: Dict[int, str] = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        # Read header rows
        header_row0 = next(reader)  # path,OpenSMT,,SMTInterpol,,...
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

        # Process data rows
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

    return MultiSolverDataset(multi_perf_dict, solver_dict, timeout)
