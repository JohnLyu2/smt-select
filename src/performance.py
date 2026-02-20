"""Data structures for performance and solver data."""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

PERF_DIFF_THRESHOLD = 1e-1  # Threshold for considering performance differences


class MultiSolverDataset:
    """A data structure for performance data across multiple solvers."""

    def __init__(
        self,
        perf_dict: Dict[str, List[Tuple[int, float]]],
        solver_id_dict: Dict[int, str],
        timeout: float,
    ):
        """
        Initialize performance dictionary.

        Args:
            perf_dict: Dictionary mapping path to list of (is_solved, wc_time) tuples
            solver_id_dict: Dictionary mapping solver ID to solver name
            timeout: Timeout in seconds
        """
        self._dict = perf_dict
        self._solver_id_dict = solver_id_dict
        self._timeout = timeout

    def __getitem__(self, smt_path: str) -> List[Tuple[int, float]]:
        """Get performance list for a SMT file path."""
        return self._dict[smt_path]

    def __len__(self) -> int:
        """Get number of instances."""
        return len(self._dict)

    def __iter__(self):
        """Iterate over SMT file paths."""
        return iter(self._dict)

    def items(self):
        """Iterate over (path, performance_list) pairs."""
        return self._dict.items()

    def keys(self):
        """Get all instance paths."""
        return self._dict.keys()

    def get_performance(
        self, smt_path: str, solver_id: int
    ) -> Optional[Tuple[int, float]]:
        """
        Get performance for a specific instance and solver.

        Args:
            smt_path: SMT file path
            solver_id: Solver ID

        Returns:
            Tuple of (is_solved, wc_time) or None if not found
        """
        if smt_path not in self._dict:
            return None
        perf_list = self._dict[smt_path]
        if solver_id >= len(perf_list):
            return None
        return perf_list[solver_id]

    def get_par2(self, smt_path: str, solver_id: int) -> Optional[float]:
        """
        Get PAR-2 score for a specific instance and solver.

        PAR-2 is the solving time if solved, otherwise twice the timeout.

        Args:
            smt_path: SMT file path
            solver_id: Solver ID

        Returns:
            PAR-2 score (solving time if solved, else 2 * timeout) or None if not found
        """
        perf = self.get_performance(smt_path, solver_id)
        if perf is None:
            return None
        is_solved, wc_time = perf
        if is_solved == 1:
            return wc_time
        else:
            return 2.0 * self._timeout

    def get_solved_count(self, solver_id: int) -> int:
        """Get number of instances solved by a specific solver."""
        count = 0
        for perf_list in self._dict.values():
            if solver_id < len(perf_list) and perf_list[solver_id][0] == 1:
                count += 1
        return count

    def is_none_solved(self, smt_path: str) -> bool:
        """Return True if no solver solved the instance."""
        perf_list = self._dict.get(smt_path)
        if perf_list is None:
            return False
        return all(is_solved == 0 for is_solved, _ in perf_list)

    def is_all_solved(self, smt_path: str) -> bool:
        """Return True if all solvers solved the instance."""
        perf_list = self._dict.get(smt_path)
        if perf_list is None:
            return False
        return all(is_solved == 1 for is_solved, _ in perf_list)

    def is_trivial(self, smt_path: str, max_runtime: float) -> bool:
        """
        Return True iff trivial: every solver has a result, every solver
        solved, and every solver's runtime <= max_runtime.
        """
        K = self.num_solvers()
        perf_list = self._dict.get(smt_path)
        if perf_list is None or len(perf_list) != K:
            return False
        for is_solved, wc_time in perf_list:
            if is_solved != 1 or wc_time > max_runtime:
                return False
        return True

    def get_solver_name(self, solver_id: int) -> Optional[str]:
        """Get solver name by ID."""
        return self._solver_id_dict.get(solver_id)

    def get_solver_id_dict(self) -> Dict[int, str]:
        """Get the solver id to name dictionary."""
        return self._solver_id_dict.copy()

    def num_solvers(self) -> int:
        """Get number of solvers."""
        return len(self._solver_id_dict)

    def get_timeout(self) -> float:
        """Get the timeout value in seconds."""
        return self._timeout

    def to_dict(self) -> Dict[str, List[Tuple[int, float]]]:
        """Convert to plain dictionary."""
        return self._dict.copy()

    def get_solver_dataset(self, solver_id: int) -> "SingleSolverDataset":
        """
        Get a SingleSolverDataset for a specific solver.

        Args:
            solver_id: Solver ID

        Returns:
            SingleSolverDataset containing performance data for the specified solver
        """
        solver_perf_dict: Dict[str, Tuple[int, float]] = {}
        for path, perf_list in self._dict.items():
            if solver_id < len(perf_list):
                solver_perf_dict[path] = perf_list[solver_id]
        solver_name = self.get_solver_name(solver_id)
        return SingleSolverDataset(solver_perf_dict, solver_name, self._timeout)

    def get_best_solver_dataset(self) -> "SingleSolverDataset":
        """
        Get the SingleSolverDataset for the best performing solver.

        The best solver is determined by the lowest average PAR-2 score.

        Returns:
            SingleSolverDataset containing performance data for the best solver

        Raises:
            ValueError: If there are no solvers or no valid performance data
        """
        if self.num_solvers() == 0:
            raise ValueError("Cannot find best solver: no solvers in dataset")

        best_solver_id = None
        best_avg_par2 = float("inf")

        for solver_id in range(self.num_solvers()):
            total_par2 = 0.0
            count = 0
            for path in self.keys():
                par2 = self.get_par2(path, solver_id)
                if par2 is not None:
                    total_par2 += par2
                    count += 1

            if count > 0:
                avg_par2 = total_par2 / count
                if avg_par2 < best_avg_par2:
                    best_avg_par2 = avg_par2
                    best_solver_id = solver_id

        if best_solver_id is None:
            raise ValueError("Cannot find best solver: no valid performance data")

        return self.get_solver_dataset(best_solver_id)

    def get_virtual_best_solver_dataset(self) -> "SingleSolverDataset":
        """
        Get a SingleSolverDataset representing the virtual best solver.

        The virtual best solver is a theoretical solver that, for each instance,
        always selects the solver with the best (lowest) PAR-2 score for that instance.
        This represents an upper bound on achievable performance.

        Returns:
            SingleSolverDataset containing performance data for the virtual best solver

        Raises:
            ValueError: If there are no solvers or no valid performance data
        """
        if self.num_solvers() == 0:
            raise ValueError("Cannot create virtual best solver: no solvers in dataset")

        virtual_best_perf_dict: Dict[str, Tuple[int, float]] = {}

        for path in self.keys():
            best_perf = None
            best_par2 = float("inf")

            # Find the solver with the best PAR-2 score for this instance
            for solver_id in range(self.num_solvers()):
                par2 = self.get_par2(path, solver_id)
                if par2 is not None and par2 < best_par2:
                    best_par2 = par2
                    best_perf = self.get_performance(path, solver_id)

            if best_perf is not None:
                virtual_best_perf_dict[path] = best_perf

        if len(virtual_best_perf_dict) == 0:
            raise ValueError(
                "Cannot create virtual best solver: no valid performance data"
            )

        return SingleSolverDataset(virtual_best_perf_dict, "VirtualBest", self._timeout)

    def get_solvers_solving_instance(self, smt_path: str) -> list[str]:
        """
        Get all solver names that solved a specific instance.

        Args:
            smt_path: Instance path

        Returns:
            List of solver names that solved the instance
        """
        perf_list = self._dict.get(smt_path, [])
        return [
            self.get_solver_name(i)
            for i, (is_solved, _) in enumerate(perf_list)
            if is_solved == 1
        ]

    def get_best_solver_for_instance(self, smt_path: str) -> Optional[str]:
        """
        Get the solver name with the best PAR-2 score for a specific instance.

        Args:
            smt_path: Instance path

        Returns:
            Solver name with the lowest PAR-2 for the instance, or None if unavailable
        """
        if smt_path not in self._dict:
            return None

        best_solver_id: int | None = None
        best_par2 = float("inf")

        for solver_id in range(self.num_solvers()):
            par2 = self.get_par2(smt_path, solver_id)
            if par2 is None:
                continue
            if par2 < best_par2:
                best_par2 = par2
                best_solver_id = solver_id

        if best_solver_id is None:
            return None

        return self.get_solver_name(best_solver_id)

    def check_dominance(self, threshold: float | None = None) -> list[str]:
        """
        Find solvers that are dominated by some other solver.

        Solver A dominates solver B if A's PAR-2 score is no worse than B's (within
        threshold) on every instance (where both have data), and strictly better on
        at least one instance. "No worse" means par2(A) <= par2(B) + threshold.

        Args:
            threshold: Performance threshold in PAR-2 seconds. A is considered no worse
                than B on an instance when par2(A) <= par2(B) + threshold.
                Defaults to PERF_DIFF_THRESHOLD (1e-1), same as in pwc_wl.

        Returns:
            List of solver names that are dominated by at least one other solver.
        """
        if threshold is None:
            threshold = PERF_DIFF_THRESHOLD
        dominated: list[str] = []
        n = self.num_solvers()

        for j in range(n):
            for i in range(n):
                if i == j:
                    continue
                # Check if solver i dominates solver j
                all_no_worse = True
                at_least_one_strictly_better = False
                for path in self.keys():
                    par2_i = self.get_par2(path, i)
                    par2_j = self.get_par2(path, j)
                    if par2_i is None or par2_j is None:
                        continue
                    if par2_i > par2_j + threshold:
                        all_no_worse = False
                        break
                    if par2_i < par2_j:
                        at_least_one_strictly_better = True

                if all_no_worse and at_least_one_strictly_better:
                    name = self.get_solver_name(j)
                    if name is not None and name not in dominated:
                        dominated.append(name)
                    break

        return dominated


def filter_training_instances(
    multi_perf_data: MultiSolverDataset,
    *,
    skip_unsolvable: bool = True,
    skip_trivial_under: float | None = None,
) -> tuple[list[str], dict]:
    """
    Return instance paths to keep for graph building and training, and stats.

    When skip_unsolvable is True, drop instances where no solver solves (VBS-unsolvable).
    When skip_trivial_under is a number, drop instances that are strictly trivial:
    every solver has a result, every solver solved, and every runtime <= skip_trivial_under.

    Returns:
        (paths_to_keep, stats) where stats has "skipped_unsolvable", "skipped_trivial"
        (lists of paths), and "n_kept", "n_unsolvable", "n_trivial" (ints).
    """
    skipped_unsolvable: list[str] = []
    skipped_trivial: list[str] = []
    paths_to_keep: list[str] = []
    for path in multi_perf_data.keys():
        if skip_unsolvable and multi_perf_data.is_none_solved(path):
            skipped_unsolvable.append(path)
            continue
        if skip_trivial_under is not None and multi_perf_data.is_trivial(
            path, skip_trivial_under
        ):
            skipped_trivial.append(path)
            continue
        paths_to_keep.append(path)
    stats = {
        "skipped_unsolvable": skipped_unsolvable,
        "skipped_trivial": skipped_trivial,
        "n_kept": len(paths_to_keep),
        "n_unsolvable": len(skipped_unsolvable),
        "n_trivial": len(skipped_trivial),
    }
    return (paths_to_keep, stats)


class SingleSolverDataset:
    """A data structure for performance data for a single solver."""

    def __init__(
        self,
        perf_dict: Dict[str, Tuple[int, float]],
        solver_name: Optional[str],
        timeout: float,
    ):
        """
        Initialize single solver performance dictionary.

        Args:
            perf_dict: Dictionary mapping path to (is_solved, wc_time) tuple
            solver_name: Solver name (optional)
            timeout: Timeout in seconds
        """
        self._dict = perf_dict
        self._solver_name = solver_name
        self._timeout = timeout

    def __getitem__(self, smt_path: str) -> Tuple[int, float]:
        """Get performance tuple for a SMT file path."""
        return self._dict[smt_path]

    def __len__(self) -> int:
        """Get number of instances."""
        return len(self._dict)

    def __iter__(self):
        """Iterate over instance paths."""
        return iter(self._dict)

    def items(self):
        """Iterate over (path, (is_solved, wc_time)) pairs."""
        return self._dict.items()

    def keys(self):
        """Get all instance paths."""
        return self._dict.keys()

    def get_performance(self, smt_path: str) -> Optional[Tuple[int, float]]:
        """
        Get performance for a specific instance.

        Args:
            smt_path: SMT file path

        Returns:
            Tuple of (is_solved, wc_time) or None if not found
        """
        return self._dict.get(smt_path)

    def get_par2(self, smt_path: str) -> Optional[float]:
        """
        Get PAR-2 score for a specific instance.

        PAR-2 is the solving time if solved, otherwise twice the timeout.

        Args:
            smt_path: SMT file path

        Returns:
            PAR-2 score (solving time if solved, else 2 * timeout) or None if not found
        """
        perf = self.get_performance(smt_path)
        if perf is None:
            return None
        is_solved, wc_time = perf
        if is_solved == 1:
            return wc_time
        else:
            return 2.0 * self._timeout

    def get_solved_count(self) -> int:
        """Get number of instances solved by this solver."""
        count = 0
        for is_solved, _ in self._dict.values():
            if is_solved == 1:
                count += 1
        return count

    def get_solver_name(self) -> Optional[str]:
        """Get the solver name."""
        return self._solver_name

    def get_timeout(self) -> float:
        """Get the timeout value in seconds."""
        return self._timeout

    def to_dict(self) -> Dict[str, Tuple[int, float]]:
        """Convert to plain dictionary."""
        return self._dict.copy()


def parse_performance_json(json_path: str, timeout: float) -> MultiSolverDataset:
    """
    Parse a performance JSON file (e.g. from data/cp26/raw_data/smtcomp24_performance)
    and create a MultiSolverDataset.

    Expects format: { benchmark_path: { solver_name: { "result", "wallclock_time", ... } } }.
    Uses wallclock_time; an instance is solved only if result is "sat" or "unsat".

    Args:
        json_path: Path to the JSON file
        timeout: Timeout value in seconds (used for PAR-2 etc.)

    Returns:
        A MultiSolverDataset with (is_solved, wallclock_time) per solver.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")

    # Deterministic solver order: union of all solver names, sorted
    solver_set: set[str] = set()
    for bench in data.values():
        if isinstance(bench, dict):
            solver_set.update(bench.keys())
    solver_names = sorted(solver_set)

    # Ensure all benchmarks use same solver set (fill missing with 0, 0.0)
    solver_dict: Dict[int, str] = {i: name for i, name in enumerate(solver_names)}
    multi_perf_dict: Dict[str, List[Tuple[int, float]]] = {}

    for path, solvers in data.items():
        if not isinstance(solvers, dict):
            continue
        row: List[Tuple[int, float]] = []
        for name in solver_names:
            run = solvers.get(name)
            if run is None or not isinstance(run, dict):
                row.append((0, 0.0))
                continue
            result = run.get("result")
            wc = run.get("wallclock_time")
            if result is None or wc is None:
                row.append((0, 0.0))
                continue
            solved = 1 if str(result).lower() in ("sat", "unsat") else 0
            try:
                wc_time = float(wc)
            except (TypeError, ValueError):
                wc_time = 0.0
            row.append((solved, wc_time))
        multi_perf_dict[path] = row

    return MultiSolverDataset(multi_perf_dict, solver_dict, timeout)


def parse_as_perf_csv(csv_path: str, timeout: float) -> SingleSolverDataset:
    """
    Parse an algorithm selection performance CSV file and create a SingleSolverDataset.

    The CSV format is: benchmark,selected,solved,runtime
    where each row represents one instance with the selected solver's performance.

    Args:
        csv_path: Path to the CSV file
        timeout: Timeout value in seconds (used for benchmarking)

    Returns:
        A SingleSolverDataset containing:
        - perf_dict: path -> (is_solved, wc_time) tuple
        - solver_name: extracted from the CSV filename (without extension)
        - timeout: timeout value in seconds
    """
    perf_dict: Dict[str, Tuple[int, float]] = {}

    # Extract solver name from CSV path (filename without extension)
    solver_name = Path(csv_path).stem

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        # Skip header row: benchmark,selected,solved,runtime
        next(reader)

        # Process data rows
        for row in reader:
            if not row or not row[0]:  # Skip empty rows
                continue

            if len(row) < 4:
                continue  # Skip rows with insufficient columns

            try:
                benchmark = row[0]
                solved = int(row[2])
                runtime = float(row[3])
                perf_dict[benchmark] = (solved, runtime)
            except (ValueError, IndexError):
                # Skip rows with invalid data
                continue

    return SingleSolverDataset(perf_dict, solver_name, timeout)
