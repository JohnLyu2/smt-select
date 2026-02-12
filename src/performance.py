"""Data structures for performance and solver data."""

import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional


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
