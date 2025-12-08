"""Data structures for performance and solver data."""

from typing import Dict, List, Tuple, Optional


class MultiSolverDataset:
    """A data structure for performance data across multiple solvers."""

    def __init__(
        self,
        perf_dict: Dict[str, List[Tuple[int, float]]],
        solver_dict: Dict[int, str],
        timeout: float,
    ):
        """
        Initialize performance dictionary.

        Args:
            perf_dict: Dictionary mapping path to list of (is_solved, wc_time) tuples
            solver_dict: Dictionary mapping solver ID to solver name
            timeout: Timeout in seconds
        """
        self._dict = perf_dict
        self._solver_dict = solver_dict
        self._timeout = timeout

    def __getitem__(self, path: str) -> List[Tuple[int, float]]:
        """Get performance list for a path."""
        return self._dict[path]

    def __len__(self) -> int:
        """Get number of instances."""
        return len(self._dict)

    def __iter__(self):
        """Iterate over instance paths."""
        return iter(self._dict)

    def items(self):
        """Iterate over (path, performance_list) pairs."""
        return self._dict.items()

    def keys(self):
        """Get all instance paths."""
        return self._dict.keys()

    def get_performance(self, path: str, solver_id: int) -> Optional[Tuple[int, float]]:
        """
        Get performance for a specific instance and solver.

        Args:
            path: Instance path
            solver_id: Solver ID

        Returns:
            Tuple of (is_solved, wc_time) or None if not found
        """
        if path not in self._dict:
            return None
        perf_list = self._dict[path]
        if solver_id >= len(perf_list):
            return None
        return perf_list[solver_id]

    def get_par2(self, path: str, solver_id: int) -> Optional[float]:
        """
        Get PAR-2 score for a specific instance and solver.

        PAR-2 is the solving time if solved, otherwise twice the timeout.

        Args:
            path: Instance path
            solver_id: Solver ID

        Returns:
            PAR-2 score (solving time if solved, else 2 * timeout) or None if not found
        """
        perf = self.get_performance(path, solver_id)
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

    def get_solver_name(self, solver_id: int) -> Optional[str]:
        """Get solver name by ID."""
        return self._solver_dict.get(solver_id)

    def get_solver_dict(self) -> Dict[int, str]:
        """Get the solver dictionary."""
        return self._solver_dict.copy()

    def num_solvers(self) -> int:
        """Get number of solvers."""
        return len(self._solver_dict)

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

    def __getitem__(self, path: str) -> Tuple[int, float]:
        """Get performance tuple for a path."""
        return self._dict[path]

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

    def get_performance(self, path: str) -> Optional[Tuple[int, float]]:
        """
        Get performance for a specific instance.

        Args:
            path: Instance path

        Returns:
            Tuple of (is_solved, wc_time) or None if not found
        """
        return self._dict.get(path)

    def get_par2(self, path: str) -> Optional[float]:
        """
        Get PAR-2 score for a specific instance.

        PAR-2 is the solving time if solved, otherwise twice the timeout.

        Args:
            path: Instance path

        Returns:
            PAR-2 score (solving time if solved, else 2 * timeout) or None if not found
        """
        perf = self.get_performance(path)
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
