"""Data structures for performance and solver data."""

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
