"""WL (Weisfeiler-Lehman) graph-kernel based pairwise algorithm selection."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import joblib
import numpy as np
from grakel import Graph
from grakel.kernels import WeisfeilerLehman
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC

from .defaults import DEFAULT_BENCHMARK_ROOT
from .performance import parse_performance_json, MultiSolverDataset
from .performance import PERF_DIFF_THRESHOLD
from .solver_selector import SolverSelector
from .graph_rep import (
    build_smt_graph_dict_timeout,
    generate_graph_dicts_parallel,
    smt_graph_to_grakel,
    _suppress_z3_destructor_noise,
)


def generate_labels_for_config_pair(
    multi_perf_data: MultiSolverDataset,
    solver_i: int,
    solver_j: int,
    instance_paths: list[str],
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """
    For each instance in instance_paths, compute pairwise label (i better=1, j better=0)
    and cost = |PAR2_i - PAR2_j|. Only include pairs with cost > PERF_DIFF_THRESHOLD.
    Returns (indices into instance_paths, labels, costs).
    """
    indices: list[int] = []
    labels: list[int] = []
    costs: list[float] = []
    for idx, path in enumerate(instance_paths):
        par2_i = multi_perf_data.get_par2(path, solver_i)
        par2_j = multi_perf_data.get_par2(path, solver_j)
        if par2_i is None or par2_j is None:
            continue
        label = 1 if par2_i < par2_j else 0
        cost = abs(par2_i - par2_j)
        if cost > PERF_DIFF_THRESHOLD:
            indices.append(idx)
            labels.append(label)
            costs.append(cost)
    return indices, np.array(labels), np.array(costs)


class PwcWlSelector(SolverSelector):
    """Pairwise WL graph-kernel SVM selector. algorithm_select(instance_path) -> solver id."""

    def __init__(
        self,
        svm_matrix: np.ndarray,
        indice_matrix: np.ndarray,
        kernel: WeisfeilerLehman,
        graph_timeout: int,
        fallback_solver_ids: list[int],
        wl_iter: int,
        solver_id_dict: dict[int, str],
        random_seed: int = 42,
    ):
        self.solver_size = svm_matrix.shape[0]
        self.svm_matrix = svm_matrix
        self.indice_matrix = indice_matrix
        self.kernel = kernel
        self.graph_timeout = graph_timeout
        self.fallback_solver_ids = fallback_solver_ids
        self.wl_iter = wl_iter
        self.solver_id_dict = solver_id_dict
        self.random_seed = random_seed

    def save(self, save_dir: str | Path) -> None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        path = Path(save_dir) / "model.joblib"
        joblib.dump(self, path)
        logging.info("Saved PWC_WL (iter=%d) model at %s", self.wl_iter, path)

    @staticmethod
    def load(load_path: str | Path) -> "PwcWlSelector":
        return joblib.load(load_path)

    def _get_rank_lst(self, graph: Graph, random_seed: int | None = None) -> list[int]:
        if random_seed is None:
            random_seed = self.random_seed
        test_kernel_mat = self.kernel.transform([graph])
        votes = np.zeros(self.solver_size, dtype=int)
        for i in range(self.solver_size):
            for j in range(i + 1, self.solver_size):
                svm = self.svm_matrix[i][j]
                if svm is None:
                    continue
                indices = self.indice_matrix[i][j]
                kernel_ij = test_kernel_mat[:, indices]
                pred = svm.predict(kernel_ij)
                if pred[0]:
                    votes[i] += 1
                else:
                    votes[j] += 1
        rng = np.random.default_rng(random_seed)
        tiebreaker = rng.random(self.solver_size)
        rec = np.rec.fromarrays(
            [votes, tiebreaker], names="votes, random_tiebreaker"
        )
        order = np.argsort(rec, order=("votes", "random_tiebreaker"))[::-1]
        return order.tolist()

    def algorithm_select(self, instance_path: str | Path) -> int:
        """Return solver id for the given instance (path to .smt2 file)."""
        selected, _, _ = self.algorithm_select_with_info(instance_path)
        return selected

    def algorithm_select_with_info(
        self, instance_path: str | Path
    ) -> tuple[int, float, bool]:
        """
        Return (solver_id, overhead_sec, feature_fail).

        overhead_sec measures time for graph construction + kernel/SVM inference.
        feature_fail is True when graph construction fails (WL features unavailable);
        in that case, the selector falls back to the train SBS solver.
        """
        path = Path(instance_path)
        t0 = time.perf_counter()
        graph_dict = build_smt_graph_dict_timeout(path, self.graph_timeout)
        if graph_dict is None:
            _suppress_z3_destructor_noise()
            overhead = time.perf_counter() - t0
            return (self.fallback_solver_ids[0], overhead, True)
        graph = smt_graph_to_grakel(graph_dict)
        rank = self._get_rank_lst(graph)
        _suppress_z3_destructor_noise()
        overhead = time.perf_counter() - t0
        return (rank[0], overhead, False)


def train_pwc_wl(
    multi_perf_data: MultiSolverDataset,
    wl_iter: int,
    save_dir: str | Path,
    graph_timeout: int = 5,
    jobs: int = 1,
) -> None:
    """Train pairwise WL-based SVM models and save PwcWlSelector.

    Graphs are built using generate_graph_dicts_parallel from graph_rep, so
    when jobs > 1 graph construction is parallelized; jobs <= 1 falls back to
    a sequential path.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    solver_size = multi_perf_data.num_solvers()
    instance_paths = list(multi_perf_data.keys())
    # Build underlying SMT graphs (dict representation) in parallel, then convert
    # to GraKel Graph objects for WL kernel training.
    graph_dict_raw, failed_list = generate_graph_dicts_parallel(
        instance_paths, graph_timeout, jobs
    )
    if not graph_dict_raw:
        raise ValueError(
            "No graphs could be built (all failed: timeout/recursion/error?). Increase --graph-timeout or check instances."
        )
    train_paths = list(graph_dict_raw.keys())
    train_graphs = [smt_graph_to_grakel(graph_dict_raw[p]) for p in train_paths]

    wl_kernel = WeisfeilerLehman(n_iter=wl_iter, normalize=True)
    k_mat = wl_kernel.fit_transform(train_graphs)
    k_mat_np = np.array(k_mat)

    svm_matrix = np.empty((solver_size, solver_size), dtype=object)
    svm_matrix[:] = None
    indice_matrix = np.empty((solver_size, solver_size), dtype=object)
    indice_matrix[:] = None

    for i in range(solver_size):
        for j in range(i + 1, solver_size):
            indices, label_arr, cost_arr = generate_labels_for_config_pair(
                multi_perf_data, i, j, train_paths
            )
            indice_matrix[i][j] = indices
            if len(indices) == 0:
                continue
            sub = k_mat_np[np.ix_(indices, indices)]
            unique_labels = np.unique(label_arr)
            if len(unique_labels) == 1:
                svm_ij = DummyClassifier(strategy="constant", constant=int(unique_labels[0]))
                svm_ij.fit(sub, label_arr)
            else:
                svm_ij = SVC(kernel="precomputed")
                svm_ij.fit(sub, label_arr, sample_weight=cost_arr)
            svm_matrix[i][j] = svm_ij

    # Use the single best solver (SBS) on the training data as the default
    # fallback when graph construction fails for an instance.
    sbs_solver_id = multi_perf_data.get_best_solver_id()
    fallback_solver_ids = [sbs_solver_id]
    solver_id_dict = multi_perf_data.get_solver_id_dict()

    model = PwcWlSelector(
        svm_matrix,
        indice_matrix,
        wl_kernel,
        graph_timeout,
        fallback_solver_ids,
        wl_iter,
        solver_id_dict,
    )
    model.save(save_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train WL-based pairwise algorithm selection model"
    )
    parser.add_argument(
        "--perf-json",
        type=str,
        required=True,
        help="Path to the performance JSON (e.g. train.json or test.json from splits)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--wl-iter",
        type=int,
        default=2,
        help="Weisfeiler-Lehman iteration count (default: 2)",
    )
    parser.add_argument(
        "--graph-timeout",
        type=int,
        default=5,
        help="Graph build timeout in seconds (default: 5)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel workers for graph building; 1 = sequential (default: 1)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1200.0,
        help="Timeout value in seconds (default: 1200.0)",
    )
    parser.add_argument(
        "--benchmark-root",
        type=str,
        default=DEFAULT_BENCHMARK_ROOT,
        help="Root directory for instance paths (default: project default).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    multi_perf_data = parse_performance_json(args.perf_json, args.timeout)
    if args.benchmark_root:
        root = Path(args.benchmark_root).resolve()
        if not root.is_dir():
            raise ValueError(f"--benchmark-root is not a directory: {root}")
        rebased = {str(root / p): multi_perf_data[p] for p in multi_perf_data.keys()}
        multi_perf_data = MultiSolverDataset(
            rebased,
            multi_perf_data.get_solver_id_dict(),
            multi_perf_data.get_timeout(),
        )
        logging.info("Instance paths rebased under benchmark root: %s", root)
    logging.info(
        "Training: %d instances, %d solvers from %s",
        len(multi_perf_data),
        multi_perf_data.num_solvers(),
        args.perf_json,
    )
    train_pwc_wl(
        multi_perf_data,
        args.wl_iter,
        args.save_dir,
        args.graph_timeout,
        jobs=args.jobs,
    )


if __name__ == "__main__":
    main()
