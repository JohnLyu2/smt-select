"""WL (Weisfeiler-Lehman) graph-kernel based pairwise algorithm selection."""

from __future__ import annotations

import argparse
import gc
import logging
import os
import signal
import sys
from pathlib import Path

import joblib
import numpy as np
from grakel import Graph
from grakel.kernels import WeisfeilerLehman
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC

from .graph_rep import smt_to_graph, smt_graph_to_grakel
from .performance import parse_performance_csv
from .performance import MultiSolverDataset
from .solver_selector import SolverSelector

PERF_DIFF_THRESHOLD = 1e-1  # Threshold for considering performance differences


def _timeout_handler(_signum: int, _frame: object) -> None:
    raise TimeoutError("Graph build timed out")


def _suppress_z3_destructor_noise() -> None:
    """Run GC with stderr suppressed to avoid z3 AstRef.__del__ messages after timeout."""
    devnull = open(os.devnull, "w")
    old = sys.stderr
    try:
        sys.stderr = devnull
        gc.collect()
    finally:
        sys.stderr = old
        devnull.close()


def build_smt_graph_timeout(smt_path: str | Path, timeout_sec: int) -> Graph | None:
    """Build a GraKel graph for the SMT instance with a timeout. Returns None on timeout."""
    path = Path(smt_path)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        graph_dict = smt_to_graph(path)
        graph = smt_graph_to_grakel(graph_dict)
        signal.alarm(0)
        return graph
    except TimeoutError:
        logging.info(
            "Timeout (%ds) while building graph for %s", timeout_sec, smt_path
        )
        _suppress_z3_destructor_noise()
        return None
    except RecursionError:
        logging.info(
            "Recursion limit exceeded while building graph for %s", smt_path
        )
        _suppress_z3_destructor_noise()
        return None
    except Exception as e:
        if "recursion" in str(e).lower():
            logging.info(
                "Recursion limit exceeded while building graph for %s", smt_path
            )
        else:
            logging.info("Error building graph for %s: %s", smt_path, e)
        _suppress_z3_destructor_noise()
        return None
    finally:
        signal.alarm(0)


def generate_graph_dict(
    instance_paths: list[str], timeout_sec: int
) -> tuple[dict[str, Graph], list[str]]:
    """Build graphs for each instance. Returns (path -> graph, list of paths that failed: timeout/recursion/error)."""
    graph_dict: dict[str, Graph] = {}
    failed_list: list[str] = []
    for p in instance_paths:
        g = build_smt_graph_timeout(p, timeout_sec)
        if g is not None:
            graph_dict[p] = g
        else:
            failed_list.append(p)
    logging.info(
        "Graphs: %d built, %d failed (of %d instances)",
        len(graph_dict),
        len(failed_list),
        len(instance_paths),
    )
    return graph_dict, failed_list


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


def sorted_timeout_solvers(
    multi_perf_data: MultiSolverDataset,
    timeout_paths: list[str],
) -> list[int]:
    """
    Rank solvers by average PAR-2 over the given paths (for instances that timed out
    during graph build). Returns solver ids from best to worst.
    """
    if not timeout_paths:
        # No timeouts: rank over full dataset
        paths = list(multi_perf_data.keys())
    else:
        paths = timeout_paths
    n_solvers = multi_perf_data.num_solvers()
    if n_solvers == 0:
        return []
    scores: list[tuple[int, float]] = []
    for sid in range(n_solvers):
        total = 0.0
        count = 0
        for p in paths:
            par2 = multi_perf_data.get_par2(p, sid)
            if par2 is not None:
                total += par2
                count += 1
        avg = total / count if count > 0 else float("inf")
        scores.append((sid, avg))
    scores.sort(key=lambda x: x[1])
    return [sid for sid, _ in scores]


class PwcWlSelector(SolverSelector):
    """Pairwise WL graph-kernel SVM selector. algorithm_select(instance_path) -> solver id."""

    def __init__(
        self,
        svm_matrix: np.ndarray,
        indice_matrix: np.ndarray,
        kernel: WeisfeilerLehman,
        graph_timeout: int,
        timeout_solver_ids: list[int],
        wl_iter: int,
        solver_id_dict: dict[int, str],
        random_seed: int = 42,
    ):
        self.solver_size = svm_matrix.shape[0]
        self.svm_matrix = svm_matrix
        self.indice_matrix = indice_matrix
        self.kernel = kernel
        self.graph_timeout = graph_timeout
        self.timeout_solver_ids = timeout_solver_ids
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
        path = Path(instance_path)
        graph = build_smt_graph_timeout(path, self.graph_timeout)
        if graph is None:
            return self.timeout_solver_ids[0]
        rank = self._get_rank_lst(graph)
        return rank[0]


def train_pwc_wl(
    multi_perf_data: MultiSolverDataset,
    wl_iter: int,
    save_dir: str | Path,
    graph_timeout: int = 10,
) -> None:
    """Train pairwise WL-based SVM models and save PwcWlSelector."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    solver_size = multi_perf_data.num_solvers()
    instance_paths = list(multi_perf_data.keys())
    graph_dict, failed_list = generate_graph_dict(instance_paths, graph_timeout)
    if not graph_dict:
        raise ValueError(
            "No graphs could be built (all failed: timeout/recursion/error?). Increase --graph-timeout or check instances."
        )
    train_paths = list(graph_dict.keys())
    train_graphs = [graph_dict[p] for p in train_paths]

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

    timeout_solver_ids = sorted_timeout_solvers(multi_perf_data, failed_list)
    solver_id_dict = multi_perf_data.get_solver_id_dict()

    model = PwcWlSelector(
        svm_matrix,
        indice_matrix,
        wl_kernel,
        graph_timeout,
        timeout_solver_ids,
        wl_iter,
        solver_id_dict,
    )
    model.save(save_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train WL-based pairwise algorithm selection model"
    )
    parser.add_argument(
        "--perf-csv",
        type=str,
        required=True,
        help="Path to the performance CSV",
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
        default=3,
        help="Weisfeiler-Lehman iteration count (default: 3)",
    )
    parser.add_argument(
        "--graph-timeout",
        type=int,
        default=10,
        help="Timeout in seconds for graph build per instance (default: 10)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1200.0,
        help="PAR-2 timeout in seconds (default: 1200.0)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    multi_perf_data = parse_performance_csv(args.perf_csv, args.timeout)
    logging.info(
        "Training: %d instances, %d solvers from %s",
        len(multi_perf_data),
        multi_perf_data.num_solvers(),
        args.perf_csv,
    )
    train_pwc_wl(
        multi_perf_data,
        args.wl_iter,
        args.save_dir,
        args.graph_timeout,
    )


if __name__ == "__main__":
    main()
