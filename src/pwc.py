import time

import numpy as np
from pathlib import Path
import joblib
import argparse
import logging

from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import StandardScaler

from .performance import parse_performance_json
from .feature import (
    extract_feature_from_csv,
    extract_feature_from_csvs_concat,
)
from .solver_selector import SolverSelector
from .utils import normalize_path

PERF_DIFF_THRESHOLD = 1e-1  # Threshold for considering performance differences


def _load_path_list(value: str | Path | None) -> list[str]:
    """Path to file (one path per line) → list of paths. None → []."""
    if value is None:
        return []
    with open(value, encoding="utf-8") as f:
        return [p.strip() for p in f if p.strip()]


def create_pairwise_samples(
    multi_perf_data,
    solver0_id,
    solver1_id,
    feature_csv_path,
    failed_instance_path: str | Path | None = None,
    extraction_time_by_path: dict[str, float] | None = None,
    feature_timeout: float | None = None,
):
    """Build pairwise (feature, label, cost) for training. Skip paths in failed_instance_path (path to file, one path per line).
    When extraction_time_by_path and feature_timeout are set, skip instances with extraction_time >= feature_timeout.
    """
    failed_paths = _load_path_list(failed_instance_path)
    failed_set = set(failed_paths)
    inputs = []
    labels = []
    costs = []
    for instance_path in multi_perf_data.keys():
        if str(instance_path) in failed_set:
            continue
        if extraction_time_by_path is not None and feature_timeout is not None:
            if extraction_time_by_path.get(normalize_path(str(instance_path)), 0.0) >= feature_timeout:
                continue
        # Handle both single CSV path (str) and multiple CSV paths (list)
        try:
            if isinstance(feature_csv_path, list):
                feature = extract_feature_from_csvs_concat(instance_path, feature_csv_path)
            else:
                feature = extract_feature_from_csv(instance_path, feature_csv_path)
        except KeyError:
            continue
        par2_0 = multi_perf_data.get_par2(instance_path, solver0_id)
        par2_1 = multi_perf_data.get_par2(instance_path, solver1_id)
        if par2_0 is None or par2_1 is None:
            continue
        label = 1 if par2_0 < par2_1 else 0  # label 1 represents solver0 is better
        cost = abs(par2_0 - par2_1)
        if cost > PERF_DIFF_THRESHOLD:
            inputs.append(feature)
            labels.append(label)
            costs.append(cost)
    inputs_array = np.array(inputs) if inputs else np.empty((0, 0))
    labels_array = np.array(labels) if labels else np.array([], dtype=int)
    costs_array = np.array(costs) if costs else np.array([])
    return inputs_array, labels_array, costs_array


class PairwiseSVM(SVC):
    def __init__(self, c_value: float = 1.0, **kwargs):
        self.c_value = c_value
        super().__init__(C=c_value, **kwargs)
        self.scaler = StandardScaler()
        self.decision_std_ = None  # Std of decision function on training data

    def fit(self, x, y, weights):
        x, y = check_X_y(x, y)
        x = self.scaler.fit_transform(x)
        super().fit(x, y, sample_weight=weights)
        # Compute std of decision function on training data for standardization
        train_decisions = super().decision_function(x)
        self.decision_std_ = np.std(train_decisions)
        # Avoid division by zero if all decisions are identical
        if self.decision_std_ == 0:
            self.decision_std_ = 1.0
        return self

    def predict(self, x):
        x = self.scaler.transform(x)
        return super().predict(x)

    def decision_function(self, x):
        x = self.scaler.transform(x)
        return super().decision_function(x)

    def get_standardized_score(self, x):
        """Get decision function score standardized by training std."""
        return self.decision_function(x) / self.decision_std_


class PwcSelector(SolverSelector):
    def __init__(
        self,
        model_matrix,
        feature_csv_path,
        random_seed: int = 42,
        fallback_solver_ids: list[int] | None = None,
        failed_instance_paths: str | Path | None = None,
    ):
        self.model_type = "SVM"
        self.model_matrix = model_matrix
        self.solver_size = model_matrix.shape[0]
        self.feature_csv_path = feature_csv_path
        self.random_seed = random_seed
        self.fallback_solver_ids = list(fallback_solver_ids) if fallback_solver_ids else []
        paths = _load_path_list(failed_instance_paths)
        self._failed_set = set(paths)
        # Optional: when set, instances with extraction_time > feature_timeout use SBS (sbs_solver_id)
        self.feature_timeout: float | None = None
        self.extraction_time_by_path: dict[str, float] | None = None
        self.sbs_solver_id: int | None = None
        # Optional: set of paths with failed=1 in extraction_times CSV; use this label for SBS at eval (not time comparison)
        self.failed_paths_from_csv: set[str] | None = None

    def save(self, save_dir):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = f"{save_dir}/model.joblib"
        joblib.dump(self, save_path)
        logging.info(f"Saved PWC_{self.model_type} model at {save_path}")

    @staticmethod
    def load(load_path):
        return joblib.load(load_path)

    def _get_rank_lst(self, feature, random_seed=42):
        btor2kw_array = np.array(feature).reshape(1, -1)
        votes = np.zeros(self.solver_size, dtype=int)
        for i in range(self.solver_size):
            for j in range(i + 1, self.solver_size):
                prediction = self.model_matrix[i, j].predict(btor2kw_array)
                if prediction[0]:
                    votes[i] += 1
                else:
                    votes[j] += 1
        rng = np.random.default_rng(random_seed)
        random_tiebreaker = rng.random(self.solver_size)
        structured_votes = np.rec.fromarrays(
            [votes, random_tiebreaker], names="votes, random_tiebreaker"
        )
        sorted_indices = np.argsort(
            structured_votes, order=("votes", "random_tiebreaker")
        )[::-1]
        return sorted_indices

    def algorithm_select(self, instance_path):
        """
        input instance path, output solver id.
        If instance is in failed list → use fallback (fallback_solver_ids[0]).
        Else extract feature; if not available, raise KeyError.
        """
        selected, _, _ = self.algorithm_select_with_info(instance_path)
        return selected

    def algorithm_select_with_info(self, instance_path):
        """
        Return (solver_id, overhead_sec, feature_fail).
        overhead_sec is the time for feature lookup + model inference (SVM).
        feature_fail is True if feature extraction failed (instance not in CSV or in failed set without fallback).
        When failed_paths_from_csv and sbs_solver_id are set, instances in that set return (sbs_solver_id, 0.0, True)
        (use SBS, no overhead). The CSV label failed=1 is used for this decision, not extraction_time comparison.
        """
        path_str = str(instance_path)
        path_norm = normalize_path(path_str)
        sbs_solver_id = getattr(self, "sbs_solver_id", None)
        failed_from_csv = getattr(self, "failed_paths_from_csv", None)
        if failed_from_csv is not None and sbs_solver_id is not None:
            if path_norm in failed_from_csv:
                return (sbs_solver_id, 0.0, True)

        random_seed = self.random_seed
        feature_csv_path = self.feature_csv_path
        if path_str in self._failed_set:
            if self.fallback_solver_ids:
                return (self.fallback_solver_ids[0], 0.0, False)
            raise ValueError(
                "Instance is in failed list but fallback_solver_ids is empty."
            )
        if feature_csv_path is None:
            raise ValueError(
                "feature_csv_path not set in PwcSelector. "
                "It must be provided during model creation or loading."
            )
        t0 = time.perf_counter()
        try:
            if isinstance(feature_csv_path, list):
                feature = extract_feature_from_csvs_concat(
                    instance_path, feature_csv_path
                )
            else:
                feature = extract_feature_from_csv(
                    instance_path, feature_csv_path
                )
            selected_id = self._get_rank_lst(feature, random_seed)[0]
            overhead = time.perf_counter() - t0
            return (selected_id, overhead, False)
        except KeyError:
            overhead = time.perf_counter() - t0
            fallback = self.fallback_solver_ids[0] if self.fallback_solver_ids else 0
            return (fallback, overhead, True)


def train_pwc(
    multi_perf_data,
    save_dir,
    feature_csv_path=None,
    svm_c: float = 1.0,
    random_seed: int = 42,
    timeout_instance_paths: str | Path | None = None,
    extraction_time_by_path: dict[str, float] | None = None,
    feature_timeout: float | None = None,
):
    """
    Train pairwise selector. timeout_instance_paths: path to file (e.g. timeout{N}_failed_paths.txt)
    listing instance paths to exclude from training; used for failback solver ranking.
    When extraction_time_by_path and feature_timeout are set, instances with extraction_time >= feature_timeout
    are excluded from training.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    if extraction_time_by_path is not None and feature_timeout is not None:
        n_skipped = sum(
            1 for p in multi_perf_data.keys()
            if extraction_time_by_path.get(normalize_path(str(p)), 0.0) >= feature_timeout
        )
        if n_skipped:
            logging.info(
                "Skipping %d instances with extraction_time >= %.1fs for training",
                n_skipped,
                feature_timeout,
            )
    solver_size = multi_perf_data.num_solvers()
    # Use the single best solver (SBS) on the training data as the default fallback.
    # This solver is used when feature extraction fails or an instance is in the
    # failed/timeout list.
    sbs_solver_id = multi_perf_data.get_best_solver_id()
    fallback_solver_ids = [sbs_solver_id]

    model_matrix = np.empty((solver_size, solver_size), dtype=object)
    model_matrix[:] = None

    for i in range(solver_size):
        for j in range(i + 1, solver_size):
            (
                inputs_array,
                labels_array,
                costs_array,
            ) = create_pairwise_samples(
                multi_perf_data,
                i,
                j,
                feature_csv_path,
                failed_instance_path=timeout_instance_paths,
                extraction_time_by_path=extraction_time_by_path,
                feature_timeout=feature_timeout,
            )
            if len(labels_array) == 0:
                continue
            unique_labels = np.unique(labels_array)
            if len(unique_labels) == 1:
                model = DummyClassifier(strategy="constant", constant=int(unique_labels[0]))
                model.fit(inputs_array, labels_array)
            else:
                model = PairwiseSVM(c_value=svm_c)
                model.fit(inputs_array, labels_array, costs_array)
            model_matrix[i, j] = model
    pwc_model = PwcSelector(
        model_matrix,
        feature_csv_path,
        random_seed=random_seed,
        fallback_solver_ids=fallback_solver_ids,
        failed_instance_paths=timeout_instance_paths,
    )
    pwc_model.save(save_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory to save the trained models",
    )
    parser.add_argument(
        "--perf-json", type=str, help="The training performance JSON path"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1200.0,
        help="Timeout value in seconds (default: 1200.0)",
    )
    parser.add_argument(
        "--feature-csv",
        type=str,
        required=True,
        help="Path to the features CSV file",
    )
    parser.add_argument(
        "--svm-c",
        type=float,
        default=1.0,
        help="Regularization parameter C for SVM (default: 1.0)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for solver selection tie-breaking (default: 42)",
    )
    parser.add_argument(
        "--failed-instances",
        type=str,
        default=None,
        help="Path to file listing failed instance paths (one per line, e.g. timeout10_failed_paths.txt); excluded from training and used for failback solver ranking.",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    save_dir = args.save_dir
    timeout = args.timeout
    train_dataset = parse_performance_json(args.perf_json, timeout)

    if args.failed_instances:
        n = len(_load_path_list(args.failed_instances))
        logging.info(
            "Failed instance paths: %d from %s",
            n,
            args.failed_instances,
        )

    logging.info(
        f"Training performance parse: {len(train_dataset)} benchmarks and {train_dataset.num_solvers()} solvers from {args.perf_json}"
    )

    train_pwc(
        train_dataset,
        save_dir,
        args.feature_csv,
        svm_c=args.svm_c,
        random_seed=args.random_seed,
        timeout_instance_paths=args.failed_instances,
    )


if __name__ == "__main__":
    main()
