import numpy as np
from pathlib import Path
import joblib
import argparse
import logging

import xgboost as xgb
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import StandardScaler

from .parser import parse_performance_csv

PERF_DIFF_THRESHOLD = 1e-3  # Threshold for considering performance differences


def placehold_feature_extract(instance_path):
    return np.random.rand(10)


# now the target func is par2
def create_pairwise_samples(perf_data, solver0_id, solver1_id):
    inputs = []
    labels = []
    costs = []
    for instance_path in perf_data.keys():
        feature = placehold_feature_extract(instance_path)
        par2_0 = perf_data.get_par2(instance_path, solver0_id)
        par2_1 = perf_data.get_par2(instance_path, solver1_id)
        label = 1 if par2_0 < par2_1 else 0  # label 1 represents solver0 is better
        cost = abs(par2_0 - par2_1)
        if cost > PERF_DIFF_THRESHOLD:  # if the performance difference is 0, ignore
            inputs.append(feature)
            labels.append(label)
            costs.append(cost)
    inputs_array = np.array(inputs)
    labels_array = np.array(labels)
    costs_array = np.array(costs)
    return inputs_array, labels_array, costs_array


class PairwiseXGBoost(xgb.XGBClassifier):
    def __init__(self, random_state=42, **kwargs):
        super().__init__(n_jobs=8, random_state=random_state, **kwargs)

    def fit(self, x, y, weights):
        x, y = check_X_y(x, y)
        super().fit(x, y, sample_weight=weights)
        return self


class PairwiseSVM(SVC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scaler = StandardScaler()

    def fit(self, x, y, weights):
        x, y = check_X_y(x, y)
        x = self.scaler.fit_transform(x)
        super().fit(x, y, sample_weight=weights)
        return self

    def predict(self, x):
        x = self.scaler.transform(x)
        return super().predict(x)

    def decision_function(self, x):
        x = self.scaler.transform(x)
        return super().decision_function(x)


class PwcModel:
    def __init__(self, model_matrix, xg_flag):
        self.model_type = "XG" if xg_flag else "SVM"
        self.model_matrix = model_matrix
        self.solver_size = model_matrix.shape[0]

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

    def algorithm_select(self, instance_path, random_seed=42):
        """
        input instance path, output list of tool-config ids; for cross validation
        """
        feature = placehold_feature_extract(instance_path)
        selected_id = self._get_rank_lst(feature, random_seed)[0]
        return selected_id


def train_pwc(perf_data, save_dir, xg_flag=False):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    solver_size = perf_data.num_solvers()

    model_matrix = np.empty((solver_size, solver_size), dtype=object)
    model_matrix[:] = None

    for i in range(solver_size):
        for j in range(i + 1, solver_size):
            (
                inputs_array,
                labels_array,
                costs_array,
            ) = create_pairwise_samples(perf_data, i, j)
            unique_labels = np.unique(labels_array)
            if len(unique_labels) == 1:
                model = DummyClassifier(strategy="constant", constant=unique_labels[0])
                model.fit(inputs_array, labels_array)
            else:
                model = PairwiseSVM()
                if xg_flag:
                    model = PairwiseXGBoost()
                model.fit(inputs_array, labels_array, costs_array)
            model_matrix[i, j] = model
    pwc_model = PwcModel(model_matrix, xg_flag)
    pwc_model.save(save_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xg", action="store_true", help="Flag to use XGBoost")
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Directory to save the trained models",
    )
    parser.add_argument(
        "--perf-csv", type=str, help="The training performance csv path"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1200.0,
        help="Timeout value in seconds (default: 1200.0)",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    xg_flag = args.xg
    save_dir = args.save_dir
    timeout = args.timeout
    train_dataset = parse_performance_csv(args.perf_csv, timeout)

    logging.info(
        f"Training performance parse: {len(train_dataset)} benchmarks and {train_dataset.num_solvers()} solvers from {args.perf_csv}"
    )

    train_pwc(train_dataset, save_dir, xg_flag)


if __name__ == "__main__":
    main()
