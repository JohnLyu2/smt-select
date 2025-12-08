import numpy as np
from pathlib import Path
from functools import partial
import joblib
import argparse
import logging

import xgboost as xgb
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import StandardScaler

from .preprocess import parse_from_tsv
from .bokw import get_bokw, create_bokws_from_perf_dict
from .evaluate import get_par_n

PERF_DIFF_THRESHOLD = 1e-10  # Threshold for considering performance differences


def generate_training_samples_for_config_pair(
    training_perf_dict, tool_config_pair, res2target_func, feature_dict
):
    tool_config0, tool_config1 = tool_config_pair
    inputs = []
    labels = []
    costs = []
    for btor2_path, perf_lst in training_perf_dict.items():
        feature = feature_dict[btor2_path]
        result_tuple0 = perf_lst[tool_config0]
        result_tuple1 = perf_lst[tool_config1]
        target0 = res2target_func(result_tuple0)
        target1 = res2target_func(result_tuple1)
        # here assume always the less target the better
        label = (
            1 if target0 < target1 else 0
        )  # label 1 represents tool_config0 is better
        cost = abs(target0 - target1)
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


class PwcBokwModel:
    def __init__(self, model_matrix, xg_flag, tool_dict):
        self.model_type = "XG" if xg_flag else "SVM"
        self.model_matrix = model_matrix
        self.tool_size = model_matrix.shape[0]
        self.tool_dict = tool_dict

    def save(self, save_dir):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = f"{save_dir}/model.joblib"
        joblib.dump(self, save_path)
        logging.info(f"Saved PWC_{self.model_type}_BoKW model at {save_path}")

    @staticmethod
    def load(load_path):
        return joblib.load(load_path)

    def _get_rank_lst(self, bokw, random_seed=42):
        btor2kw_array = np.array(bokw).reshape(1, -1)
        votes = np.zeros(self.tool_size, dtype=int)
        for i in range(self.tool_size):
            for j in range(i + 1, self.tool_size):
                prediction = self.model_matrix[i, j].predict(btor2kw_array)
                if prediction[0]:
                    votes[i] += 1
                else:
                    votes[j] += 1
        rng = np.random.default_rng(random_seed)
        random_tiebreaker = rng.random(self.tool_size)
        structured_votes = np.rec.fromarrays(
            [votes, random_tiebreaker], names="votes, random_tiebreaker"
        )
        sorted_indices = np.argsort(
            structured_votes, order=("votes", "random_tiebreaker")
        )[::-1]
        return sorted_indices

    def algorithm_select(self, btor2_path, top_k=1, random_seed=42):
        """
        input btor2 path, output list of tool-config ids; for cross validation
        """
        assert top_k > 0, f"invalid top_k value {top_k}"
        assert top_k <= self.tool_size, f"invalid top_k value {top_k}"
        bokw = get_bokw(btor2_path)
        selected_ids = self._get_rank_lst(bokw, random_seed)[:top_k]
        return selected_ids

    def algorithm_select_name(self, btor2path, top_k=1, random_seed=42):
        """
        input btor2 path, output list of (tool, config) tuples
        Now only use bokw as the feature
        """
        selected_ids = self.algorithm_select(btor2path, top_k, random_seed)
        tool_configs = []
        for selected_id in selected_ids:
            tool, config, _ = self.tool_dict[selected_id][1].split(".")
            tool_configs.append((tool, config))
        return tool_configs


def train_pwc_bokw(perf_dict, tool_dict, target_func, save_dir, xg_flag=False):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    tool_size = len(tool_dict)
    bokw_dict = create_bokws_from_perf_dict(perf_dict)
    logging.info("BoKW features extracted")

    model_matrix = np.empty((tool_size, tool_size), dtype=object)
    model_matrix[:] = None

    for i in range(tool_size):
        for j in range(i + 1, tool_size):
            (
                inputs_array,
                labels_array,
                costs_array,
            ) = generate_training_samples_for_config_pair(
                perf_dict, (i, j), target_func, bokw_dict
            )
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
    pwc_model = PwcBokwModel(model_matrix, xg_flag, tool_dict)
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

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    xg_flag = args.xg
    save_dir = args.save_dir
    train_perf_dict, tool_dict = parse_from_tsv(args.perf_csv)
    logging.info(
        f"Training performance parse: {len(train_perf_dict)} benchmarks and {len(tool_dict)} tools from {args.perf_csv}"
    )

    par2_func = partial(get_par_n, n=2, timeout=900.0)

    train_pwc_bokw(train_perf_dict, tool_dict, par2_func, save_dir, xg_flag)


if __name__ == "__main__":
    main()
