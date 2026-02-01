#!/usr/bin/env python3
"""
Count disagreements between predict() and predict_proba() > 0.5 for PairwiseSVM.
"""

import numpy as np
from pathlib import Path

from src.pwc import train_pwc, PwcSelector
from src.feature import extract_feature_from_csv
from scripts.cross_validate import load_all_instances_from_folds, create_subset_dataset


def count_disagreements(model, feature_csv_path, test_instances):
    """Count disagreements between predict() and predict_proba() > 0.5."""
    total = 0
    disagreements = 0

    for instance_path in test_instances:
        feature = extract_feature_from_csv(instance_path, feature_csv_path).reshape(1, -1)

        for i in range(model.solver_size):
            for j in range(i + 1, model.solver_size):
                clf = model.model_matrix[i, j]
                pred = clf.predict(feature)[0]

                probs = clf.predict_proba(feature)[0]
                if len(probs) == 1:
                    prob_class1 = 1.0 if clf.classes_[0] == 1 else 0.0
                else:
                    class_1_idx = np.where(clf.classes_ == 1)[0][0]
                    prob_class1 = probs[class_1_idx]

                pred_winner = i if pred == 1 else j
                proba_winner = i if prob_class1 > 0.5 else j

                total += 1
                if pred_winner != proba_winner:
                    disagreements += 1

    return disagreements, total


def main():
    import argparse
    import tempfile

    parser = argparse.ArgumentParser()
    parser.add_argument("--logic", type=str, default="BV")
    args = parser.parse_args()

    logic = args.logic
    folds_dir = Path(f"data/perf_data/folds/{logic}")
    feature_csv = f"data/features/syntactic/catalog0/{logic}.CSV"
    if not Path(feature_csv).exists():
        feature_csv = f"data/features/syntactic/catalog0/{logic}.csv"

    # Load data and split
    multi_perf_data = load_all_instances_from_folds(folds_dir, timeout=1200.0)
    all_instances = list(multi_perf_data.keys())

    fold_files = sorted(folds_dir.glob("*.csv"), key=lambda x: int(x.stem))
    test_instances = set()
    with open(fold_files[0]) as f:
        for line in f:
            parts = line.strip().split(",")
            if parts[0] and not parts[0].startswith("instance"):
                test_instances.add(parts[0])
    test_instances = test_instances.intersection(set(all_instances))

    train_instances = [p for p in all_instances if p not in test_instances]
    train_data = create_subset_dataset(multi_perf_data, train_instances)

    # Train model
    with tempfile.TemporaryDirectory() as tmpdir:
        train_pwc(train_data, save_dir=tmpdir, xg_flag=False, feature_csv_path=feature_csv)
        model = PwcSelector.load(f"{tmpdir}/model.joblib")

    # Count disagreements
    disagreements, total = count_disagreements(model, feature_csv, test_instances)
    print(f"Logic: {logic}")
    print(f"Disagreements: {disagreements}/{total} ({100*disagreements/total:.2f}%)")


if __name__ == "__main__":
    main()
