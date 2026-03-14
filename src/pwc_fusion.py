"""
Decision-level fusion for pairwise models.

This module implements fusion of two pairwise models (syntactic and description)
using standardized hyperplane distance scoring. See doc/fusion_method_selection.md
for detailed rationale.

Fusion method:
- fixed_alpha: Weighted average with fixed alpha (α * score_synt + (1-α) * score_desc)

Note: Classifiers must support get_standardized_score() (SVM decision function / std).
"""

import numpy as np
import joblib
import logging
from pathlib import Path

from .feature import extract_feature_from_csv


class PwcModelFusion:
    """
    Fuses two pairwise models using standardized hyperplane distance fusion.

    Combines predictions from models trained on different feature types by:
    1. Computing standardized decision scores (distance to hyperplane / std)
    2. Weighted linear combination: α * score_synt + (1-α) * score_desc
    3. Voting based on combined score (>0 → solver i wins)

    Classifiers must support get_standardized_score() method.
    """

    def __init__(
        self,
        model_synt,
        model_desc,
        alpha,
        feature_csv_synt,
        feature_csv_desc,
    ):
        """
        Initialize fusion model.

        Args:
            model_synt: PwcModel trained on syntactic features
            model_desc: PwcModel trained on description features
            alpha: Weight for syntactic model (α ∈ [0,1])
            feature_csv_synt: Path to syntactic features CSV
            feature_csv_desc: Path to description features CSV
        """
        self.model_synt = model_synt
        self.model_desc = model_desc
        self.alpha = alpha
        self.feature_csv_synt = feature_csv_synt
        self.feature_csv_desc = feature_csv_desc
        self.solver_size = model_synt.solver_size
        self.fusion_method = "fixed_alpha"

        # Validate models have same solver size
        if model_synt.solver_size != model_desc.solver_size:
            raise ValueError(
                f"Models have different solver sizes: {model_synt.solver_size} vs {model_desc.solver_size}"
            )

    def _get_score(self, model, feature, i, j):
        """
        Get standardized score indicating solver i vs solver j preference.

        Args:
            model: PwcModel (either syntactic or description)
            feature: Feature vector for instance
            i: First solver index
            j: Second solver index

        Returns:
            Standardized score: positive means solver i wins, negative means solver j wins
        """
        classifier = model.model_matrix[i, j]
        feature_reshaped = feature.reshape(1, -1)

        if hasattr(classifier, "get_standardized_score"):
            return classifier.get_standardized_score(feature_reshaped)[0]
        else:
            # Fallback for classifiers without get_standardized_score (e.g., DummyClassifier)
            # Convert prediction {0,1} to score {-1.0, 1.0}
            pred = classifier.predict(feature_reshaped)[0]
            return 1.0 if pred == 1 else -1.0

    def _get_rank_lst(self, feature_synt, feature_desc, random_seed=42):
        """
        Rank solvers using fused standardized scores.

        For each pairwise comparison [i,j]:
        1. Get standardized score from syntactic model
        2. Get standardized score from description model
        3. Combine: α * score_synt + (1-α) * score_desc
        4. Vote based on combined score (>0 → solver i wins)

        Args:
            feature_synt: Syntactic feature vector
            feature_desc: Description feature vector
            random_seed: Random seed for tie-breaking

        Returns:
            Array of solver indices sorted by votes (descending)
        """
        votes = np.zeros(self.solver_size, dtype=float)

        for i in range(self.solver_size):
            for j in range(i + 1, self.solver_size):
                # Get standardized scores from both models
                score_synt = self._get_score(
                    self.model_synt, feature_synt, i, j
                )
                score_desc = self._get_score(
                    self.model_desc, feature_desc, i, j
                )

                # Weighted fusion
                score_combined = self.alpha * score_synt + (1 - self.alpha) * score_desc

                # Vote based on combined score
                if score_combined > 0:
                    votes[i] += 1
                else:
                    votes[j] += 1

        # Rank by votes with random tie-breaking
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
        Select best solver for instance using fusion model.

        Args:
            instance_path: Path to instance
            random_seed: Random seed for tie-breaking

        Returns:
            Solver ID (index) of selected solver
        """
        # Extract features from both CSVs
        feature_synt = extract_feature_from_csv(instance_path, self.feature_csv_synt)
        feature_desc = extract_feature_from_csv(instance_path, self.feature_csv_desc)

        # Get ranking and return top solver
        ranked_indices = self._get_rank_lst(feature_synt, feature_desc, random_seed)
        return ranked_indices[0]

    def save(self, save_dir):
        """Save fusion model to directory."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = f"{save_dir}/model_fusion.joblib"
        joblib.dump(self, save_path)
        logging.info(f"Saved fusion model (α={self.alpha:.2f}) at {save_path}")

    @staticmethod
    def load(load_path):
        """Load fusion model from path."""
        return joblib.load(load_path)
