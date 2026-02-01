"""
Decision-level fusion for pairwise models.

This module implements fusion of two pairwise models (syntactic and description)
using probability-based scoring. See doc/fusion_method_selection.md for
detailed rationale.

Fusion method:
- fixed_alpha: Weighted average with fixed alpha (α * prob_synt + (1-α) * prob_desc)

Note: Classifiers must support predict_proba (e.g., SVC with probability=True).
"""

import numpy as np
import joblib
import logging
from pathlib import Path

from .feature import extract_feature_from_csv


class PwcModelFusion:
    """
    Fuses two pairwise models using probability-based fusion.

    Combines predictions from models trained on different feature types by:
    1. Computing class probabilities from each model via predict_proba
    2. Weighted linear combination: α * prob_synt + (1-α) * prob_desc
    3. Voting based on combined probability (>0.5 → solver i wins)

    Note: No standardization is needed since probabilities are already in [0, 1].
    Classifiers must support predict_proba (e.g., SVC with probability=True).
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

    def _get_probability(self, model, feature, i, j):
        """
        Get probability that solver i wins over solver j.

        Args:
            model: PwcModel (either syntactic or description)
            feature: Feature vector for instance
            i: First solver index
            j: Second solver index

        Returns:
            Probability in [0, 1] that solver i wins (class 1)
        """
        classifier = model.model_matrix[i, j]
        feature_reshaped = feature.reshape(1, -1)

        if hasattr(classifier, "predict_proba"):
            probs = classifier.predict_proba(feature_reshaped)[0]
            if len(probs) == 1:
                # Only one class seen during training (degenerate case)
                # Return 1.0 if only class 1, 0.0 if only class 0
                return 1.0 if classifier.classes_[0] == 1 else 0.0
            else:
                # Find index of class 1 in classes_ array
                # classes_ could be [0, 1] or [1, 0] depending on training order
                class_1_idx = np.where(classifier.classes_ == 1)[0][0]
                return probs[class_1_idx]  # Probability of class 1 (solver i wins)
        else:
            # Fallback for classifiers without predict_proba
            # Convert prediction {0,1} to probability {0.0, 1.0}
            return float(classifier.predict(feature_reshaped)[0])

    def _get_rank_lst(self, feature_synt, feature_desc, random_seed=42):
        """
        Rank solvers using fused probability scores.

        For each pairwise comparison [i,j]:
        1. Get probability from syntactic model
        2. Get probability from description model
        3. Combine: α * prob_synt + (1-α) * prob_desc
        4. Vote based on combined probability (>0.5 → solver i wins)

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
                # Get probabilities from both models
                prob_synt = self._get_probability(
                    self.model_synt, feature_synt, i, j
                )
                prob_desc = self._get_probability(
                    self.model_desc, feature_desc, i, j
                )

                # Weighted fusion
                prob_combined = self.alpha * prob_synt + (1 - self.alpha) * prob_desc

                # Vote based on combined probability
                if prob_combined > 0.5:
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
