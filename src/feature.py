"""Feature extraction utilities for SMT instances."""

import csv
import numpy as np

# Global dictionary to cache features, keyed by CSV path
_feature_cache = {}


def _normalize_path(path: str) -> str:
    """Normalize path for matching (strip whitespace, normalize separators)."""
    return path.strip().replace("\\", "/")


def _load_feature_cache(csv_path: str):
    """Load features from CSV file into a dictionary with normalized paths."""
    global _feature_cache
    # Check if this specific CSV path is already cached
    if csv_path in _feature_cache:
        return _feature_cache[csv_path]

    # Load and cache features for this CSV path
    cache = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = _normalize_path(row["path"])
            # Extract all feature values (excluding 'path' column)
            features = [float(row[key]) for key in reader.fieldnames if key != "path"]
            cache[path] = np.array(features, dtype=np.float64)

    _feature_cache[csv_path] = cache
    return cache


def extract_feature_from_csv(instance_path: str, feature_csv_path: str):
    """
    Extract features for an instance from the features CSV file.

    Args:
        instance_path: Path to the instance
        feature_csv_path: Path to the features CSV file

    Returns:
        numpy array of features

    Raises:
        KeyError: If the instance path is not found in the features CSV
    """
    cache = _load_feature_cache(feature_csv_path)
    normalized_path = _normalize_path(instance_path)

    if normalized_path in cache:
        return cache[normalized_path]
    else:
        # If still not found, raise an error with helpful message
        raise KeyError(
            f"Instance path '{instance_path}' (normalized: '{normalized_path}') not found in features CSV '{feature_csv_path}'. "
        )
