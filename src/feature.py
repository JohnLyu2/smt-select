"""Feature extraction utilities for SMT instances."""

import csv
import numpy as np

from src.utils import normalize_path

# Global dictionary to cache features, keyed by CSV path
_feature_cache = {}


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
            path = normalize_path(row["path"])
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
    normalized_path = normalize_path(instance_path)

    if normalized_path in cache:
        return cache[normalized_path]
    else:
        # If still not found, raise an error with helpful message
        raise KeyError(
            f"Instance path '{instance_path}' (normalized: '{normalized_path}') not found in features CSV '{feature_csv_path}'. "
        )


def extract_feature_from_csvs_concat(instance_path: str, feature_csv_paths: list[str]):
    """
    Extract features for an instance from multiple feature CSV files and concatenate them.

    Args:
        instance_path: Path to the instance
        feature_csv_paths: List of paths to the feature CSV files

    Returns:
        numpy array of concatenated features from all CSVs

    Raises:
        KeyError: If the instance path is not found in any of the features CSVs
    """
    features_list = []
    normalized_path = normalize_path(instance_path)
    missing_paths = []

    for feature_csv_path in feature_csv_paths:
        cache = _load_feature_cache(feature_csv_path)
        if normalized_path in cache:
            features_list.append(cache[normalized_path])
        else:
            missing_paths.append(feature_csv_path)

    if missing_paths:
        raise KeyError(
            f"Instance path '{instance_path}' (normalized: '{normalized_path}') not found in feature CSVs: {missing_paths}"
        )

    if not features_list:
        raise KeyError(
            f"Instance path '{instance_path}' (normalized: '{normalized_path}') not found in any of the provided feature CSVs"
        )

    return np.concatenate(features_list)


def validate_feature_coverage(
    instance_paths: set[str], feature_csv_path: str | list[str]
) -> tuple[list[str], dict[str, list[str]]]:
    """
    Validate that all instances have corresponding features in the feature CSV(s).

    Args:
        instance_paths: Set of instance paths to validate
        feature_csv_path: Path to features CSV file, or list of paths to multiple CSV files

    Returns:
        Tuple of (missing_instances, instance_missing_in_csvs) where:
        - missing_instances: List of instance paths that are missing in ALL feature CSVs
        - instance_missing_in_csvs: Dict mapping instance paths to list of CSVs where they're missing

    Raises:
        ValueError: If any instances are missing features
    """
    missing_instances = []
    instance_missing_in_csvs = {}

    if isinstance(feature_csv_path, list):
        # Multiple CSVs: instance must be in ALL CSVs
        csv_paths = feature_csv_path
    else:
        # Single CSV
        csv_paths = [feature_csv_path]

    # Load all caches
    caches = {}
    for csv_path in csv_paths:
        caches[csv_path] = _load_feature_cache(csv_path)

    # Check each instance
    for instance_path in instance_paths:
        normalized_path = normalize_path(instance_path)
        missing_in = []

        for csv_path in csv_paths:
            if normalized_path not in caches[csv_path]:
                missing_in.append(csv_path)

        if missing_in:
            instance_missing_in_csvs[instance_path] = missing_in
            # If missing in all CSVs, add to missing_instances
            if len(missing_in) == len(csv_paths):
                missing_instances.append(instance_path)

    return missing_instances, instance_missing_in_csvs
