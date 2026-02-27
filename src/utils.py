"""Shared utilities."""


def normalize_path(path: str) -> str:
    """Normalize path for matching (strip whitespace, normalize separators)."""
    return path.strip().replace("\\", "/")
