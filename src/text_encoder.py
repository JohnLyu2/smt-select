"""Text embedding function for SMT descriptions."""

import numpy as np
from sentence_transformers import SentenceTransformer

# Global model cache
_model_cache = {}


def get_embedding_model(
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
) -> SentenceTransformer:
    """
    Get or load an embedding model (cached for efficiency).

    Args:
        model_name: Name of the sentence transformer model to use

    Returns:
        SentenceTransformer model instance
    """
    global _model_cache
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def encode_text(
    text: str | list[str],
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    normalize: bool = False,
    batch_size: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Transform text description(s) into embedding vector(s).

    Args:
        text: Single text string or list of text strings to encode
        model_name: Name of the sentence transformer model to use
        normalize: Whether to normalize embeddings to unit length
        batch_size: Batch size for processing multiple texts
        show_progress: Whether to show progress bar when encoding multiple texts

    Returns:
        numpy array of embeddings:
        - For single text: shape (embedding_dim,)
        - For multiple texts: shape (num_texts, embedding_dim)

    Examples:
        >>> # Single text
        >>> embedding = encode_text("This is a description")
        >>> print(embedding.shape)  # (768,)

        >>> # Multiple texts
        >>> embeddings = encode_text(["Text 1", "Text 2", "Text 3"])
        >>> print(embeddings.shape)  # (3, 768)
    """
    model = get_embedding_model(model_name)

    # Handle single text vs list of texts
    is_single = isinstance(text, str)
    texts = [text] if is_single else text

    # Encode texts
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
        batch_size=batch_size,
        normalize_embeddings=normalize,
    )

    # Return single embedding for single text, array for multiple
    if is_single:
        return embeddings[0]
    return embeddings
