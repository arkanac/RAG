"""Validation of CLI parameters."""

MAX_CHUNK_SIZE_LIMIT = 2000


def validate_chunk_size(max_chunk_size: object) -> int:
    """Return max_chunk_size if it is an integer between 1 and 2000."""
    if not isinstance(max_chunk_size, int) or isinstance(max_chunk_size,
                                                         bool):
        raise ValueError(
            f"max_chunk_size must be an integer, got {max_chunk_size!r}")
    if not 1 <= max_chunk_size <= MAX_CHUNK_SIZE_LIMIT:
        raise ValueError(
            f"max_chunk_size must be between 1 and {MAX_CHUNK_SIZE_LIMIT},"
            f" got {max_chunk_size}")
    return max_chunk_size


def validate_k(k: object) -> int:
    """Return k if it is an integer of at least 1."""
    if not isinstance(k, int) or isinstance(k, bool):
        raise ValueError(f"k must be an integer, got {k!r}")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    return k
