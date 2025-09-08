# repository/utils.py

import re

def safe_filename(name: str, max_len: int = 100) -> str:
    """
    Sanitize a string to be safe for use as a filename or folder.

    - Removes unsafe characters (e.g., slashes, colons)
    - Replaces spaces with underscores
    - Truncates to max_len characters

    Args:
        name: Raw name string (e.g. circuit name)
        max_len: Max length of the final filename

    Returns:
        Safe filename string
    """
    name = name.strip()
    name = name.replace(" ", "_")
    name = re.sub(r"[^a-zA-Z0-9_.-]", "", name)
    return name[:max_len]

def permutation_to_str(perm: list[tuple[int, int]]) -> str:
    return "_".join(f"{p}.{l}" for p, l in perm)
