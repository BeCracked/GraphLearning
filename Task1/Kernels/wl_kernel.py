import networkx as nx
import numpy as np
from typing import Tuple


def _hash_to_colour(s: str, *, colour_seed: int = 42) -> Tuple[int, int, int]:
    """

    Parameters
    ----------
    s           - The string to hash into a colour
    colour_seed - Random seed for deterministic but varied colours

    Returns
    -------
    An RGB tuple representing a colour.
    """
    # Encode input into bytes
    input_bytes = bytes(s, "UTF-8")
    count = len(input_bytes)
    np.random.seed(colour_seed)
    r = g = b = 0
    for i in range(0, count, 3):
        # Add bytes round-robin style onto r, g and b modulo 256
        # To avoid out of bounds exceptions we are simply content with double sampling the first two values
        r += (input_bytes[i] * np.random.randint(0, 256)) % 256
        g += (input_bytes[i + 1 % count] * np.random.randint(0, 256)) % 256
        b += (input_bytes[i + 2 % count] * np.random.randint(0, 256)) % 256

    return r, g, b
