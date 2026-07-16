import numpy as np


def delta_normalized_difference(node1, node2):
    """
    Delta function for unordered edit distance.

    Returns the absolute difference between `v1` and `v2` normalized by `max(v1, v2)`.
    Addition and deletion both cost 1.
    """
    if node1 is None or node2 is None:
        return 1
    else:
        return np.abs(node1 - node2) / np.max([node1, node2])


def delta_nd_norm(node1, node2):
    """
    Delta function for unordered edit distance.

    Returns the norm between `np.linalg.norm(v1 - v2)`.
    Addition and deletion both cost
    `np.linalg.norm(v1)` or `np.linalg.norm(v2)`.
    """
    if node1 is None:
        return np.linalg.norm(node2)
    elif node2 is None:
        return np.linalg.norm(node1)
    else:
        return np.linalg.norm(node1 - node2)


def delta_difference(node1, node2):
    """
    Delta function for unordered edit distance.

    Returns the absolute difference between `v1` and `v2`.
    Addition and deletion both cost `v1` or `v2` accordingly.
    """
    if node1 is None:
        return node2
    elif node2 is None:
        return node1
    else:
        return np.abs(node1 - node2)


def delta_binary(node1, node2):
    """
    Delta function for unordered edit distance.

    Matching costs 0, addition and deletion cost 1
    """
    if node1 is None or node2 is None:
        return 1
    else:
        return 0
