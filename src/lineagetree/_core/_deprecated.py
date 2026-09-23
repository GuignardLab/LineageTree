"""Method names of lineagetree 3.2, kept as deprecated aliases.

The spatial and DTW methods were renamed in 3.3.0 without a transition, so
code written for 3.2 broke on upgrade. Each old name is kept here as an alias
that warns and forwards to its current name.
"""

from __future__ import annotations

import inspect
import warnings
from typing import TYPE_CHECKING

from ..measure.dynamic_time_warping import dtw
from ..measure.spatial import (
    gabriel_graph,
    idx3d,
    k_nearest_neighbours,
    neighbours_in_radius,
    spatial_density,
    spatial_edges,
)

if TYPE_CHECKING:
    from ..lineage_tree import LineageTree


def _warn(old: str, new: str) -> None:
    """Warn that ``old`` is the 3.2 name of ``new``.

    Parameters
    ----------
    old : str
        Name used by the caller.
    new : str
        Name to use instead.
    """
    warnings.warn(
        f"`LineageTree.{old}` was renamed `LineageTree.{new}` in 3.3, "
        "the old name will be removed in a future release.",
        DeprecationWarning,
        stacklevel=4,  # caller -> bound method -> alias -> _warn
    )


def _deprecated_alias(func, old: str):
    """Build a deprecated alias of ``func`` named ``old``.

    Parameters
    ----------
    func : callable
        The function the alias forwards to.
    old : str
        The 3.2 name.

    Returns
    -------
    callable
        A function with the signature of ``func`` that warns and forwards.
    """

    def alias(lT, *args, **kwargs):
        _warn(old, func.__name__)
        return func(lT, *args, **kwargs)

    alias.__name__ = alias.__qualname__ = old
    alias.__signature__ = inspect.signature(func)
    alias.__doc__ = (
        f"Deprecated alias of ``{func.__name__}``.\n\n"
        f".. deprecated:: 3.3\n    Use ``{func.__name__}`` instead."
    )
    return alias


calculate_dtw = _deprecated_alias(dtw, "calculate_dtw")
compute_k_nearest_neighbours = _deprecated_alias(
    k_nearest_neighbours, "compute_k_nearest_neighbours"
)
compute_spatial_density = _deprecated_alias(
    spatial_density, "compute_spatial_density"
)
compute_spatial_edges = _deprecated_alias(
    spatial_edges, "compute_spatial_edges"
)
get_gabriel_graph = _deprecated_alias(gabriel_graph, "get_gabriel_graph")
get_idx3d = _deprecated_alias(idx3d, "get_idx3d")


def compute_neighbours_in_radius(
    lT: LineageTree,
    t_b: int | None = None,
    t_e: int | None = None,
    th: float = 50,
) -> dict[int, int]:
    """Count the neighbours of each node within a radius, as 3.2 did.

    .. deprecated:: 3.3
        Use ``neighbours_in_radius``, which returns the neighbours
        themselves rather than how many there are.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    t_b : int, optional
        First time point to consider.
    t_e : int, optional
        Last time point to consider.
    th : float, default=50
        Radius within which nodes count as neighbours.

    Returns
    -------
    dict of {int: int}
        Dictionary that maps a node id to its number of neighbours.
    """
    _warn("compute_neighbours_in_radius", "neighbours_in_radius")
    return {
        node: len(neighbours)
        for node, neighbours in neighbours_in_radius(lT, t_b, t_e, th).items()
    }
