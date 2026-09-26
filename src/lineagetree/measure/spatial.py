"""Spatial neighbourhoods of the nodes at each time point.

Neighbourhoods are computed among the nodes of a single time point.
Positions are multiplied by ``lT.spatial_resolution`` first, so distances are
in physical units.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Iterable

import numpy as np
from scipy.spatial import Delaunay, KDTree

if TYPE_CHECKING:
    from ..lineage_tree import LineageTree


def idx3d(lT: LineageTree, t: int) -> tuple[KDTree, np.ndarray]:
    """Get a KDTree of the node positions at time ``t``.

    Positions are multiplied by ``lT.spatial_resolution``. The KDTree is
    cached in ``lT.kdtrees[t]``.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    t : int
        Time point.

    Returns
    -------
    scipy.spatial.KDTree
        The KDTree of the positions of the nodes at time ``t``.
    numpy.ndarray
        The node ids, in KDTree order: index ``i`` returned by a query on
        the KDTree is the node ``ids[i]``.

    Warnings
    --------
    The cached KDTree is not updated when positions change.
    """
    to_check_lT = list(lT.time_nodes[t])

    if not hasattr(lT, "kdtrees"):
        lT.kdtrees = {}

    if t not in lT.kdtrees:
        data_corres = {}
        data = []
        for i, C in enumerate(to_check_lT):
            data.append(tuple(lT.pos[C] * lT.spatial_resolution))
            data_corres[i] = C
        idx3d = KDTree(data)
        lT.kdtrees[t] = idx3d
    else:
        idx3d = lT.kdtrees[t]
    return idx3d, np.array(to_check_lT)


def gabriel_graph(
    lT: LineageTree, time: int | Iterable[int] | None = None
) -> dict[int, set[int]]:
    """Build the Gabriel graph of the nodes at the given time point(s).

    Two nodes are neighbours in the Gabriel graph when no other node lies
    inside the sphere whose diameter is the segment between them. The graph
    is stored in ``lT.Gabriel_graph``.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    time : int or Iterable of int, optional
        Time point(s) to compute the graph for. If not given, all time points
        are computed.

    Returns
    -------
    dict of {int: set of int}
        ``lT.Gabriel_graph``: maps each node to the set of its neighbours. It
        holds every time point computed so far, not only ``time``.

    Warnings
    --------
    Time points already in ``lT.Gabriel_graph`` are not recomputed, even if
    the positions have changed.
    """
    if not hasattr(lT, "Gabriel_graph"):
        lT.Gabriel_graph = {}

    if time is None:
        time = lT.time_nodes.keys()
    elif not isinstance(time, Iterable):
        time = [time]

    for t in time:
        if lT.time_nodes[t] - lT.Gabriel_graph.keys():
            nodes = lT.time_nodes[t]

            data_corres = {}
            data = []
            for i, C in enumerate(nodes):
                data.append(lT.pos[C] * lT.spatial_resolution)
                data_corres[i] = C

            delaunay_graph = {}

            # The delaunay triangulation is only usefult to compute
            # when the number of points is higher than the spatial dimension + 1
            if len(data[0]) + 1 < len(data):
                tmp = Delaunay(data)
                for N in tmp.simplices:
                    for e1, e2 in combinations(np.sort(N), 2):
                        delaunay_graph.setdefault(e1, set()).add(e2)
                        delaunay_graph.setdefault(e2, set()).add(e1)
            # When there are fewer nodes than the number of dimensions + 2
            # The Delaunay is the complete graph
            else:
                for e1, e2 in combinations(data_corres, 2):
                    delaunay_graph.setdefault(e1, set()).add(e2)
                    delaunay_graph.setdefault(e2, set()).add(e1)

            Gabriel_graph = {}

            for e1, neighbs in delaunay_graph.items():
                for ni in neighbs:
                    if not any(
                        np.linalg.norm((data[ni] + data[e1]) / 2 - data[i])
                        < np.linalg.norm(data[ni] - data[e1]) / 2
                        for i in delaunay_graph[e1].intersection(
                            delaunay_graph[ni]
                        )
                    ):
                        Gabriel_graph.setdefault(data_corres[e1], set()).add(
                            data_corres[ni]
                        )
                        Gabriel_graph.setdefault(data_corres[ni], set()).add(
                            data_corres[e1]
                        )
            lT.Gabriel_graph.update(Gabriel_graph)

    return lT.Gabriel_graph


def neighbours_in_radius(
    lT: LineageTree,
    t_b: int | None = None,
    t_e: int | None = None,
    th: float = 50,
) -> dict[int, set[int]]:
    """Find the neighbours of each node within a radius.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    t_b : int, optional
        First time point to consider. Defaults to the first time point of
        the dataset.
    t_e : int, optional
        End of the time range, excluded. Defaults to the last time point of
        the dataset, which is therefore left out.
    th : float, default=50
        Radius of the neighbourhood.

    Returns
    -------
    dict of {int: set of int}
        Maps each node in ``[t_b, t_e)`` to the other nodes of its time point
        within distance ``th``.
    """
    neighbours = {}
    if t_b is None:
        t_b = lT.t_b
    if t_e is None:
        t_e = lT.t_e
    time_range = set(range(t_b, t_e)).intersection(lT._time.values())
    for t in time_range:
        idx3d, nodes = lT.idx3d(t)
        idx = idx3d.query_ball_tree(idx3d, th)
        neighbours.update(
            {
                node: set(nodes[nb_idx]) - {node}
                for node, nb_idx in zip(nodes, idx)
            }
        )
    return neighbours


def spatial_density(
    lT: LineageTree,
    t_b: int | None = None,
    t_e: int | None = None,
    th: float = 50,
) -> dict[int, float]:
    """Compute the spatial density around each node.

    The density of a node is the number of nodes within distance ``th``
    (itself included) divided by the volume of the sphere of radius ``th``.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    t_b : int, optional
        First time point to consider. Defaults to the first time point of
        the dataset.
    t_e : int, optional
        End of the time range, excluded. Defaults to the last time point of
        the dataset, which is therefore left out.
    th : float, default=50
        Radius of the neighbourhood.

    Returns
    -------
    dict of {int: float}
        Maps each node in ``[t_b, t_e)`` to its spatial density.
    """
    s_vol = 4 / 3.0 * np.pi * th**3
    spatial_density = {
        k: (len(v) + 1) / s_vol
        for k, v in lT.neighbours_in_radius(t_b, t_e, th).items()
    }
    return spatial_density


def k_nearest_neighbours(
    lT: LineageTree, k: int = 10
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Find the k nearest neighbours of every node.

    Neighbours are searched among the other nodes of the same time point. A
    node alone at its time point gets no entry, and a time point with fewer
    than ``k + 1`` nodes gives fewer than ``k`` neighbours. The results are
    stored in ``lT.kn_graph`` and ``lT.kn_distances``.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    k : int, default=10
        Number of nearest neighbours.

    Returns
    -------
    dict of {int: numpy.ndarray}
        Maps each node to its nearest neighbours, closest first.
    dict of {int: numpy.ndarray}
        Maps each node to the distances to these neighbours.
    """
    lT.kn_graph = {}
    lT.kn_distances = {}
    k = k + 1
    for t, nodes in lT.time_nodes.items():
        if 1 < len(nodes):
            use_k = k if k < len(nodes) else len(nodes)
            idx3d, nodes = lT.idx3d(t)
            distances, neighbs = idx3d.query(idx3d.data, use_k)
            out = dict(
                zip(
                    nodes,
                    nodes[neighbs[:, 1:]],
                    strict=True,
                )
            )
            out_distances = dict(
                zip(
                    nodes,
                    distances[:, 1:],
                    strict=True,
                )
            )
            lT.kn_graph.update(out)
            lT.kn_distances.update(out_distances)
    return lT.kn_graph, lT.kn_distances


def spatial_edges(lT: LineageTree, th: float = 50) -> dict[int, set[int]]:
    """Find the neighbours of every node within a distance, at all times.

    Unlike [`neighbours_in_radius`][lineagetree.LineageTree.neighbours_in_radius],
    all time points are used. The result is stored in ``lT.th_edges``.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    th : float, default=50
        Distance within which two nodes are considered neighbours.

    Returns
    -------
    dict of {int: set of int}
        Maps each node to the other nodes of its time point within distance
        ``th``.
    """
    lT.th_edges = {}
    for t in set(lT._time.values()):
        nodes = lT.time_nodes[t]
        idx3d, nodes = lT.idx3d(t)
        neighbs = idx3d.query_ball_tree(idx3d, th)
        out = dict(zip(nodes, [set(nodes[ni]) for ni in neighbs], strict=True))
        lT.th_edges.update({k: v.difference([k]) for k, v in out.items()})
    return lT.th_edges
