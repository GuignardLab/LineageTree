from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Iterable

import numpy as np
from scipy.spatial import Delaunay, KDTree

if TYPE_CHECKING:
    from ..lineage_tree import LineageTree


def idx3d(lT: LineageTree, t: int) -> tuple[KDTree, np.ndarray]:
    """Get a 3D KDTree for the dataset at time ``t``.

    The KDTree is stored in ``lT.kdtrees[t]`` and returned together with the
    correspondence list.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    t : int
        Time point.

    Returns
    -------
    KDTree
        The KDTree corresponding to the lineage tree at time ``t``.
    numpy.ndarray
        The correspondence list in the KDTree. If a query in the KDTree returns
        the value ``i``, it corresponds to the id ``to_check_lT[i]`` in the
        tree.
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
    """Build the Gabriel graph of the dataset for the given time point(s).

    The Gabriel graph is stored in ``lT.Gabriel_graph`` and returned.

    .. warning::
        The graph is not recomputed if already computed, even if the point
        cloud has changed.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    time : int or Iterable of int, optional
        Time or iterable of times. If not given, the Gabriel graph is
        calculated for all time points.

    Returns
    -------
    dict of {int: set of int}
        A dictionary that maps a node to the set of its neighbours.
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
                data.append(lT.pos[C])
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
) -> dict[int, float]:
    """Compute the neighbours within radius ``th`` for nodes in a time range.

    The result is stored in ``lT.neighbours`` and returned.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    t_b : int, optional
        Starting time to look at. Defaults to the first time point.
    t_e : int, optional
        Ending time to look at. Defaults to the last time point.
    th : float, default=50
        Size of the neighbourhood.

    Returns
    -------
    dict of {int: set of int}
        Dictionary that maps a node id to the set of its neighbours within
        radius ``th``.
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
    """Compute the spatial density of nodes in a time range.

    The result is stored in ``lT.spatial_density`` and returned.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    t_b : int, optional
        Starting time to look at. Defaults to the first time point.
    t_e : int, optional
        Ending time to look at. Defaults to the last time point.
    th : float, default=50
        Size of the neighbourhood.

    Returns
    -------
    dict of {int: float}
        Dictionary that maps a node id to its spatial density.
    """
    s_vol = 4 / 3.0 * np.pi * th**3
    spatial_density = {
        k: (len(v) + 1) / s_vol
        for k, v in lT.neighbours_in_radius(t_b, t_e, th).items()
    }
    return spatial_density


def k_nearest_neighbours(lT: LineageTree, k: int = 10) -> dict[int, set[int]]:
    """Compute the k-nearest neighbours of every node.

    The output is written to the attribute ``kn_graph`` and returned.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    k : int, default=10
        Number of nearest neighbours.

    Returns
    -------
    dict of {int: set of int}
        Dictionary that maps a node id to its ``k`` nearest neighbours.
    dict of {int: set of float}
        Dictionary that maps a node id to the distances of its ``k`` nearest
        neighbours.
    """
    lT.kn_graph = {}
    lT.kn_distances = {}
    k = k + 1
    for t, nodes in lT.time_nodes.items():
        if 1 < len(nodes):
            use_k = k if k < len(nodes) else len(nodes)
            idx3d, nodes = lT.idx3d(t)
            pos = [lT.pos[c] for c in nodes]
            distances, neighbs = idx3d.query(pos, use_k)
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


def spatial_edges(lT: LineageTree, th: int = 50) -> dict[int, set[int]]:
    """Compute the neighbours at a distance ``th`` of every node.

    The output is written to the attribute ``th_edges`` and returned.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    th : float, default=50
        Distance below which two nodes are considered neighbours.

    Returns
    -------
    dict of {int: set of int}
        Dictionary that maps a node id to its neighbours within distance
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
