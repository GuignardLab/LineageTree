from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import Delaunay, KDTree

if TYPE_CHECKING:
    from ..lineage_tree import LineageTree


def get_idx3d(lT: LineageTree, t: int) -> tuple[KDTree, np.ndarray]:
    """Get a 3d kdtree for the dataset at time `t`.
    The  kdtree is stored in `lT.kdtrees[t]` and returned.
    The correspondancy list is also returned.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    t : int
        time

    Returns
    -------
    KDTree
        The KDTree corresponding to the lineage tree at time `t`
    np.ndarray
        The correspondancy list in the KDTree.
        If the query in the kdtree gives you the value `i`,
        then it corresponds to the id in the tree `to_check_lT[i]`
    """
    to_check_lT = list(lT.time_nodes[t])

    if not hasattr(lT, "kdtrees"):
        lT.kdtrees = {}

    if t not in lT.kdtrees:
        data_corres = {}
        data = []
        for i, C in enumerate(to_check_lT):
            data.append(tuple(lT.pos[C]))
            data_corres[i] = C
        idx3d = KDTree(data)
        lT.kdtrees[t] = idx3d
    else:
        idx3d = lT.kdtrees[t]
    return idx3d, np.array(to_check_lT)


def get_gabriel_graph(lT: LineageTree, t: int) -> dict[int, set[int]]:
    """Build the Gabriel graph of the given graph for time point `t`.
    The Garbiel graph is then stored in `lT.Gabriel_graph` and returned.

    .. warning:: the graph is not recomputed if already computed, even if the point cloud has changed

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    t : int
        time

    Returns
    -------
    dict of int to set of int
        A dictionary that maps a node to the set of its neighbors
    """
    if not hasattr(lT, "Gabriel_graph"):
        lT.Gabriel_graph = {}

    if t not in lT.Gabriel_graph:
        _, nodes = lT.get_idx3d(t)

        data_corres = {}
        data = []
        for i, C in enumerate(nodes):
            data.append(lT.pos[C])
            data_corres[i] = C

        tmp = Delaunay(data)

        delaunay_graph = {}

        for N in tmp.simplices:
            for e1, e2 in combinations(np.sort(N), 2):
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

        lT.Gabriel_graph[t] = Gabriel_graph

    return lT.Gabriel_graph[t]


def compute_spatial_density(
    lT: LineageTree,
    t_b: int | None = None,
    t_e: int | None = None,
    th: float = 50,
) -> dict[int, float]:
    """Computes the spatial density of nodes between `t_b` and `t_e`.
    The results is stored in `lT.spatial_density` and returned.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    t_b : int, optional
        starting time to look at, default first time point
    t_e : int, optional
        ending time to look at, default last time point
    th : float, default=50
        size of the neighbourhood

    Returns
    -------
    dict mapping int to float
        dictionary that maps a node id to its spatial density
    """
    if not hasattr(lT, "spatial_density"):
        lT.spatial_density = {}
    s_vol = 4 / 3.0 * np.pi * th**3
    if t_b is None:
        t_b = lT.t_b
    if t_e is None:
        t_e = lT.t_e
    time_range = set(range(t_b, t_e)).intersection(lT._time.values())
    for t in time_range:
        idx3d, nodes = lT.get_idx3d(t)
        nb_ni = [
            (len(ni) - 1) / s_vol for ni in idx3d.query_ball_tree(idx3d, th)
        ]
        lT.spatial_density.update(dict(zip(nodes, nb_ni, strict=True)))
    return lT.spatial_density


def compute_k_nearest_neighbours(
    lT: LineageTree, k: int = 10
) -> dict[int, set[int]]:
    """Computes the k-nearest neighbors
    Writes the output in the attribute `kn_graph`
    and returns it.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    k : float
        number of nearest neighours

    Returns
    -------
    dict mapping int to set of int
        dictionary that maps
        a node id to its `k` nearest neighbors
    """
    lT.kn_graph = {}
    for t in set(lT._time.values()):
        nodes = lT.time_nodes[t]
        if 1 < len(nodes):
            use_k = k if k < len(nodes) else len(nodes)
            idx3d, nodes = lT.get_idx3d(t)
            pos = [lT.pos[c] for c in nodes]
            _, neighbs = idx3d.query(pos, use_k)
            out = dict(
                zip(
                    nodes,
                    map(set, nodes[neighbs]),
                    strict=True,
                )
            )
            lT.kn_graph.update(out)
        else:
            n = nodes.pop
            lT.kn_graph.update({n: {n}})
    return lT.kn_graph


def compute_spatial_edges(
    lT: LineageTree, th: int = 50
) -> dict[int, set[int]]:
    """Computes the neighbors at a distance `th`
    Writes the output in the attribute `th_edge`
    and returns it.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    th : float, default=50
        distance to consider neighbors

    Returns
    -------
    dict mapping int to set of int
        dictionary that maps a node id to its neighbors at a distance `th`
    """
    lT.th_edges = {}
    for t in set(lT._time.values()):
        nodes = lT.time_nodes[t]
        idx3d, nodes = lT.get_idx3d(t)
        neighbs = idx3d.query_ball_tree(idx3d, th)
        out = dict(zip(nodes, [set(nodes[ni]) for ni in neighbs], strict=True))
        lT.th_edges.update({k: v.difference([k]) for k, v in out.items()})
    return lT.th_edges
