from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Iterable

import numpy as np
from scipy.spatial import Delaunay, KDTree

if TYPE_CHECKING:
    from ..lineage_tree import LineageTree


def idx3d(lT: LineageTree, t: int) -> tuple[KDTree, np.ndarray]:
    """Get a 3d kdtree for the dataset at time `t`.
    The  kdtree is stored in `lT._property_dict["kdtrees"][t]` and returned.
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

    if not lT.get_property("kdtree"):
        lT.add_property("kdtree", {}, time_property=True)

    if t not in lT.properties.kdtree:
        data_corres = {}
        data = []
        for i, C in enumerate(to_check_lT):
            data.append(tuple(lT.pos[C] * lT.spatial_resolution))
            data_corres[i] = C
        idx3d = KDTree(data)
        lT.properties.kdtree[t] = idx3d
    else:
        idx3d = lT.properties.kdtree[t]
    return idx3d, np.array(to_check_lT)


def gabriel_graph(
    lT: LineageTree, time: int | Iterable[int] | None = None
) -> dict[int, set[int]]:
    """Build the Gabriel graph of the given graph for time point `t`.
    The Garbiel graph is then stored in `lT._property_dict["gabriel_graph"]` and returned.

    .. warning:: the graph is not recomputed if already computed, even if the point cloud has changed

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    time : int or Iterable of int, optional
        time or iterable of times.
        If not given the gabriel graph will be calculated for all timepoints.

    Returns
    -------
    dict of int to set of int
        A dictionary that maps a node to the set of its neighbors
    """
    if not lT.get_property("gabriel_graph"):
        lT.add_property("gabriel_graph", {}, time_property=True)

    if time is None:
        time = lT.time_nodes.keys()
    elif not isinstance(time, Iterable):
        time = [time]

    for t in time:
        if lT.time_nodes[t] - lT.properties.gabriel_graph.keys():
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
            lT.properties.gabriel_graph.update(Gabriel_graph)

    return lT.properties.gabriel_graph


def neighbours_in_radius(
    lT: LineageTree,
    t_b: int | None = None,
    t_e: int | None = None,
    th: float = 50,
) -> dict[int, set]:
    """Computes the number of neighbours for nodes between `t_b` and `t_e`.
    The results is stored in `lT.neighbours` and returned.

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
    dict mapping int to set
        dictionary that maps a node id to its neighbours
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
        #### TODO OVERWRITE
    if not lT.get_property("neighbours_in_radius"):
        lT.add_property("neighbours_in_radius", neighbours, False)
    else:
        lT.properties.neighbours_in_radius.update(neighbours)
    return neighbours


def spatial_density(
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
    s_vol = 4 / 3.0 * np.pi * th**3
    spatial_density = {
        k: (len(v) + 1) / s_vol
        for k, v in lT.neighbours_in_radius(t_b, t_e, th).items()
    }
    if not lT.get_property("spatial_density"):
        lT.add_property("spatial_density", spatial_density, False)
    else:
        lT.properties.spatial_density.update(spatial_density)

    return spatial_density


def k_nearest_neighbours(lT: LineageTree, k: int = 10) -> dict[int, set[int]]:
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
    dict mapping int to set of float
        dictionary that maps
        a node id to the distances of its `k` nearest neighbors
    """
    kn_graph = {}
    kn_distances = {}
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
            kn_graph.update(out)
            kn_distances.update(out_distances)
    if not lT.get_property("kn_graph"):
        lT.add_property("kn_graph", kn_graph, False)
    else:
        lT.properties.kn_graph.update(kn_graph)
    if not lT.get_property("kn_distances"):
        lT.add_property("kn_distances", kn_distances, False)
    else:
        lT.properties.kn_distances.update(kn_distances)
    return kn_graph, kn_distances


def spatial_edges(lT: LineageTree, th: int = 50) -> dict[int, set[int]]:
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
    th_edges = {}
    for t in set(lT._time.values()):
        nodes = lT.time_nodes[t]
        idx3d, nodes = lT.idx3d(t)
        neighbs = idx3d.query_ball_tree(idx3d, th)
        out = dict(zip(nodes, [set(nodes[ni]) for ni in neighbs], strict=True))
        th_edges.update({k: v.difference([k]) for k, v in out.items()})
    if not lT.get_property("th_edges"):
        lT.add_property("th_edges", th_edges, False)
    else:
        lT.properties.th_edges.update(th_edges)
    return th_edges
