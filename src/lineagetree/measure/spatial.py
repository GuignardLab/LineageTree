from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Iterable

import numpy as np
from scipy.spatial import Delaunay, KDTree
from .._core._modifier import anchored_gaussian_smooth

if TYPE_CHECKING:
    from ..lineage_tree import LineageTree


def idx3d(lT: LineageTree, t: int) -> tuple[KDTree, np.ndarray]:
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
    """Build the Gabriel graph of the given graph for time point `t`.
    The Garbiel graph is then stored in `lT.Gabriel_graph` and returned.

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
    if not hasattr(lT, "Gabriel_graph"):
        lT.Gabriel_graph = {}

    if time is None:
        time = lT.time_nodes.keys()
    elif not isinstance(time, Iterable):
        time = [time]

    for t in time:
        if lT.time_nodes[t] - lT.Gabriel_graph.keys():
            nodes = np.fromiter(list(lT.time_nodes[t]), dtype=int)

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
    dict mapping int to float
        dictionary that maps a node id to its spatial density
    """
    neighbours = {}
    if t_b is None:
        t_b = lT.t_b
    if t_e is None:
        t_e = lT.t_e
    time_range = set(range(t_b, t_e)).intersection(lT._time.values())
    for t in time_range:
        idx3d, nodes = lT.idx3d(t)
        nb_ni = [(len(ni) - 1) for ni in idx3d.query_ball_tree(idx3d, th)]
        neighbours.update(dict(zip(nodes, nb_ni, strict=True)))
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
        k: (v + 1) / s_vol
        for k, v in lT.neighbours_in_radius(t_b, t_e, th).items()
    }
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
        idx3d, nodes = lT.idx3d(t)
        neighbs = idx3d.query_ball_tree(idx3d, th)
        out = dict(zip(nodes, [set(nodes[ni]) for ni in neighbs], strict=True))
        lT.th_edges.update({k: v.difference([k]) for k, v in out.items()})
    return lT.th_edges


def _track_length(chain: np.ndarray, sigma: float = 1.5) -> float:
    """Computes the anchored gaussian smooth of a chain and returns the distance travelled for a given chain.

    Parameters
    ----------
    chain : np.ndarray
        3D numpy array with the positions of all nodes in a chain.
    sigma : int
        Standard deviation of the Gaussian kernel used for smoothing.
        Higher values produce stronger smoothing. Default is 1.5.
    Returns
    -------
    float
        The sum of all the distances from the start to the end of the chain, after the data has been smoothed or not.
    """

    chain = np.array(chain)
    smoothed_chain = np.zeros_like(chain, dtype=float)
    if sigma != 0:
        smoothed_chain[:, 0] = anchored_gaussian_smooth(chain[:, 0], sigma)
        smoothed_chain[:, 1] = anchored_gaussian_smooth(chain[:, 1], sigma)
        smoothed_chain[:, 2] = anchored_gaussian_smooth(chain[:, 2], sigma)
    else:
        smoothed_chain = chain  # Not smoothed
    distance_travelled = np.linalg.norm(
        smoothed_chain[:-1] - smoothed_chain[1:], axis=0
    )
    return sum(distance_travelled)


def get_track_length(lT: LineageTree, sigma: float = 1.5) -> dict[int, float]:
    """Returns the distance travelled for each timepoint.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree object.
    sigma : float, optional
        Standard deviation of the Gaussian kernel used for smoothing.
        Higher values produce stronger smoothing, by default is 1.5.

    Returns
    -------
    dict[int, float]
        dictionary that maps a node id to its track length
    """
    track_length = {}
    for track in lT.all_chains:
        data = np.array([lT.pos[c] for c in track])
        dist = _track_length(data, sigma)
        track_length.update({node: dist for node in track})
    lT.track_length = track_length
    return lT.track_length


def get_duration(lT: LineageTree) -> dict[int, float]:
    """The duration of each chain.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree object.

    Returns
    -------
    dict[int,float]
       dictionary that maps a node id to the temporal duration of its chain.
    """
    lT.duration = {
        node: len(lT.get_chain_of_node(node)) * lT.time_resolution
        for node in lT.nodes
    }
    return lT.duration


def get_max_displacement(lT: LineageTree) -> dict[int, float]:
    """Maximal Euclidean distance of any position on the subtrack from the first position.
    Each node in a given chain has the same max displacement.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree object.

    Returns
    -------
    dict[int, float]
        a dictionary that maps each node to the max displacement of its chain.
    """

    lT.max_displacement = {}
    for chain in lT.all_chains:
        root = lT.get_ancestor_at_t(chain[0])
        if root not in lT.nodes:
            continue
        root_pos = lT.pos[root]
        positions = np.array([lT.pos[c] for c in chain])
        displacements = np.cumsum(
            np.linalg.norm((positions - root_pos), axis=1)
        )
        lT.max_displacement.update(
            {node: disp for node, disp in zip(chain, displacements)}
        )
    return lT.max_displacement


def get_speed(lT: LineageTree, sigma: float) -> dict[int, float]:
    """The speed of each node.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree object.
    sigma : float
        The smoothing factor

    Returns
    -------
    dict[int, float]
        dictionary that maps each node to its speed ( distance travelled/time_resolution)
    """

    track_length = get_track_length(lT, sigma)
    lT.speed = {
        node: dist / lT.time_resolution for node, dist in track_length.items()
    }
    return lT.speed


def get_displacement(lT: LineageTree) -> dict[int, float]:
    """Displacement between a the start and the end of each chain.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree object.

    Returns
    -------
    dict[int, float]
        dictionary that maps each node to the displacement of its chain (start to end)
    """
    displacement = {}
    for chain in lT.all_chains:
        displacement.update(
            {
                node: np.linalg.norm(lT.pos[chain[0]] - lT.pos[chain[-1]])
                for node in chain
            }
        )
    lT.displacement = displacement
    return lT.displacement


def get_velocity(lT: LineageTree) -> dict[int, float]:
    """The velocity of each node.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree object.

    Returns
    -------
    dict[int, float]
        dictionary that maps each node ti its velocity
    """
    disp = get_displacement(lT)
    lT.velocity = {
        node: dist / lT.time_resolution for node, dist in disp.items()
    }
    return lT.velocity


def get_mean_squared_displacement(
    lT: LineageTree,
) -> dict[int, float]:
    """The Mean Squared Euclidean distance between the track starting and endpoints

    Parameters
    ----------
    lT : LineageTree
        The LineageTree object.

    Returns
    -------
    dict[int, float]
        dictionary that maps each node to the msd of its respective chain.
    """
    lT.msd = {}
    for chain in lT.all_chains:
        positions = np.array([lT.pos[c] for c in chain])
        MSD = np.cumsum(
            np.linalg.norm((positions - positions[0]) ** 2, axis=1)
        ) / np.arange(1, len(chain) + 1)
        lT.msd.update({node: msd for node, msd in zip(chain, MSD)})
    return lT.msd


def get_displacement_ratio(lT: LineageTree) -> dict[int, float]:
    """The displacement ratio, displacement/max_displacement, displacement of each chain divided by
    the displacement from the root node.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree object.

    Returns
    -------
    dict[int, float]
        dictionary that maps each node to its displacement ratio.
    """
    disp = get_displacement(lT)
    max_disp = get_max_displacement(lT)
    lT.displacement_ratio = {
        node: disp[node] / max_disp[node] for node in disp if max_disp[node]
    }
    return lT.displacement_ratio


def get_outreach_ratio(
    lT: LineageTree, sigma: float = 1.5
) -> dict[int, float]:
    max_disp = get_max_displacement(lT)
    track_length = get_track_length(lT, sigma)
    lT.displacement_ratio = {
        node: max_disp[node] / track_length[node]
        for node in track_length
        if track_length[node]
    }
    return lT.displacement_ratio


def get_straightness(lT: LineageTree, sigma: float = 1.5) -> dict[int, float]:
    displacement = get_displacement(lT)
    track_length = get_track_length(lT, sigma)
    lT.straightness = {
        node: displacement[node] / track_length[node] for node in displacement
    }
    return lT.straightness


def _inertia_matrix(chain, sigma: float = 1.5) -> np.ndarray:
    """Calculates the inertia of a given point cloud.

    Parameters
    ----------
    lT : LineageTree
        The lineageTree object
    sigma : float, optional
        The smoothing factor, by default 1.5

    Returns
    -------
    np.ndarray
        The 3x3 inertia array of a point cloud.

    """
    smoothed_chain = np.zeros_like(chain, dtype=float)
    if sigma != 0:
        smoothed_chain[:, 0] = anchored_gaussian_smooth(chain[:, 0], sigma)
        smoothed_chain[:, 1] = anchored_gaussian_smooth(chain[:, 1], sigma)
        smoothed_chain[:, 2] = anchored_gaussian_smooth(chain[:, 2], sigma)
    else:
        smoothed_chain = chain  # Not smoothed
    cloud = smoothed_chain - np.mean(smoothed_chain, axis=0)  # Centering
    Ixx = np.sum(cloud[:, 1] ** 2 + cloud[:, 2] ** 2)
    Iyy = np.sum(cloud[:, 2] ** 2 + cloud[:, 0] ** 2)
    Izz = np.sum(cloud[:, 0] ** 2 + cloud[:, 1] ** 2)

    Ixy = -np.sum(cloud[:, 0] * cloud[:, 1])
    Ixz = -np.sum(cloud[:, 0] * cloud[:, 2])
    Iyz = -np.sum(cloud[:, 1] * cloud[:, 2])

    return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])


def get_asphericity(lT: LineageTree, sigma: float = 1.5) -> dict[int, float]:
    """Calculate the asphericity of a track. For 0 it is perfectly spherical, while for 1 it is not spherical.
    Adapted from : J Rudnick and G Gaspari 1986 J. Phys. A: Math. Gen. 19 L191

    Parameters
    ----------
    lT : LineageTree
        The LineageTree object
    sigma : _type_
        Smoothing factor.

    Returns
    -------
    dict[int, float]:
        dictionary that maps each node to the sphericity of each respective track.

    """
    lT.asphericity = {}

    for chain in lT.all_chains:
        chain_pos = np.array([lT.pos[c] for c in chain])
        if len(chain) < 4:
            continue
        else:
            inertia = _inertia_matrix(chain_pos, sigma)
        eig_vals = np.linalg.eigvals(inertia)
        tr = eig_vals[0] ** 2 + eig_vals[1] ** 2 + eig_vals[2] ** 2
        M = (
            eig_vals[0] ** 2 * eig_vals[1] ** 2
            + eig_vals[1] ** 2 * eig_vals[2] ** 2
            + eig_vals[0] ** 2 * eig_vals[2] ** 2
        )
        asphericity = (tr**2 - 3 * M) / (tr**2)
        lT.asphericity.update({node: asphericity for node in chain})
    return lT.asphericity


def get_angles(lT: LineageTree, sigma: float = 1.5) -> dict[int, float]:
    """Angles between adjacent edges.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree object.
    sigma : float, optional
        The smoothing factor, by default 1.5

    Returns
    -------
    dict[int, float]
        dictionary that maps each node to the angle between its diplacement and the next ones displacement.
        For example for and edge containing the nodes o,i,e,a:  o -> i -> e -> a
        for i its the angle of o->i to i->e, for e is the angle of i->e to e->a
    """
    lT.angles = {}
    for chain in lT.all_chains:
        if len(chain) < 3:
            continue
        else:
            positions = np.array([lT.pos[c] for c in chain])
            smoothed_positions = np.zeros_like(positions, dtype=float)
            if sigma != 0:
                smoothed_positions[:, 0] = anchored_gaussian_smooth(
                    positions[:, 0], sigma
                )
                smoothed_positions[:, 1] = anchored_gaussian_smooth(
                    positions[:, 1], sigma
                )
                smoothed_positions[:, 2] = anchored_gaussian_smooth(
                    positions[:, 2], sigma
                )
            else:
                smoothed_positions = positions  # Not smoothed

            vectors1 = smoothed_positions[:-1] - smoothed_positions[1:]
            vectors2 = vectors1[1:]
            angles = [
                (v1 @ v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                for v1, v2 in zip(vectors1, vectors2)
            ]
            angles = [0] + angles + [0]  # first and last node have no angle
            lT.angles.update(
                {node: angle for node, angle in zip(chain, angles)}
            )
    return lT.angles


def get_overall_angle(lT: LineageTree) -> dict[int, float]:
    """The angle (degrees) between the first and the last segment of the given track.
    Angles are measured symmetrically, thus the return values range from 0 to pi;
    for instance, both a 90 degrees left and right turns yield the same value pi/2 radians.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree object.

    Returns
    -------
    dict[int, float]
        dictionary that maps the overall angle of a given chain to every node of its chain
    """
    lT.overall_angles = {}
    for chain in lT.all_chains:
        if len(chain) < 3:
            continue
        else:
            vector1 = lT.pos[chain[1]] - lT.pos[chain[0]]
            vector2 = lT.pos[chain[-1]] - lT.pos[chain[-2]]
            overangle = (vector1 @ vector2) / (
                np.linalg.norm(vector1) * np.linalg.norm(vector2)
            )
            lT.overall_angles.update({node: overangle for node in chain})
    return lT.overall_angles


# TODO

# * **trackLength**: sums up the distances between subsequent positions; in other words, it estimates the length of the underlying subtrack by linear interpolation (usually an underestimation)
# * **duration**: time elapsed between first and last positions of the subtrack
# * **maxDisplacement**: maximal Euclidean distance of any position on the subtrack from the first position
# * **speed**: trackLength/duration
# * **displacement**: Euclidean distance between the track starting and endpoints
# * **squareDisplacement**: squared Euclidean distance between the track starting and endpoints
# * **displacementRatio**: displacement/maxDisplacement (values between 0 and 1, where 1 means a perfectly straight track)
# * **outreachRatio**: maxDisplacement/trackLength (values between 0 and 1, where 1 means a perfectly straight track)
# * **straightness**: displacement/trackLength (values between 0 and 1, where 1 means a perfectly straight track)
# * **asphericity**: a different appraoch to measure straightness, that computes the asphericity of the set of positions on the subtrack _via_ the length of its principal components (number between 0 and 1, with higher values indicating straighter tracks). Unlike straightness, however, asphericity ignores back-and-forth motion of the object, so something that bounces between two positions will have low straightness but high asphericity. We define the asphericity of every track with two or fewer positions to be 1.
# * **overallAngle**: angle (degrees) between the first and the last segment of the given track. Angles are measured symmetrically, thus the return values range from 0 to pi; for instance, both a 90 degrees left and right turns yield the same value pi/2 radians.
# * **meanTurningAngle**: averages the overallAngle over all adjacent segments of a given track; a low meanTurningAngle indicates high persistence of orientation, whereas for an uncorrelated random walk we expect 90 degrees. Note that angle measurements will yield NA values for tracks in which two subsequent positions are identical.
# * **overallDot**: computes the dot product between the first and the last segment of the given track.
# * **overallNormDot**: computes the dot product between the unit vectors along the first and the last segment of the given track. These two functions may be useful to generate autocovariance plots.
# * **fractalDimension**: fractal dimension is a mathematical concept used to describe the complexity of self-similar patterns or structures, such as fractals. It is a measure of how much detail or irregularity is present in a pattern or structure. This function estimates the fractal dimension of a track using the function fd.estim.boxcount, which involves dividing the cell trajectories into smaller and smaller boxes of a given size, counting the number of boxes that contain part of the trajectory, and then using this information to estimate the fractal dimension. In general, a higher fractal dimension can indicate that the cell track is more complex or irregular, and may be more invasive or aggressive. Conversely, a lower fractal dimension may indicate a more regular or uniform track shape, and may be associated with less invasive or aggressive behavior.

# NOTE: With increasing window sizes, the number of available timepoints per cell decrease, since we can not create subtracks of length w starting in the last w timepoints.
