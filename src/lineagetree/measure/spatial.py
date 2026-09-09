from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, Iterable

import numpy as np
from scipy.spatial import Delaunay, KDTree
from .._core._modifier import anchored_gaussian_smooth
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import curve_fit

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


def track_length(lT: LineageTree, sigma: float = 1.5) -> dict[int, float]:
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
    tr_length = {}
    for track in lT.all_chains:
        data = np.array([lT.pos[c] for c in track])
        dist = _track_length(data, sigma)
        tr_length.update({node: dist for node in track})
    lT.Track_length = tr_length
    return lT.Track_length


def duration(lT: LineageTree) -> dict[int, float]:
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
    lT.Duration = {
        node: len(lT.get_chain_of_node(node)) * lT.time_resolution
        for node in lT.nodes
    }
    return lT.Duration


def max_displacement(lT: LineageTree) -> dict[int, float]:
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

    lT.Max_displacement = {}
    for chain in lT.all_chains:
        root = lT.get_ancestor_at_t(chain[0])
        if root not in lT.nodes:
            continue
        root_pos = lT.pos[root]
        positions = np.array([lT.pos[c] for c in chain])
        displacements = np.cumsum(
            np.linalg.norm((positions - root_pos), axis=1)
        )
        lT.Max_displacement.update(
            {node: disp for node, disp in zip(chain, displacements)}
        )
    return lT.Max_displacement


def speed(lT: LineageTree, sigma: float = 1.5) -> dict[int, float]:
    """The speed of each node.

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
        dictionary that maps each node to its speed (distance travelled/time_resolution)
    """

    tr_length = track_length(lT, sigma)
    lT.Speed = {
        node: dist / lT.time_resolution for node, dist in tr_length.items()
    }
    return lT.Speed


def displacement(lT: LineageTree) -> dict[int, float]:
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
    lT.Displacement = displacement
    return lT.Displacement


def velocity(lT: LineageTree) -> dict[int, float]:
    """The velocity of each node.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree object.

    Returns
    -------
    dict[int, float]
        dictionary that maps each node to its velocity
    """
    disp = displacement(lT)
    lT.Velocity = {
        node: dist / lT.time_resolution for node, dist in disp.items()
    }
    return lT.Velocity


def mean_squared_displacement(
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


def displacement_ratio(lT: LineageTree) -> dict[int, float]:
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
    disp = displacement(lT)
    max_disp = max_displacement(lT)
    lT.Displacement_ratio = {
        node: disp[node] / max_disp[node] for node in disp if max_disp[node]
    }
    return lT.Displacement_ratio


def outreach_ratio(lT: LineageTree, sigma: float = 1.5) -> dict[int, float]:
    """maxDisplacement/trackLength
    (values between 0 and 1, where 1 means a perfectly straight track)

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
        dictionary that maps erach node to the outreach ratio of its chain
    """
    max_disp = max_displacement(lT)
    tr_length = track_length(lT, sigma)
    lT.Outreach_ratio = {
        node: max_disp[node] / tr_length[node]
        for node in tr_length
        if tr_length[node]
    }
    return lT.Outreach_ratio


def straightness(lT: LineageTree, sigma: float = 1.5) -> dict[int, float]:
    """displacement/trackLength
    (values between 0 and 1, where 1 means a perfectly straight track)

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
        dictionary that maps each node to the straightness of its chain
    """
    displ = displacement(lT)
    tr_length = track_length(lT, sigma)
    lT.Straightness = {node: displ[node] / tr_length[node] for node in displ}
    return lT.Straightness


def _inertia_matrix(chain, sigma: float = 1.5) -> np.ndarray:
    """Calculates the inertia of a given point cloud.

    Parameters
    ----------
    lT : LineageTree
        The lineageTree object
    sigma : float, optional
        Standard deviation of the Gaussian kernel used for smoothing.
        Higher values produce stronger smoothing, by default is 1.5.

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


def asphericity(lT: LineageTree, sigma: float = 1.5) -> dict[int, float]:
    """Calculate the asphericity of a track. For 0 it is perfectly spherical, while for 1 it is not spherical.
    Adapted from : J Rudnick and G Gaspari 1986 J. Phys. A: Math. Gen. 19 L191

    Parameters
    ----------
    lT : LineageTree
        The LineageTree object
    sigma : float, by degault 1.5
        Standard deviation of the Gaussian kernel used for smoothing.
        Higher values produce stronger smoothing, by default is 1.5.

    Returns
    -------
    dict[int, float]:
        dictionary that maps each node to the sphericity of each respective track.

    """
    lT.Asphericity = {}

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
        lT.Asphericity.update({node: asphericity for node in chain})
    return lT.Asphericity


def angles(lT: LineageTree, sigma: float = 1.5) -> dict[int, float]:
    """Angles between adjacent edges.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree object.
    sigma : float, by degault 1.5
        Standard deviation of the Gaussian kernel used for smoothing.
        Higher values produce stronger smoothing, by default is 1.5.

    Returns
    -------
    dict[int, float]
        dictionary that maps each node to the angle between its diplacement and the next ones displacement.
        For example for and edge containing the nodes o,i,e,a:  o -> i -> e -> a
        for i its the angle of o->i to i->e, for e is the angle of i->e to e->a
    """
    lT.Angles = {}
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
            lT.Angles.update(
                {node: angle for node, angle in zip(chain, angles)}
            )
    return lT.Angles


def overall_angle(lT: LineageTree) -> dict[int, float]:
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
    lT.Overall_angles = {}
    for chain in lT.all_chains:
        if len(chain) < 3:
            continue
        else:
            vector1 = lT.pos[chain[1]] - lT.pos[chain[0]]
            vector2 = lT.pos[chain[-1]] - lT.pos[chain[-2]]
            overangle = (vector1 @ vector2) / (
                np.linalg.norm(vector1) * np.linalg.norm(vector2)
            )
            lT.Overall_angles.update({node: overangle for node in chain})
    return lT.Overall_angles


def alpha_score(
    lT: LineageTree,
):
    def _model(x, a, b):
        return b * (x**a)

    alphas = {}
    for chain in lT.all_chains:
        if len(chain) < 5 or chain[0] in lT.roots or chain[-1] in lT.leaves:
            alphas.update({node: np.nan for node in chain})
        positions = np.array([lT.pos[c] for c in chain])
        MSD = np.cumsum(
            np.linalg.norm((positions - positions[0]) ** 2, axis=1)
        ) / np.arange(1, len(chain) + 1)
        pars, *_ = curve_fit(_model, np.linspace(0, len(MSD), 50), MSD)
        alphas.update({node: pars[0] for node in chain})
    return alphas


# TODO

# * **meanTurningAngle**: averages the overallAngle over all adjacent segments of a given track; a low meanTurningAngle indicates high persistence of orientation, whereas for an uncorrelated random walk we expect 90 degrees. Note that angle measurements will yield NA values for tracks in which two subsequent positions are identical.
# * **overallDot**: computes the dot product between the first and the last segment of the given track.
# * **overallNormDot**: computes the dot product between the unit vectors along the first and the last segment of the given track. These two functions may be useful to generate autocovariance plots.
# * **fractalDimension**: fractal dimension is a mathematical concept used to describe the complexity of self-similar patterns or structures, such as fractals. It is a measure of how much detail or irregularity is present in a pattern or structure. This function estimates the fractal dimension of a track using the function fd.estim.boxcount, which involves dividing the cell trajectories into smaller and smaller boxes of a given size, counting the number of boxes that contain part of the trajectory, and then using this information to estimate the fractal dimension. In general, a higher fractal dimension can indicate that the cell track is more complex or irregular, and may be more invasive or aggressive. Conversely, a lower fractal dimension may indicate a more regular or uniform track shape, and may be associated with less invasive or aggressive behavior.

# NOTE: With increasing window sizes, the number of available timepoints per cell decrease, since we can not create subtracks of length w starting in the last w timepoints.
