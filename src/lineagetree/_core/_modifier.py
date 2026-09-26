from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import wraps
from typing import TYPE_CHECKING
import numpy as np
from scipy.ndimage import gaussian_filter1d

if TYPE_CHECKING:
    from ..lineage_tree import LineageTree


def modifier(wrapped_func):
    """Decorator that invalidates all cached dynamic properties after a mutation.

    Wrap any function that mutates the tree's topology, times, or positions
    with ``@modifier``. After the wrapped function returns, every backing
    attribute listed in ``self._protected_dynamic_properties`` is reset to
    ``None``, causing ``dynamic_property`` descriptors to recompute their
    values on the next access.

    Re-entrant calls (modifier functions calling other modifier functions) are
    detected via ``self._has_been_reset``; only the outermost call performs the
    invalidation so the cache is flushed exactly once per logical mutation.

    Parameters
    ----------
    wrapped_func : callable
        The mutation function to wrap. Its first argument must be the
        ``LineageTree`` instance (``self``).

    Returns
    -------
    callable
        The wrapped function with cache-invalidation behaviour.
    """

    @wraps(wrapped_func)
    def raising_flag(self, *args, **kwargs):
        should_reset = (
            not hasattr(self, "_has_been_reset") or not self._has_been_reset
        )
        out_func = wrapped_func(self, *args, **kwargs)
        if should_reset:
            for prop in self._protected_dynamic_properties:
                self.__dict__[prop] = None
            self._has_been_reset = True
        return out_func

    return raising_flag


###TODO pos can be callable and stay motionless (copy the position of the succ node, use something like optical flow)
@modifier
def add_chain(
    lT: LineageTree,
    node: int,
    length: int,
    downstream: bool,
    pos: Callable | None = None,
) -> int:
    """Add a chain of new nodes before or after a node.

    Each new node is one time point after (``downstream=True``) or before
    (``downstream=False``) the previous one. Adding a chain downstream of a
    node that already has successors makes that node divide.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    node : int
        The node to attach the chain to. With ``downstream=False`` it must
        be a root.
    length : int
        Number of nodes to add. With 0, nothing is added and ``node`` is
        returned.
    downstream : bool
        If True, the chain follows ``node`` in time; otherwise it precedes
        it, and ``node`` stops being a root.
    pos : Callable, optional
        Not used yet; the new nodes get no position.

    Returns
    -------
    int
        Id of the last node added: the new leaf when ``downstream`` is
        True, the new root otherwise.

    Raises
    ------
    ValueError
        If ``length`` is negative.
    Warning
        If ``downstream`` is False and ``node`` has a predecessor, or the
        chain would start before ``lT.t_b``.
    """
    if length == 0:
        return node
    if length < 1:
        raise ValueError("Length cannot be <1")
    if downstream:
        for _ in range(int(length)):
            old_node = node
            node = lT._add_node(pred=[old_node])
            lT._time[node] = lT._time[old_node] + 1
    else:
        if lT._predecessor[node]:
            raise Warning("The node already has a predecessor.")
        if lT._time[node] - length < lT.t_b:
            raise Warning(
                "A node cannot created outside the lower bound of the dataset."
                "(It is possible to change it by lT.t_b = int(...))"
            )
        for _ in range(int(length)):
            old_node = node
            node = lT._add_node(succ=[old_node])
            lT._time[node] = lT._time[old_node] - 1
    return node


@modifier
def add_root(lT: LineageTree, t: int, pos: list | None = None) -> int:
    """Add an isolated node, which is both a root and a leaf.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    t : int
        The time point of the new node.
    pos : list or tuple, optional
        The position of the new node.

    Returns
    -------
    int
        The id of the new root.
    """
    C_next = lT.get_next_id()
    lT._successor[C_next] = ()
    lT._predecessor[C_next] = ()
    lT._time[C_next] = t
    if isinstance(pos, (list, tuple)):
        lT.pos[C_next] = np.array(pos)
    lT._changed_roots = True
    return C_next


def get_next_id(lT) -> int:
    """Return an unused node id and reserve it.

    Ids listed in ``lT.next_id``, if any, are used first; otherwise the id
    is one more than the largest id given so far.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.

    Returns
    -------
    int
        The next authorized id.
    """
    if not hasattr(lT, "max_id") or (lT.max_id == -1 and lT.nodes):
        lT.max_id = max(lT.nodes) if len(lT.nodes) else 0
    if not hasattr(lT, "next_id") or lT.next_id == []:
        lT.max_id += 1
        return lT.max_id
    else:
        return lT.next_id.pop()


@modifier
def _add_node(
    lT: LineageTree,
    succ: list | None = None,
    pred: list | None = None,
    pos: Iterable | None = None,
    nid: int | None = None,
) -> int:
    """Add a node before or after existing nodes.

    The time of the new node is not set; the caller has to set it.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    succ : list of int, optional
        Successors of the new node, which becomes their predecessor.
    pred : list of int, optional
        One-element list holding the predecessor of the new node, which
        becomes one of its successors.
    pos : list, optional
        Position of the new node. Ignored unless it is a ``list``.
    nid : int, optional
        Id value of the new node, to be used carefully. If None, the new id is
        automatically computed.

    Returns
    -------
    int
        Id of the new node.

    Raises
    ------
    Warning
        If neither ``succ`` nor ``pred`` is given; use
        [`add_root`][lineagetree.LineageTree.add_root] instead.
    """
    if not succ and not pred:
        raise Warning(
            "Please enter a successor or a predecessor, otherwise use the add_roots() function."
        )
    C_next = lT.get_next_id() if nid is None else nid
    if succ:
        lT._successor[C_next] = succ
        for suc in succ:
            lT._predecessor[suc] = (C_next,)
    else:
        lT._successor[C_next] = ()
    if pred:
        lT._predecessor[C_next] = pred
        lT._successor[pred[0]] = lT._successor.setdefault(pred[0], ()) + (
            C_next,
        )
    else:
        lT._predecessor[C_next] = ()
    if isinstance(pos, list):
        lT.pos[C_next] = pos
    return C_next


@modifier
def remove_nodes(lT: LineageTree, group: int | set | list) -> None:
    """Remove one or more nodes from the tree.

    The nodes are also removed from every property of the tree that is a
    dictionary. The successors of a removed node become roots.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    group : int or set of int or list of int
        The node(s) to remove. Ids that are not nodes of the tree are
        ignored.
    """
    if isinstance(group, int | float):
        group = {group}
    if isinstance(group, list):
        group = set(group)
    group = lT.nodes.intersection(group)
    for node in group:
        for attr in lT.__dict__:
            attr_value = lT.__getattribute__(attr)
            if isinstance(attr_value, dict) and attr not in [
                "successor",
                "predecessor",
                "_successor",
                "_predecessor",
            ]:
                attr_value.pop(node, ())
        if lT._predecessor.get(node):
            lT._successor[lT._predecessor[node][0]] = tuple(
                set(lT._successor[lT._predecessor[node][0]]).difference(group)
            )
        for p_node in lT._successor.get(node, []):
            lT._predecessor[p_node] = ()
        lT._predecessor.pop(node, ())
        lT._successor.pop(node, ())


@staticmethod
def compute_rigid_transform(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute the rigid transformation between two paired 3D point clouds.

    The transformation is a rotation followed by a translation.

    Parameters
    ----------
    A : numpy.ndarray of shape (N, 3)
        Source point cloud.
    B : numpy.ndarray of shape (N, 3)
        Target point cloud.

    Returns
    -------
    numpy.ndarray of shape (4, 4)
        The rigid transformation matrix in homogeneous coordinates.
    """
    assert A.shape == B.shape, "Point clouds A and B must have the same shape."

    # Compute centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # Center the point clouds
    AA = A - centroid_A
    BB = B - centroid_B

    # Compute the covariance matrix
    H = AA.T @ BB

    # Compute the SVD
    U, _, Vt = np.linalg.svd(H)

    # Compute the rotation matrix
    R = Vt.T @ U.T

    # Ensure the rotation matrix is proper (no reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute the translation vector
    t = centroid_B - R @ centroid_A

    m = np.identity(4)
    m[:-1, :-1] = R
    m[:-1, -1] = t

    return m


@staticmethod
def iterative_composition(trsfs: list[np.ndarray]) -> list[np.ndarray]:
    """Iteratively compose a list of transformations.

    The composition is defined such that ``result[0]`` is the identity and
    ``result[i] = result[i - 1] @ trsfs[i - 1]``.

    Parameters
    ----------
    trsfs : list of numpy.ndarray of shape (N, N)
        List of transformation matrices in homogeneous coordinates.

    Returns
    -------
    list of numpy.ndarray of shape (N, N)
        List of iteratively composed transformations.
    """
    new_trsfs = [np.identity(4)]
    for trsf in trsfs:
        new_trsfs.append(new_trsfs[-1] @ trsf)
    return new_trsfs


@staticmethod
def apply_trsf(m: np.ndarray, pos: np.ndarray):
    """Apply a transformation to an array of positions.

    Parameters
    ----------
    m : numpy.ndarray of shape (N, N)
        A transformation matrix in homogeneous coordinates for positions in
        N-1 dimensions.
    pos : numpy.ndarray of shape (M, N-1)
        A list of positions in N-1 dimensions to be transformed.

    Returns
    -------
    numpy.ndarray of shape (M, N-1)
        The original positions transformed by ``m``.
    """
    pos_padded = np.pad(pos, ((0, 0), (0, 1)), "constant", constant_values=1).T

    return np.dot(m, pos_padded)[:-1].T


@modifier
def stabilise_positions(lT: LineageTree) -> dict[int, np.ndarray]:
    """Register node positions to minimise inter-frame displacement.

    The positions of each time point are rotated and translated (a rigid
    transformation) so that the sum of the squared displacements between
    consecutive time points is minimal. ``lT.pos`` is replaced by the new
    positions and the old ones are kept in ``lT.old_pos``. Positions must be
    3D.

    Warnings
    --------
    Strongly coordinated movements, such as a rotation of the whole
    embryo, may be smoothed out significantly.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.

    Returns
    -------
    dict of {int: numpy.ndarray}
        A dictionary similar to ``lT.pos`` with the registered positions.
    """
    times = sorted(lT.time_nodes)
    trsfs_i_j = []
    # ok_cells = set([c for c, pos in lT.pos.items() if pos.shape == (3,)])

    for t1 in times[:-1]:
        nodes_i, nodes_j = [], []
        for c in lT.time_nodes[t1]:
            putative_next_n = lT.successor[c]
            next_n = []
            for ci in putative_next_n:
                if 3 <= len(lT.pos.get(ci, [])):
                    next_n.append(ci)
            if 0 < len(next_n):
                nodes_i.append(c)
                nodes_j.append(next_n)

        pos_i = np.array([lT.pos[c] for c in nodes_i])
        pos_j = np.array(
            [np.mean([lT.pos[ci] for ci in c], axis=0) for c in nodes_j]
        )

        trsf_i_j = compute_rigid_transform(pos_j, pos_i)
        trsfs_i_j.append(trsf_i_j)

    final_trsfs = iterative_composition(trsfs_i_j)

    new_pos = {}
    for i, trsf in enumerate(final_trsfs):
        nodes = lT.time_nodes[times[i]]
        pos = np.array([lT.pos[c] for c in nodes])
        new_pos.update(zip(nodes, apply_trsf(trsf, pos)))

    lT.old_pos = lT.pos
    lT.pos = new_pos

    return new_pos


def anchored_gaussian_smooth(data, sigma=1.5, anchor_strength=3.0):
    """Apply anchored Gaussian smoothing to a 1D sequence.

    Smooth a 1D sequence while anchoring the endpoints and suppressing drift
    near the boundaries.

    This function performs standard Gaussian smoothing and then blends the
    smoothed result with the original data using a position-dependent weight.
    The weights enforce exact anchoring at the first and last elements and
    progressively relax toward the center of the array.

    Parameters
    ----------
    data : array_like
        Input 1D sequence of numeric values.
    sigma : float, optional
        Standard deviation of the Gaussian kernel used for smoothing.
        Higher values produce stronger smoothing. Default is 1.5.
    anchor_strength : float, optional
        Controls how strongly the endpoints influence nearby values.
        Smaller values result in tighter anchoring (less smoothing near edges),
        while larger values allow more smoothing across the entire array.
        Default is 3.0.

    Returns
    -------
    numpy.ndarray
        Smoothed array of the same shape as the input, with the first and last
        elements exactly equal to the original values.

    Notes
    -----
    - The method applies a Gaussian filter followed by a spatially varying
      convex combination:
          result[i] = alpha[i] * data[i] + (1 - alpha[i]) * smoothed[i]
      where alpha[i] decays exponentially with distance from the nearest
      endpoint.
    - This introduces soft boundary conditions (anchoring), breaking the
      shift-invariance of standard convolution-based smoothing.
    - The endpoints are strictly preserved (Dirichlet boundary condition),
      and nearby points are partially constrained depending on their distance
      to the boundaries.

    Examples
    --------
    >>> out = anchored_gaussian_smooth([10, 12, 15, 20, 18, 16, 14], sigma=1.5)
    >>> float(out[0]), float(out[-1])
    (10.0, 14.0)
    """
    data = np.asarray(data, dtype=float)

    # Standard Gaussian smoothing
    smoothed = gaussian_filter1d(data, sigma=sigma, mode="nearest")

    n = data.size
    i = np.arange(n)

    # Distance to nearest endpoint
    dist = np.minimum(i, n - 1 - i)

    # Exponential decay: strong anchoring near edges
    alpha = np.exp(-dist / anchor_strength)

    # Ensure exact anchoring at endpoints
    alpha[0] = 1.0
    alpha[-1] = 1.0

    # Blend original and smoothed
    return alpha * data + (1 - alpha) * smoothed


@modifier
def smooth_trajectories(lT: LineageTree, sigma=1.0, ancor_strength=3):
    """Smooth 3D trajectories of all chains using anchored Gaussian filtering.

    For each chain in the lineage tree, the x-, y-, and z-coordinates are
    independently smoothed using a Gaussian filter with soft endpoint
    constraints. The first and last positions of each chain are preserved
    exactly, while nearby points are partially constrained to reduce drift.
    ``lT.pos`` is replaced by the smoothed positions and the old ones are
    kept in ``lT.old_pos``.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    sigma : float, default=1.0
        Standard deviation, in time points, of the Gaussian kernel. Higher
        values produce smoother trajectories.
    ancor_strength : float, default=3
        Controls the strength of endpoint anchoring. Smaller values enforce
        stronger constraints near the start and end of each chain (less
        drift), while larger values allow more global smoothing.

    Returns
    -------
    dict of {int: numpy.ndarray}
        The new ``lT.pos``: each node mapped to its smoothed 3D position.
    """
    new_pos = {}
    for chain in lT.all_chains:
        X, Y, Z = np.array([lT.pos[c] for c in chain]).T
        X_new = anchored_gaussian_smooth(
            X, sigma=sigma, anchor_strength=ancor_strength
        )
        Y_new = anchored_gaussian_smooth(
            Y, sigma=sigma, anchor_strength=ancor_strength
        )
        Z_new = anchored_gaussian_smooth(
            Z, sigma=sigma, anchor_strength=ancor_strength
        )
        new_pos.update(zip(chain, np.transpose([X_new, Y_new, Z_new])))
    lT.old_pos = lT.pos
    lT.pos = new_pos
    return lT.pos
