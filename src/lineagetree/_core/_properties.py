from __future__ import annotations

import warnings
from itertools import combinations
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import dok_array

if TYPE_CHECKING:
    from ..lineage_tree import LineageTree


class dynamic_property(property):
    """A cached property descriptor that supports automatic cache invalidation.

    Extends :class:`property` to add lazy evaluation and caching. The computed
    value is stored in a backing attribute (named ``_<property_name>`` by
    default) on the instance. The backing attribute is set to ``None`` by the
    ``modifier`` decorator whenever the tree is mutated, triggering
    recomputation on the next access.

    Parameters
    ----------
    fget : callable, optional
        Getter function, as for :class:`property`.
    fset : callable, optional
        Setter function, as for :class:`property`.
    fdel : callable, optional
        Deleter function, as for :class:`property`.
    doc : str, optional
        Docstring, as for :class:`property`.
    protected_name : str, optional
        Name of the backing attribute used to store the cached value.
        Defaults to ``'_<property_name>'``.
    """

    def __init__(
        self, fget=None, fset=None, fdel=None, doc=None, protected_name=None
    ):
        super().__init__(fget, fset, fdel, doc)
        self.protected_name = protected_name

    def __set_name__(self, owner, name):
        """Register this descriptor on the owner class.

        Parameters
        ----------
        owner : type
            The class that owns this descriptor.
        name : str
            The attribute name assigned to this descriptor.
        """
        self.name = name
        if self.protected_name is None:
            self.protected_name = f"_{name}"
        if not hasattr(owner, "_protected_dynamic_properties"):
            owner._protected_dynamic_properties = []
        owner._protected_dynamic_properties.append(self.protected_name)
        if not hasattr(owner, "_dynamic_properties"):
            owner._dynamic_properties = []
        owner._dynamic_properties += [name, self.protected_name]
        setattr(owner, self.protected_name, None)

    def __get__(self, instance, owner):
        """Return the cached value, computing it on first access.

        Parameters
        ----------
        instance : LineageTree or None
            The instance the descriptor is accessed on. If ``None``, the
            descriptor itself is returned (class-level access).
        owner : type
            The owner class.

        Returns
        -------
        object
            The cached (or freshly computed) property value.
        """
        if instance is None:
            return self
        instance._has_been_reset = False
        if getattr(instance, self.protected_name) is None:
            value = super().__get__(instance, owner)
            setattr(instance, self.protected_name, value)
            return value
        else:
            return getattr(instance, self.protected_name)


def _compute_all_chains(lT: LineageTree) -> tuple[tuple[int]]:
    """Compute all the chains of a given lineage tree.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.

    Returns
    -------
    tuple of tuple of int
        The chains of the lineage tree.
    """
    all_chains = []
    to_do = sorted(lT.roots, key=lT.time.get, reverse=True)
    while len(to_do) != 0:
        current = to_do.pop()
        chain = lT.get_chain_of_node(current)
        all_chains += [chain]
        to_do.extend(lT._successor[chain[-1]])
    return tuple(tuple(chain) for chain in all_chains)


@property
def successor(lT: LineageTree) -> MappingProxyType[int, tuple[int]]:
    """Dictionary that maps a node to its successors."""
    if not hasattr(lT, "_protected_successor"):
        lT._protected_successor = MappingProxyType(lT._successor)
    return lT._protected_successor


@property
def predecessor(lT: LineageTree) -> MappingProxyType[int, tuple[int]]:
    """Dictionary that maps a node to its predecessors."""
    if not hasattr(lT, "_protected_predecessor"):
        lT._protected_predecessor = MappingProxyType(lT._predecessor)
    return lT._protected_predecessor


@property
def time(lT: LineageTree) -> MappingProxyType[int, int]:
    """Dictionary that maps a node to its time."""
    if not hasattr(lT, "_protected_time"):
        lT._protected_time = MappingProxyType(lT._time)
    return lT._protected_time


@dynamic_property
def t_b(lT: LineageTree) -> int:
    """The first timepoint of the lineage tree."""
    return min(lT._time.values())


@dynamic_property
def t_e(lT: LineageTree) -> int:
    """The last timepoint of the lineage tree."""
    return max(lT._time.values())


@dynamic_property
def nodes(lT: LineageTree) -> frozenset[int]:
    """Set of node ids of the lineage tree."""
    return frozenset(lT._successor.keys())


@dynamic_property
def number_of_nodes(lT: LineageTree) -> int:
    """Number of nodes in the lineage tree."""
    return len(lT.nodes)


@dynamic_property
def depth(lT: LineageTree) -> dict[int, int]:
    """The depth of each node in the lineage tree."""
    _depth = {r: 0 for r in lT.roots}
    for root in lT.roots:
        to_do = list(lT.successor[root])
        while to_do:
            current = to_do.pop()
            _depth[current] = _depth[lT.predecessor[current][0]] + 1
            to_do.extend(lT.successor[current])
    return _depth


@dynamic_property
def roots(lT: LineageTree) -> frozenset[int]:
    """Set of roots of the lineage tree."""
    return frozenset({s for s, p in lT._predecessor.items() if p == ()})


@dynamic_property
def leaves(lT: LineageTree) -> frozenset[int]:
    """Set of leaves of the lineage tree."""
    return frozenset({p for p, s in lT._successor.items() if s == ()})


@dynamic_property
def edges(lT: LineageTree) -> tuple[tuple[int, int]]:
    """Set of edges of the lineage tree."""
    return tuple((p, si) for p, s in lT._successor.items() for si in s)


@property
def labels(lT: LineageTree) -> dict[int, str]:
    """Dictionary that maps a node to its string label.

    Labels are determined by the following priority:

    1. If ``lT._labels`` is already set (e.g. loaded from file), use it.
    2. Else if ``lT.node_name`` exists, use it as the label dictionary.
    3. Else apply a heuristic: label a root as ``"Unlabeled"`` only when at
       least one of its leaves is far enough in time (≥ 1/4 of the full time
       range) from the root.

    The name of the attribute that was used as labels is stored in
    ``lT.labels_name``.
    """
    if not hasattr(lT, "_labels"):
        if hasattr(lT, "node_name"):
            lT.labels_name = "node_name"
            lT._labels = {
                chain[0]: lT.node_name.get(chain[0], "Unlabeled")
                for chain in lT.all_chains
            }
        else:
            lT.labels_name = ""
            lT._labels = {
                root: "Unlabeled"
                for root in lT.roots
                for leaf in lT.find_leaves(root)
                if abs(lT._time[leaf] - lT._time[root])
                >= abs(lT.t_e - lT.t_b) / 4
            }
    return lT._labels


@property
def time_resolution(lT: LineageTree) -> float:
    """Time resolution of the lineage tree in minutes (or the unit chosen by the user).

    Internally stored as ``int(_time_resolution * 10)`` to avoid floating-
    point accumulation; the getter divides by 10 to restore the original
    scale. A value of ``0`` means "unset / unknown".

    Returns
    -------
    float
        Time resolution. ``0.0`` when not set.
    """
    if not hasattr(lT, "_time_resolution"):
        lT._time_resolution = 0
    return lT._time_resolution / 10


@time_resolution.setter
def time_resolution(lT, time_resolution: float) -> None:
    """Set the time resolution.

    Parameters
    ----------
    time_resolution : float
        Positive time resolution value. Non-positive values or ``None`` are
        rejected and the resolution is reset to ``0`` with a warning.
    """
    if time_resolution is not None and time_resolution > 0:
        lT._time_resolution = int(time_resolution * 10)
    else:
        warnings.warn("Time resolution set to default 0", stacklevel=2)
        lT._time_resolution = 0


@dynamic_property
def all_chains(lT: LineageTree) -> tuple[tuple[int]]:
    """List of all chains in the tree, ordered in depth-first search."""
    return _compute_all_chains(lT)


@dynamic_property
def time_nodes(lT: LineageTree) -> dict[int, set[int]]:
    """Dictionary that maps a time to the set of nodes at that time."""
    _time_nodes = {}
    for c, t in lT._time.items():
        _time_nodes.setdefault(t, set()).add(c)
    return _time_nodes


def _m(lT: LineageTree, i: int, j: int) -> float:
    """Compute the signed ancestor distance from node ``i`` to node ``j``.

    The value is the number of edges from ``i`` to ``j`` travelling only
    through predecessors of ``j``. A positive value means ``i`` is an ancestor
    of ``j``; a negative value is stored on the transposed index ``(j, i)``.
    Returns ``np.inf`` when ``i`` is not an ancestor of ``j``.

    Results are memoised in ``lT._tmp_parenting`` (a temporary dict that is
    deleted by :data:`parenting` after the full computation).

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    i : int
        Id of the ancestor candidate node.
    j : int
        Id of the descendant candidate node. Must satisfy ``lT.time[i] <= lT.time[j]``.

    Returns
    -------
    float
        Signed ancestor distance, or ``np.inf`` if ``i`` is not an ancestor
        of ``j``.
    """
    if (i, j) not in lT._tmp_parenting:
        if i == j:  # the distance to the node itself is 0
            lT._tmp_parenting[(i, j)] = 0
            lT._parenting[i, j] = lT._tmp_parenting[(i, j)]

        # j and i are not connected so the distance is inf
        elif not lT._predecessor[j]:
            lT._tmp_parenting[(i, j)] = np.inf
        else:  # the distance between i and j is the distance between i and pred(j) + 1
            lT._tmp_parenting[(i, j)] = _m(lT, i, lT._predecessor[j][0]) + 1
            lT._parenting[i, j] = lT._tmp_parenting[(i, j)]
            lT._parenting[j, i] = -lT._tmp_parenting[(i, j)]
    return lT._tmp_parenting[(i, j)]


@property
def parenting(lT: LineageTree):
    """Sparse signed ancestor-distance matrix between all pairs of nodes.

    ``parenting[i, j]`` is positive when ``i`` is a strict ancestor of ``j``
    (value equals the number of edges on the path), negative when ``j`` is an
    ancestor of ``i``, and zero when ``i == j``. The entry is absent (zero in
    the sparse representation) when ``i`` and ``j`` are not on the same root-
    to-leaf path.

    The matrix is computed on first access and stored in ``lT._parenting``
    as a :class:`scipy.sparse.dok_array` of shape
    ``(max_node_id + 1, max_node_id + 1)``.

    .. warning::
        For trees with large node IDs the matrix can be very large in memory
        even though it is sparse.
    """
    if not hasattr(lT, "_parenting"):
        lT._parenting = dok_array((max(lT.nodes) + 1,) * 2)
        lT._tmp_parenting = {}
        for i, j in combinations(lT.nodes, 2):
            if lT._time[j] < lT.time[i]:
                i, j = j, i
            lT._tmp_parenting[(i, j)] = _m(lT, i, j)
        del lT._tmp_parenting
    return lT._parenting


@property
def temporal(lT: LineageTree) -> bool:
    """Whether the tree structure encodes a temporal dimension.

    ``True`` for the standard use-case (cell tracking over time). ``False``
    for static trees such as neuron morphologies loaded via
    :func:`read_from_swc`.
    """
    if not hasattr(lT, "_temporal"):
        lT._temporal = True
    return lT._temporal
