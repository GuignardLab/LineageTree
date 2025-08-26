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
    def __init__(
        self, fget=None, fset=None, fdel=None, doc=None, protected_name=None
    ):
        super().__init__(fget, fset, fdel, doc)
        self.protected_name = protected_name

    def __set_name__(self, owner, name):
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
    """Computes all the chains of a given lineage tree,
    stores it in `lT.all_chains` and returns it.

    Returns
    -------
    tuple of tuple of int
        tuple of chains
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
    """Dictionary that maps a node to its successors"""
    if not hasattr(lT, "_protected_successor"):
        lT._protected_successor = MappingProxyType(lT._successor)
    return lT._protected_successor


@property
def predecessor(lT: LineageTree) -> MappingProxyType[int, tuple[int]]:
    """Dictionary that maps a node to its predecessors"""
    if not hasattr(lT, "_protected_predecessor"):
        lT._protected_predecessor = MappingProxyType(lT._predecessor)
    return lT._protected_predecessor


@property
def time(lT: LineageTree) -> MappingProxyType[int, int]:
    """Dictionary that maps a node to its time"""
    if not hasattr(lT, "_protected_time"):
        lT._protected_time = MappingProxyType(lT._time)
    return lT._protected_time


@dynamic_property
def t_b(lT: LineageTree) -> int:
    """The first timepoint of the lineage tree"""
    return min(lT._time.values())


@dynamic_property
def t_e(lT: LineageTree) -> int:
    """The last timepoint of the lineage tree"""
    return max(lT._time.values())


@dynamic_property
def nodes(lT: LineageTree) -> frozenset[int]:
    """Set of node ids of the lineage tree"""
    return frozenset(lT._successor.keys())


@dynamic_property
def number_of_nodes(lT: LineageTree) -> int:
    """Number of nodes in the lineage tree"""
    return len(lT.nodes)


@dynamic_property
def depth(lT: LineageTree) -> dict[int, int]:
    """The depth of each node in the lineage tree"""
    _depth = {}
    for leaf in lT.leaves:
        _depth[leaf] = 1
        while leaf in lT._predecessor and lT._predecessor[leaf]:
            parent = lT._predecessor[leaf][0]
            current_depth = _depth.get(parent, 0)
            _depth[parent] = max(_depth[leaf] + 1, current_depth)
            leaf = parent
    for root in lT.roots - set(_depth):
        _depth[root] = 1
    return _depth


@dynamic_property
def roots(lT: LineageTree) -> frozenset[int]:
    """Set of roots of the lineage tree"""
    return frozenset({s for s, p in lT._predecessor.items() if p == ()})


@dynamic_property
def leaves(lT: LineageTree) -> frozenset[int]:
    """Set of leaves of the lineage tree"""
    return frozenset({p for p, s in lT._successor.items() if s == ()})


@dynamic_property
def edges(lT: LineageTree) -> tuple[tuple[int, int]]:
    """Set of edges of the lineage tree"""
    return tuple((p, si) for p, s in lT._successor.items() for si in s)


@property
def labels(lT: LineageTree) -> dict[int, str]:
    """Dictionary that maps a node to its label"""
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
    """Time resolution of the lineage tree"""
    if not hasattr(lT, "_time_resolution"):
        lT._time_resolution = 0
    return lT._time_resolution / 10


@time_resolution.setter
def time_resolution(lT, time_resolution) -> None:
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


def _m(lT: LineageTree, i, j):
    if (i, j) not in lT._tmp_parenting:
        if i == j:  # the distance to the node itlT is 0
            lT._tmp_parenting[(i, j)] = 0
            lT._parenting[i, j] = lT._tmp_parenting[(i, j)]

        # j and i are note connected so the distance if inf
        elif not lT._predecessor[j]:
            lT._tmp_parenting[(i, j)] = np.inf
        else:  # the distance between i and j is the distance between i and pred(j) + 1
            lT._tmp_parenting[(i, j)] = _m(lT, i, lT._predecessor[j][0]) + 1
            lT._parenting[i, j] = lT._tmp_parenting[(i, j)]
            lT._parenting[j, i] = -lT._tmp_parenting[(i, j)]
    return lT._tmp_parenting[(i, j)]


@property
def parenting(lT: LineageTree):
    if not hasattr(lT, "_parenting"):
        lT._parenting = dok_array((max(lT.nodes) + 1,) * 2)
        lT._tmp_parenting = {}
        for i, j in combinations(lT.nodes, 2):
            if lT._time[j] < lT.time[i]:
                i, j = j, i
            lT._tmp_parenting[(i, j)] = _m(lT, i, j)
        del lT._tmp_parenting
    return lT._parenting
