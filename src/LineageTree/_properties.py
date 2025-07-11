import warnings
from itertools import combinations
from types import MappingProxyType

import numpy as np
from scipy.sparse import dok_array


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


@property
def successor(self) -> MappingProxyType[int, tuple[int]]:
    """The successor of the tree."""
    if not hasattr(self, "_protected_successor"):
        self._protected_successor = MappingProxyType(self._successor)
    return self._protected_successor


@property
def predecessor(self) -> MappingProxyType[int, tuple[int]]:
    """The predecessor of the tree."""
    if not hasattr(self, "_protected_predecessor"):
        self._protected_predecessor = MappingProxyType(self._predecessor)
    return self._protected_predecessor


@property
def time(self) -> MappingProxyType[int, int]:
    """The time of the tree."""
    if not hasattr(self, "_protected_time"):
        self._protected_time = MappingProxyType(self._time)
    return self._protected_time


@dynamic_property
def t_b(self) -> int:
    """The first timepoint of the tree."""
    return min(self._time.values())


@dynamic_property
def t_e(self) -> int:
    """The last timepoint of the tree."""
    return max(self._time.values())


@dynamic_property
def nodes(self) -> frozenset[int]:
    """Nodes of the tree"""
    return frozenset(self._successor.keys())


@dynamic_property
def number_of_nodes(self) -> int:
    return len(self.nodes)


@dynamic_property
def depth(self) -> dict[int, int]:
    """The depth of each node in the tree."""
    _depth = {}
    for leaf in self.leaves:
        _depth[leaf] = 1
        while leaf in self._predecessor and self._predecessor[leaf]:
            parent = self._predecessor[leaf][0]
            current_depth = _depth.get(parent, 0)
            _depth[parent] = max(_depth[leaf] + 1, current_depth)
            leaf = parent
    for root in self.roots - set(_depth):
        _depth[root] = 1
    return _depth


@dynamic_property
def roots(self) -> frozenset[int]:
    """Set of roots of the tree"""
    return frozenset({s for s, p in self._predecessor.items() if p == ()})


@dynamic_property
def leaves(self) -> frozenset[int]:
    """Set of leaves"""
    return frozenset({p for p, s in self._successor.items() if s == ()})


@dynamic_property
def edges(self) -> tuple[tuple[int, int]]:
    """Set of edges"""
    return tuple((p, si) for p, s in self._successor.items() for si in s)


@property
def labels(self) -> dict[int, str]:
    """The labels of the nodes."""
    if not hasattr(self, "_labels"):
        if hasattr(self, "node_name"):
            self._labels = {
                i: self.node_name.get(i, "Unlabeled") for i in self.roots
            }
        else:
            self._labels = {
                root: "Unlabeled"
                for root in self.roots
                for leaf in self.find_leaves(root)
                if abs(self._time[leaf] - self._time[root])
                >= abs(self.t_e - self.t_b) / 4
            }
    return self._labels


@property
def time_resolution(self) -> float:
    if not hasattr(self, "_time_resolution"):
        self._time_resolution = 0
    return self._time_resolution / 10


@time_resolution.setter
def time_resolution(self, time_resolution) -> None:
    if time_resolution is not None and time_resolution > 0:
        self._time_resolution = int(time_resolution * 10)
    else:
        warnings.warn("Time resolution set to default 0", stacklevel=2)
        self._time_resolution = 0


@dynamic_property
def all_chains(self) -> list[list[int]]:
    """List of all chains in the tree, ordered in depth-first search."""
    return self._compute_all_chains()


@dynamic_property
def time_nodes(self):
    _time_nodes = {}
    for c, t in self._time.items():
        _time_nodes.setdefault(t, set()).add(c)
    return _time_nodes


def _m(self, i, j):
    if (i, j) not in self._tmp_parenting:
        if i == j:  # the distance to the node itself is 0
            self._tmp_parenting[(i, j)] = 0
            self._parenting[i, j] = self._tmp_parenting[(i, j)]
        elif not self._predecessor[
            j
        ]:  # j and i are note connected so the distance if inf
            self._tmp_parenting[(i, j)] = np.inf
        else:  # the distance between i and j is the distance between i and pred(j) + 1
            self._tmp_parenting[(i, j)] = (
                self.m(i, self._predecessor[j][0]) + 1
            )
            self._parenting[i, j] = self._tmp_parenting[(i, j)]
            self._parenting[j, i] = -self._tmp_parenting[(i, j)]
    return self._tmp_parenting[(i, j)]


@property
def parenting(self):
    if not hasattr(self, "_parenting"):
        self._parenting = dok_array((max(self.nodes) + 1,) * 2)
        self._tmp_parenting = {}
        for i, j in combinations(self.nodes, 2):
            if self._time[j] < self.time[i]:
                i, j = j, i
            self._tmp_parenting[(i, j)] = self._m(i, j)
        del self._tmp_parenting
    return self._parenting
