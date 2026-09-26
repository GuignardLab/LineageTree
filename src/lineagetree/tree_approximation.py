"""Tree styles: simplified versions of subtrees used for tree comparison.

Comparing whole subtrees node by node is exact but slow. A tree style turns
the subtree spawned by a node into a smaller tree, and defines the cost of
matching, inserting or deleting its nodes. The built-in styles are listed in
[`tree_style`][lineagetree.tree_approximation.tree_style]; custom styles
subclass
[`TreeApproximationTemplate`][lineagetree.tree_approximation.TreeApproximationTemplate].
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .lineage_tree import LineageTree


class TreeApproximationTemplate(ABC):
    """Base class of the tree styles used to compare subtrees.

    To create a new style, subclass this class or one of the built-in styles
    and pass the class itself as the ``style`` argument of the comparison
    methods, e.g.
    [`unordered_tree_edit_distance`][lineagetree.LineageTree.unordered_tree_edit_distance].

    A style implements four methods:

    - [`get_tree`][lineagetree.tree_approximation.TreeApproximationTemplate.get_tree]
      builds the simplified tree: an adjacency dictionary and the duration
      of each of its nodes;
    - [`delta`][lineagetree.tree_approximation.TreeApproximationTemplate.delta]
      gives the cost of matching two nodes, or of inserting or deleting one;
    - [`get_norm`][lineagetree.tree_approximation.TreeApproximationTemplate.get_norm]
      gives the size of a tree, used to normalise the distance;
    - [`handle_resolutions`][lineagetree.tree_approximation.TreeApproximationTemplate.handle_resolutions]
      gives the ``time_scale`` of each tree when comparing datasets with
      different time resolutions.

    The tree is built when the object is created and stored in ``tree``, and
    its ``edist`` version in ``edist``.
    """

    def __init__(
        self,
        lT: LineageTree,
        root: int,
        downsample: int | None = None,
        end_time: int | None = None,
        time_scale: int = 1,
    ):
        """Initialise the tree approximation for the subtree rooted at ``root``.

        Parameters
        ----------
        lT : LineageTree
            The source lineage tree.
        root : int
            Id of the node that is the root of the subtree to approximate.
        downsample : int, optional
            Downsampling factor (used by ``downsample_tree``).
        end_time : int, optional
            Last time point to include in the approximation. Defaults to
            ``lT.t_e``.
        time_scale : int, default=1
            Scaling factor applied to node durations, used to align trees
            sampled at different time resolutions. 0 or None is read as 1.

        Raises
        ------
        Exception
            If ``time_scale <= 0``.
        """
        self.lT: LineageTree = lT
        self.internal_ids = max(self.lT.nodes)
        self.root: int = root
        self.downsample: int = downsample
        self.end_time: int = end_time if end_time else self.lT.t_e
        self.time_scale: int = int(time_scale) if time_scale else 1
        if time_scale <= 0:
            raise Exception("Please use a valid time_scale (Larger than 0)")
        self.tree: tuple = self.get_tree()
        self.edist = self._edist_format(self.tree[0])

    def get_next_id(self) -> int:
        """Return the next available internal node id for synthetic nodes.

        The counter starts above the maximum real node id so that synthetic
        nodes created during tree construction never collide with real ones.

        Returns
        -------
        int
            A unique id larger than any existing node id.
        """
        self.internal_ids += 1
        return self.internal_ids

    @staticmethod
    @abstractmethod
    def handle_resolutions(
        time_resolution1: float | int,
        time_resolution2: float | int,
        gcd: int,
        downsample: int,
    ) -> tuple[int | float, int | float]:
        """Compute the time scale of each tree for a cross-dataset comparison.

        Used by
        [`LineageTreeManager`][lineagetree.LineageTreeManager] to compare
        datasets recorded at different time resolutions: the returned values
        are passed as ``time_scale`` to the two trees.

        Parameters
        ----------
        time_resolution1 : int or float
            Time resolution of the first dataset, as stored in
            ``lT._time_resolution`` (ten times ``lT.time_resolution``).
        time_resolution2 : int or float
            Time resolution of the second dataset, stored the same way.
        gcd : int
            Greatest common divisor of the time resolutions of all the
            datasets of the manager.
        downsample : int
            Downsampling factor.

        Returns
        -------
        int or float
            The time scale of the first tree.
        int or float
            The time scale of the second tree.
        """

    @abstractmethod
    def get_tree(self) -> tuple[dict, dict]:
        """Build the simplified tree spawned by ``self.root``.

        Returns
        -------
        dict of {int: list of int}
            Adjacency dictionary of the simplified tree. Its ids are node
            ids of the original tree, usually the first node of each chain
            (``self.root`` for the first one).
        dict of {int: float}
            Duration of each node of the simplified tree.
        """

    @abstractmethod
    def delta(
        self,
        x: int,
        y: int,
        corres1: dict[int, int],
        corres2: dict[int, int],
        times1: dict[int, float],
        times2: dict[int, float],
    ) -> int | float:
        """Return the cost of matching, inserting or deleting nodes.

        ``edist`` calls this function with ``x`` or ``y`` set to None for a
        deletion or an insertion. The default implementation, used by the
        ``simple`` style, costs a deletion or insertion the duration of the
        node and a match the difference of the two durations.

        Parameters
        ----------
        x : int or None
            Node of the first tree, as numbered by edist, or None when ``y``
            is inserted.
        y : int or None
            Node of the second tree, as numbered by edist, or None when ``x``
            is deleted.
        corres1 : dict
            Dictionary mapping ``x`` ids to the corresponding id in the
            original tree.
        corres2 : dict
            Dictionary mapping ``y`` ids to the corresponding id in the
            original tree.
        times1 : dict
            Durations of the nodes of the first tree, from
            [`get_tree`][lineagetree.tree_approximation.TreeApproximationTemplate.get_tree].
        times2 : dict
            Durations of the nodes of the second tree.

        Returns
        -------
        int or float
            The cost of the operation.
        """
        if x is None and y is None:
            return 0
        if x is None:
            return times2[corres2[y]]
        if y is None:
            return times1[corres1[x]]
        len_x = times1[corres1[x]]
        len_y = times2[corres2[y]]
        return np.abs(len_x - len_y)

    @abstractmethod
    def get_norm(self, root: int) -> int | float:
        """Return the size of a tree, used to normalise the edit distance.

        Parameters
        ----------
        root : int
            The node spawning the subtree.

        Returns
        -------
        int or float
            The size of the subtree, measured in a way that matches the
            costs of ``delta`` (e.g. the cost of deleting the whole tree).
        """

    def _edist_format(
        self, adj_dict: dict
    ) -> tuple[list, list[list], dict[int, int]]:
        """Convert an adjacency dictionary to the format used by edist.

        Parameters
        ----------
        adj_dict : dict of {int: list of int}
            The adjacency dictionary produced by ``get_tree``.

        Returns
        -------
        list of int
            The nodes, numbered from 0 in depth-first pre-order.
        list of list of int
            The successors of each node, in the same numbering.
        dict of {int: int}
            Maps the edist numbering back to the node ids of ``adj_dict``.
        """
        inv_adj = {vi: k for k, v in adj_dict.items() for vi in v}
        roots = set(adj_dict).difference(inv_adj)
        nid2list = {}
        list2nid = {}
        nodes = []
        adj_list = []
        curr_id = 0
        for r in roots:
            to_do = [r]
            while to_do:
                curr = to_do.pop(0)
                nid2list[curr] = curr_id
                list2nid[curr_id] = curr
                nodes.append(curr_id)
                to_do = adj_dict.get(curr, []) + to_do
                curr_id += 1
            adj_list = [
                [nid2list[d] for d in adj_dict.get(list2nid[_id], [])]
                for _id in nodes
            ]
        return nodes, adj_list, list2nid


class mini_tree(TreeApproximationTemplate):
    """Style that keeps only the branching pattern.

    Each chain becomes one node and durations are ignored: two trees are at
    distance 0 when they divide in the same pattern. Extremely fast; useful
    for comparing synchronously developing cells, and for testing.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def handle_resolutions(
        time_resolution1: float | int,
        time_resolution2: float | int,
        gcd,
        downsample: int,
    ) -> tuple[int | float, int | float]:
        """Return ``(1, 1)``: durations are ignored."""
        return (1, 1)

    def get_tree(self):
        """Build a tree with one node per chain, without durations.

        Returns
        -------
        dict of {int: list of int}
            Adjacency dictionary, keyed by the first node of each chain.
        None
            This style has no durations.
        """
        if self.end_time is None:
            self.end_time = self.lT.t_e
        out_dict = {}
        self.times = {}
        to_do = [self.root]
        while to_do:
            current = to_do.pop()
            cycle = np.array(self.lT.get_successors(current))
            cycle_times = np.array([self.lT.time[c] for c in cycle])
            cycle = cycle[cycle_times <= self.end_time]
            if cycle.size:
                _next = list(self.lT.successor[cycle[-1]])
                if 1 < len(_next):
                    out_dict[current] = _next
                    to_do.extend(_next)
                else:
                    out_dict[current] = []
        self.length = len(out_dict)
        return out_dict, None

    def get_norm(self, root) -> int:
        """Return the number of chains of the subtree spawned by ``root``."""
        return len(
            self.lT.get_all_chains_of_subtree(root, end_time=self.end_time)
        )

    def _edist_format(self, adj_dict: dict):
        return super()._edist_format(adj_dict)

    def delta(self, x, y, corres1, corres2, times1, times2):
        """Cost 1 to insert or delete a node, 0 to match two nodes."""
        if x is None and y is None:
            return 0
        if x is None:
            return 1
        if y is None:
            return 1
        return 0


class simple_tree(TreeApproximationTemplate):
    """Style where each chain becomes one node weighted by its length.

    The default style. Matching two chains costs the difference of their
    lengths, and inserting or deleting a chain costs its length. Fast, but
    imprecise for small trees (the recommended tree height is at least 100
    time points).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def handle_resolutions(
        time_resolution1: float | int,
        time_resolution2: float | int,
        gcd: int,
        downsample: int,
    ) -> tuple[int | float, int | float]:
        """Scale each tree by its own time resolution.

        Chain lengths are then multiplied by the time resolution of their
        dataset, so that they are comparable across datasets.
        """
        return (time_resolution1, time_resolution2)

    def get_tree(self) -> tuple[dict, dict]:
        """Build a tree with one node per chain.

        Chains are cut at ``end_time``.

        Returns
        -------
        dict of {int: list of int}
            Adjacency dictionary, keyed by the first node of each chain.
        dict of {int: int}
            Number of nodes of each chain, times ``time_scale``.
        """
        if self.end_time is None:
            self.end_time = self.lT.t_e
        out_dict = {}
        self.times = {}
        to_do = [self.root]
        while to_do:
            current = to_do.pop()
            cycle = np.array(self.lT.get_successors(current))
            cycle_times = np.array([self.lT.time[c] for c in cycle])
            cycle = cycle[cycle_times <= self.end_time]
            if cycle.size:
                _next = list(self.lT.successor[cycle[-1]])
                if len(_next) > 1 and self.lT.time[cycle[-1]] < self.end_time:
                    out_dict[current] = _next
                    to_do.extend(_next)
                else:
                    out_dict[current] = []
            self.times[current] = len(cycle) * self.time_scale
        return out_dict, self.times

    def delta(self, x, y, corres1, corres2, times1, times2):
        """Cost of the length difference for a match, the length otherwise."""
        return super().delta(x, y, corres1, corres2, times1, times2)

    def get_norm(self, root) -> int:
        """Return the number of nodes of the subtree, times ``time_scale``.

        This is the cost of deleting the whole subtree.
        """
        return (
            len(self.lT.get_subtree_nodes(root, end_time=self.end_time))
            * self.time_scale
        )


class downsample_tree(TreeApproximationTemplate):
    """Style that keeps one time point every ``downsample`` time points.

    Each kept node costs 1 to insert or delete, and 0 to match. More precise
    than the chain-based styles and faster than ``full``.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.downsample == 0:
            raise Exception("Please use a valid downsampling rate")
        if self.downsample == 1:
            warnings.warn(
                "Downsampling rate of 1 is identical to the full tree.",
                stacklevel=1,
            )

    @staticmethod
    def handle_resolutions(
        time_resolution1: float | int,
        time_resolution2: float | int,
        gcd: int,
        downsample: int,
    ) -> tuple[int | float, int | float]:
        """Compute the time scale of each tree from ``downsample``.

        Raises
        ------
        Exception
            If ``downsample`` is not a multiple of the least common multiple
            of the two time resolutions.
        """
        lcm = time_resolution1 * time_resolution2 / gcd
        if downsample % (lcm / 10) != 0:
            raise Exception(
                f"Use a valid downsampling rate (multiple of {lcm/10})"
            )
        return (
            downsample / (time_resolution2 / 10),
            downsample / (time_resolution1 / 10),
        )

    def get_tree(self) -> tuple[dict, dict]:
        """Build a tree that keeps one time point every ``downsample``.

        Returns
        -------
        dict of {int: list of int}
            Adjacency dictionary linking each kept node to its descendants
            ``downsample`` time points later.
        dict of {int: int}
            Duration of each kept node, always 1.
        """
        self.out_dict = {}
        self.times = {}
        to_do = [self.root]
        while to_do:
            current = to_do.pop()
            _next = self.lT.nodes_at_t(
                r=current,
                t=self.lT.time[current] + (self.downsample / self.time_scale),
            )
            if _next == [current]:
                _next = None
            if _next and self.lT.time[_next[0]] <= self.end_time:
                self.out_dict[current] = _next
                to_do.extend(_next)
            else:
                self.out_dict[current] = []
            self.times[current] = 1  # self.downsample
        return self.out_dict, self.times

    def get_norm(self, root) -> float:  ###Temporary###
        """Return the number of kept nodes in the subtree of ``root``."""
        return len(
            downsample_tree(
                lT=self.lT,
                root=root,
                downsample=self.downsample,
                end_time=self.end_time,
                time_scale=self.time_scale,
            ).out_dict
        )

    def delta(self, x, y, corres1, corres2, times1, times2):
        """Cost 1 to insert or delete a node, 0 to match two nodes."""
        if x is None and y is None:
            return 0
        if x is None:
            return 1
        if y is None:
            return 1
        return 0


class normalized_simple_tree(simple_tree):
    """Style like ``simple`` where the cost of a match is relative.

    Identical to ``simple`` except that matching two chains costs
    ``|len_x - len_y| / (len_x + len_y)`` instead of the raw difference, so
    each match costs less than 1, and inserting or deleting a chain costs 1.
    This makes the distance less sensitive to the absolute duration of
    chains and more sensitive to their relative lengths.

    The norm is the number of chains in the subtree (not the total number of
    nodes), as for ``mini``.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def delta(self, x, y, corres1, corres2, times1, times2):
        """Cost 1 to insert or delete, the relative length difference to match."""
        if x is None and y is None:
            return 0
        if x is None:
            return 1
        if y is None:
            return 1
        return abs(times1[corres1[x]] - times2[corres2[y]]) / (
            times1[corres1[x]] + times2[corres2[y]]
        )

    def get_norm(self, root) -> int:
        """Return the number of chains of the subtree spawned by ``root``."""
        return len(
            self.lT.get_all_chains_of_subtree(root, end_time=self.end_time)
        )


class full_tree(TreeApproximationTemplate):
    """Style that keeps every node, without approximation.

    Each node costs 1 to insert or delete, and 0 to match. Exact, but heavy
    on memory and slow. Not recommended for use in napari.
    """

    def _edist_format(
        self, adj_dict: dict
    ) -> tuple[list, list[list], dict[int, int]]:
        """Format the custom tree style to the format needed by edist.

        Unlike the default version, nodes added to rescale time (see
        ``time_scale``) are mapped back to the node they were added for.

        Parameters
        ----------
        adj_dict : dict
            The adjacency dictionary produced by :meth:`get_tree`.

        Returns
        -------
        list of int
            The list of the new nodes to be used for edist.
        list of list
            The adjacency list of these nodes.
        dict of {int: int}
            The correspondence between the nodes used in edist and the
            LineageTree.
        """
        inv_adj = {vi: k for k, v in adj_dict.items() for vi in v}
        roots = set(adj_dict).difference(inv_adj)
        nid2list = {}
        list2nid = {}
        nodes = []
        adj_list = []
        curr_id = 0
        to_update = {}
        for r in roots:
            to_do = [r]
            while to_do:
                curr = to_do.pop(0)
                nid2list[curr] = curr_id
                list2nid[curr_id] = curr
                if curr in self.corres_added_nodes:
                    to_update[curr_id] = self.corres_added_nodes[curr]
                nodes.append(curr_id)
                to_do = adj_dict.get(curr, []) + to_do
                curr_id += 1
            adj_list = [
                [nid2list[d] for d in adj_dict.get(list2nid[_id], [])]
                for _id in nodes
            ]
            list2nid.update(to_update)
        return nodes, adj_list, list2nid

    @staticmethod
    def handle_resolutions(
        time_resolution1: float | int,
        time_resolution2: float | int,
        gcd: int,
        downsample: int,
    ) -> tuple[int | float, int | float]:
        """Compute how many times each node of each tree is repeated.

        ``(1, 1)`` when both datasets have the same time resolution.
        """
        if time_resolution1 == time_resolution2:
            return (1, 1)
        lcm = time_resolution1 * time_resolution2 / gcd
        return (
            lcm / time_resolution2,
            lcm / time_resolution1,
        )

    def get_tree(self) -> tuple[dict, dict]:
        """Build a tree with every node of the subtree, up to ``end_time``.

        When ``time_scale`` is larger than 1, each node is followed by
        ``time_scale - 1`` added nodes.

        Returns
        -------
        dict of {int: list of int}
            Adjacency dictionary of the tree.
        dict
            Empty: every node has the same duration.
        """
        self.out_dict = {}
        self.times = {}
        self.corres_added_nodes = {}
        to_do = [self.root]
        while to_do:
            current = to_do.pop()
            _next = list(self.lT.successor[current])
            if _next and self.lT.time[_next[0]] <= self.end_time:
                if self.time_scale > 1:
                    tmp_cur = current
                    for _ in range(self.time_scale - 1):
                        next_id = self.get_next_id()
                        self.out_dict[current] = [next_id]
                        current = int(next_id)
                        self.corres_added_nodes[current] = tmp_cur
                self.out_dict[current] = _next
                to_do.extend(_next)
            else:
                if self.time_scale > 1:
                    tmp_cur = current
                    for _ in range(self.time_scale - 1):
                        next_id = self.get_next_id()
                        self.out_dict[current] = [next_id]
                        current = int(next_id)
                        self.corres_added_nodes[current] = tmp_cur
                self.out_dict[current] = []
        return self.out_dict, self.times

    def get_norm(self, root) -> int:
        """Return the number of nodes of the subtree, times ``time_scale``."""
        return (
            len(self.lT.get_subtree_nodes(root, end_time=self.end_time))
            * self.time_scale
        )

    def delta(self, x, y, corres1, corres2, times1, times2):
        """Cost 1 to insert or delete a node, 0 to match two nodes."""
        if x is None and y is None:
            return 0
        if x is None:
            return 1
        if y is None:
            return 1
        return 0


class tree_style(Enum):
    """The built-in tree styles, by the name used as ``style`` argument.

    From the fastest and least precise to the slowest and exact:

    Attributes
    ----------
    mini : mini_tree
        Each chain becomes a node of cost 1; only the branching pattern
        counts.
    simple : simple_tree
        Each chain becomes a node weighted by its length. The default.
    normalized_simple : normalized_simple_tree
        Like ``simple``, but the cost of a match is relative (below 1).
    downsampled : downsample_tree
        Keeps one time point every ``downsample`` time points.
    full : full_tree
        No approximation; every node is kept.
    """

    mini = mini_tree
    simple = simple_tree
    normalized_simple = normalized_simple_tree
    downsampled = downsample_tree
    full = full_tree

    @classmethod
    def list_names(cls) -> list[str]:
        """Return a list of all available style names.

        Returns
        -------
        list of str
            Names of all members of this enum.
        """
        return [style.name for style in cls]
