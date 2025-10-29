from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .lineage_tree import LineageTree


class TreeApproximationTemplate(ABC):
    """Template class to produce different tree styles to comapare LineageTrees.
    To add a new style you need to inherit this class or one of its children
    and add them to the tree_style enum, or use it immediately on the function called.
    The main products of this class are:
    - tree constructor (get_tree) that produces one dictionary that contains
    arbitary unique labels and one dictionary that contains the duration of each node.
    - delta function: A function that handles the cost of comparing nodes to each other.
    - normalization function, a function that returns the length of the tree or any interger.
    """

    def __init__(
        self,
        lT: LineageTree,
        root: int,
        downsample: int | None = None,
        end_time: int | None = None,
        time_scale: int = 1,
    ):
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
        """Handle different time resolutions.

        Parameters
        ----------
        time_resolution1 : int or float
            Time resolution of the first dataset. (Extracted from lT._time_resolution)
        time_resolution2 : int or float
            Time resolution of the second dataset. (Extracted from lT._time_resolution)

        Returns
        -------
        int or float
            The time resolution fix for the first dataset
        int or float
            The time resolution fix for the second dataset
        """

    @abstractmethod
    def get_tree(self) -> tuple[dict, dict]:
        """
        Get a tree version of the tree spawned by the node `r`

        Returns
        -------
        dict mapping an int to a list of int
            an adjacency dictionnary where the ids are the ids of the
            cells in the original tree at their first time point
            (except for the cell `r` if it was not the first time point).
        dict mapping an int to a float
            life time duration of the key cell `m`
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
        """The distance of two nodes inside a tree. Behaves like a staticmethod.
            The corres1/2 and time1/2 should always be provided and will be handled accordingly by the specific
            delta of each tree style.

        Parameters
        ----------
        x : int
            The first node to compare, takes the names provided by the edist.
        y : int
            The second node to compare, takes the names provided by the edist
        corres1 : dict
            Dictionary mapping node1 ids to the corresponding id in the original tree.
        corres2 : dict
            Dictionary mapping node2 ids to the corresponding id in the original tree.
        times1 : dict
            The dictionary of the chain lengths of the tree that n1 is spawned from.
        times2 : dict
            The dictionary of the chain lengths of the tree that n2 is spawned from.

        Returns
        -------
        int or float
            The distance between 'x' and 'y'.
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
        """
        Returns the valid value for normalizing the edit distance.

        Parameters
        ----------
        root : int
            The starting node of the subtree.

        Returns
        -------
        int or float
            The number of nodes of each tree according to each style, or the sum of the length of all the nodes in a tree.
        """

    def _edist_format(
        self, adj_dict: dict
    ) -> tuple[list, list[list], dict[int, int]]:
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
    """Each branch is converted to a node of length 1, it is useful for comparing synchronous developing cells, extremely fast.
    Mainly used for testing.
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
        return (1, 1)

    def get_tree(self):
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
        return len(
            self.lT.get_all_chains_of_subtree(root, end_time=self.end_time)
        )

    def _edist_format(self, adj_dict: dict):
        return super()._edist_format(adj_dict)

    def delta(self, x, y, corres1, corres2, times1, times2):
        if x is None and y is None:
            return 0
        if x is None:
            return 1
        if y is None:
            return 1
        return 0


class simple_tree(TreeApproximationTemplate):
    """Each branch is converted to one node with length the same as the life cycle of the cell.
    This method is fast, but imprecise, especialy for small trees (recommended height of the trees should be 100 at least).
    Use with CAUTION.
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
        return (time_resolution1, time_resolution2)

    def get_tree(self) -> tuple[dict, dict]:
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
        return super().delta(x, y, corres1, corres2, times1, times2)

    def get_norm(self, root) -> int:
        return (
            len(self.lT.get_subtree_nodes(root, end_time=self.end_time))
            * self.time_scale
        )


class downsample_tree(TreeApproximationTemplate):
    """Downsamples a tree so every n nodes are being used as one."""

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
        if x is None and y is None:
            return 0
        if x is None:
            return 1
        if y is None:
            return 1
        return 0


class normalized_simple_tree(simple_tree):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def delta(self, x, y, corres1, corres2, times1, times2):
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
        return len(
            self.lT.get_all_chains_of_subtree(root, end_time=self.end_time)
        )


class full_tree(TreeApproximationTemplate):
    """No approximations the whole tree is used here. Perfect accuracy, but heavy on ram and speed.
    Not recommended to use on napari.

    """

    def _edist_format(
        self, adj_dict: dict
    ) -> tuple[list, list[list], dict[int, int]]:
        """Formating the custom tree style to the format needed by edist.
        .. warning:: Modifying this function might break your code.

        Parameters
        ----------
            adj_dict : dict
                The adjacency dictionary produced by 'get_tree'

        Returns
        -------
            list[int]
                The list of the new nodes to be used for edist
            list[list]
                The adjacency list of these nodes
            dict[int,int]
                The correspondance between the nodes used in edist and LineageTree
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
        if time_resolution1 == time_resolution2:
            return (1, 1)
        lcm = time_resolution1 * time_resolution2 / gcd
        return (
            lcm / time_resolution2,
            lcm / time_resolution1,
        )

    def get_tree(self) -> tuple[dict, dict]:
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
        return (
            len(self.lT.get_subtree_nodes(root, end_time=self.end_time))
            * self.time_scale
        )

    def delta(self, x, y, corres1, corres2, times1, times2):
        if x is None and y is None:
            return 0
        if x is None:
            return 1
        if y is None:
            return 1
        return 0


class tree_style(Enum):
    mini = mini_tree
    simple = simple_tree
    normalized_simple = normalized_simple_tree
    downsampled = downsample_tree
    full = full_tree

    @classmethod
    def list_names(self):
        return [style.name for style in self]
