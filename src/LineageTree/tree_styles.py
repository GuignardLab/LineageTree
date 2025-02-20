import warnings
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from LineageTree import lineageTree


class abstract_trees(ABC):
    """Template class to produce different techniques to comapare lineageTrees.
    To add a new technique you need to iherit this class or one of its children
    and add them to the tree_style enum.
    For a class to be valid you need a
    - tree constructor (get_tree) that produces one dictionary that contains
    arbitary unique labels and one dictionary that contains the duration of each node.
    - delta function: A function that handles the cost of comparing nodes to each other.
    - normalization function, a function that returns the length of the tree or any interger.
    """

    def __init__(
        self,
        lT: lineageTree,
        root: int,
        downsample: int,
        end_time: int = None,
        time_scale: int = 1,
    ):
        self.lT: lineageTree = lT
        self.internal_ids = max(self.lT.nodes)
        self.root: int = root
        self.downsample: int = downsample
        self.end_time: int = end_time if end_time else self.lT.t_e
        self.time_scale: int = int(time_scale) if time_scale else 1
        if time_scale <= 0:
            raise Exception("Please used a valid time_scale (Larger than 0)")
        self.tree: tuple = self.get_tree()
        self.edist = self._edist_format(self.tree[0])

    def get_next_id(self):
        self.internal_ids += 1
        return self.internal_ids

    @abstractmethod
    def get_tree(self) -> tuple[dict, dict]:
        """
        Get a tree version of the tree spawned by the node `r`

        Args:
            r (int): root of the tree to spawn
            end_time (int): the last time point to consider
            time_resolution (float): the time between two consecutive time points

        Returns:
            (dict) {m (int): [d1 (int), d2 (int)]}: an adjacency dictionnary
                where the ids are the ids of the cells in the original tree
                at their first time point (except for the cell `r` if it was
                not the first time point).
            (dict) {m (int): duration (float)}: life time duration of the cell `m`
        """

    @abstractmethod
    def delta(self, x, y, corres1, corres2, times1, times2):
        """The distance of two nodes inside a tree. Behaves like a staticmethod.
            The corres1/2 and time1/2 should always be provided and will be handled accordingly by the specific
            delta of each tree style.

        Args:
            x (int): The first node to compare, takes the names provided by the edist.
            y (int): The second node to compare, takes the names provided by the edist
            corres1 (dict): Correspondance between node1 and its name in the real tree.
            corres2 (dict): Correspondance between node2 and its name in the real tree.
            times1 (dict): The dictionary of the branch lengths of the tree that n1 is spawned from.
            times2 (dict): The dictionary of the branch lengths of the tree that n2 is spawned from.

        Returns:
            (int|float): The diatance between these 2 nodes.
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
    def get_norm(self) -> int:
        """
        Returns the valid value for normalizing the edit distance.
        Returns:
            (int|float): The number of nodes of each tree according to each style.
        """

    def _edist_format(self, adj_dict: dict):
        """Formating the custom tree style to the format needed by edist.

        Args:
            adj_dict (dict): _description_

        Returns:
            _type_: _description_
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


class mini_tree(abstract_trees):
    """Each branch is converted to a node of length 1, it is useful for comparing synchronous developing cells, extremely fast.
    Mainly used for testing.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    def get_norm(self) -> int:
        return len(
            self.lT.get_all_branches_of_node(self.root, end_time=self.end_time)
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


class simple_tree(abstract_trees):
    """Each branch is converted to one node with length the same as the life cycle of the cell.
    This method is fast, but imprecise, especialy for small trees (recommended height of the trees should be 100 at least).
    Use with CAUTION.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    def get_norm(self) -> int:
        return (
            len(self.lT.get_sub_tree(self.root, end_time=self.end_time))
            * self.time_scale
        )


class downsample_tree(abstract_trees):
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

    def get_norm(self) -> int:
        return len(self.times.values()) * self.downsample / self.time_scale

    def delta(self, x, y, corres1, corres2, times1, times2):
        if x is None and y is None:
            return 0
        if x is None:
            return self.downsample
        if y is None:
            return self.downsample
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

    def get_norm(self) -> int:
        return len(
            self.lT.get_all_branches_of_node(self.root, end_time=self.end_time)
        )


class full_tree(abstract_trees):
    """No approximations the whole tree is used here. Perfect accuracy, but heavy on ram and speed.
    Not recommended to use on napari.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_tree(self) -> tuple[dict, dict]:
        self.out_dict = {}
        self.times = {}
        to_do = [self.root]
        while to_do:
            current = to_do.pop()
            _next = list(self.lT.successor[current])
            if _next and self.lT.time[_next[0]] <= self.end_time:
                if self.time_scale > 1:
                    for _ in range(self.time_scale - 1):
                        next_id = self.get_next_id()
                        self.out_dict[current] = [next_id]
                        current = int(next_id)
                self.out_dict[current] = _next
                to_do.extend(_next)
            else:
                if self.time_scale > 1:
                    for _ in range(self.time_scale - 1):
                        next_id = self.get_next_id()
                        self.out_dict[current] = [next_id]
                        current = int(next_id)
                self.out_dict[current] = []
        self.times = {n_id: 1 for n_id in self.out_dict}
        return self.out_dict, self.times

    def get_norm(self) -> int:
        return len(self.out_dict) * self.time_scale

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
