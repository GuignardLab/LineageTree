from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from LineageTree import lineageTree


class abstract_trees(ABC):
    def __init__(
        self, lT: lineageTree, root: int, downsample: int, end_time: int
    ):
        self.lT = lT
        self.root = root
        self.downsample = downsample
        self.end_time = end_time if end_time else self.lT.t_e
        self.tree = self.get_tree()
        self.edist = self._edist_format(self.tree[0])

    @abstractmethod
    def get_tree(self):
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
    def get_norm(self):
        """
        Returns the valid value for normalizing the edit distance.
        Returns:
            (int|float): The number of nodes of each tree according to each style.
        """

    def _edist_format(self, adj_dict: dict):
        """Formating the custom tree style to the format needed by edist.
        SHOULD NOT BE CHANGED.

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
                _next = self.lT[cycle[-1]]
                if len(_next) > 1:
                    out_dict[current] = _next
                    to_do.extend(_next)
                else:
                    out_dict[current] = []
        self.length = len(out_dict)
        return out_dict, None

    def get_norm(self):
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
                _next = self.lT[cycle[-1]]
                if 1 < len(_next) and self.lT.time[cycle[-1]] < self.end_time:
                    out_dict[current] = _next
                    to_do.extend(_next)
                else:
                    out_dict[current] = []
            self.times[current] = len(
                cycle
            )  # * time_resolution will be fixed when working on registered trees.
        return out_dict, self.times

    def delta(self, x, y, corres1, corres2, times1, times2):
        return super().delta(x, y, corres1, corres2, times1, times2)

    def get_norm(self):
        return len(self.lT.get_sub_tree(self.root, end_time=self.end_time))


class downsample_tree(abstract_trees):
    """Downsamples a tree so every n nodes are being used as one."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_tree(self):
        self.out_dict = {}
        self.times = {}
        to_do = [self.root]
        while to_do:
            current = to_do.pop()
            _next = self.lT.get_cells_at_t_from_root(
                current, self.lT.time[current] + self.downsample
            )
            if _next == [current]:
                _next = None
            if _next and self.lT.time[_next[0]] <= self.end_time:
                self.out_dict[current] = _next
                to_do.extend(_next)
            else:
                self.out_dict[current] = []
            self.times[current] = self.downsample
        return self.out_dict, self.times

    def get_norm(self):
        return sum(self.times.values())

    def delta(self, x, y, corres1, corres2, times1, times2):
        if x is None and y is None:
            return 0
        if x is None:
            return self.downsample
        if y is None:
            return self.downsample
        return 0


class normalized_tree(simple_tree):
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

    def get_norm(self):
        return len(
            self.lT.get_all_branches_of_node(self.root, end_time=self.end_time)
        )


class full_tree(abstract_trees):
    """No approximations the whole tree is used here. Perfect accuracy, but heavy on ram and speed.
    Not recommended to use on napari.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_tree(self) -> dict:
        self.out_dict = {}
        self.times = {}
        to_do = [self.root]
        while to_do:
            current = to_do.pop()
            _next = self.lT.successor.get(current, [])
            if _next and self.lT.time[_next[0]] <= self.end_time:
                self.out_dict[current] = _next
                to_do.extend(_next)
            else:
                self.out_dict[current] = []
            self.times[current] = 1
        return self.out_dict, self.times

    def get_norm(self):
        return len(self.lT.get_sub_tree(self.root, end_time=self.end_time))

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
    normalized = normalized_tree
    downsampled = downsample_tree
    full = full_tree

    @classmethod
    def list_names(self):
        return [style.name for style in self]
