from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from  LineageTree import lineageTree


class abstract_trees(ABC):
    def __init__(
        self, lT: lineageTree, root: int, node_length: int, end_time: int
    ):
        self.lT = lT
        self.root = root
        self.node_length = node_length
        self.end_time = end_time

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
        pass

    @abstractmethod
    def delta(self, x, y, corres1, corres2, times1, times2):
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
            (int|float): The number of nodes of each tree.
        """
        pass

    def _edist_format(self, adj_dict: dict):
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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_tree(self):
        return self

    def get_norm(self):
        return super().get_norm()

    def _edist_format(self, adj_dict: dict):
        return super()._edist_format(adj_dict)


class simple_tree(abstract_trees):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class fragmented_tree(abstract_trees):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_tree(self):
        if self.end_time is None:
            end_time = self.lT.t_e
        self.out_dict = {}
        self.times = {}
        to_do = [self.root]
        if not isinstance(self.node_length, list):
            self.node_length = list(self.node_length)
        while to_do:
            current = to_do.pop()
            cycle = np.array(
                self.lT.get_successors(current, end_time=end_time)
            )
            if 0 < cycle.size:
                cumul_sum_of_nodes = np.cumsum(self.node_length) * 2 + 1
                max_number_fragments = len(
                    cumul_sum_of_nodes[cumul_sum_of_nodes < len(cycle)]
                )
                if max_number_fragments > 0:
                    current_node_lengths = self.node_length[
                        :max_number_fragments
                    ].copy()
                    length_middle_node = (
                        len(cycle) - sum(current_node_lengths) * 2
                    )
                    times_tmp = (
                        current_node_lengths
                        + [length_middle_node]
                        + current_node_lengths[::-1]
                    )
                    pos_all_nodes = np.concatenate(
                        [[0], np.cumsum(times_tmp[:-1])]
                    )
                    track = cycle[pos_all_nodes]
                    self.out_dict.update(
                        {k: [v] for k, v in zip(track[:-1], track[1:])}
                    )
                    self.times.update(zip(track, times_tmp))
                else:
                    for i, cell in enumerate(cycle[:-1]):
                        self.out_dict[cell] = [cycle[i + 1]]
                        self.times[cell] = 1
                current = cycle[-1]
                _next = self.lT[current]
                self.times[current] = 1
                if _next and self.lT.time[_next[0]] <= end_time:
                    to_do.extend(_next)
                    self.out_dict[current] = _next
                else:
                    self.out_dict[current] = []

        return self.out_dict, self.times

    def get_norm(self):
        return sum(self.times.values())

    def delta(self, x, y, corres1, corres2, times1, times2):
        return super().delta(x, y, corres1, corres2, times1, times2)


class full_tree(abstract_trees):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class tree_style(Enum):
    mini = mini_tree
    simple = simple_tree
    fragmented = fragmented_tree
    full = full_tree
