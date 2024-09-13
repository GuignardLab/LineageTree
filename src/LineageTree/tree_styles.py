from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

from LineageTree import lineageTree


class abstract_trees(ABC):
    def __init__(
        self, lT: lineageTree, root: int, node_length: int, end_time: int
    ):
        self.lT = lT
        self.root = root
        self.node_length = node_length
        self.end_time = end_time
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
        return self.length

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
            self.times[current] = len(
                cycle
            )  # * time_resolution will be fixed when working on registered trees.
        return out_dict, self.times

    def delta(self, x, y, corres1, corres2, times1, times2):
        return super().delta(x, y, corres1, corres2, times1, times2)

    def get_norm(self):
        return len(self.lT.get_sub_tree(self.root)) #sum(self.times.values())


class fragmented_tree(abstract_trees):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_tree(self):
        if self.end_time is None:
            self.end_time = self.lT.t_e
        self.out_dict = {}
        self.times = {}
        to_do = [self.root]
        if not isinstance(self.node_length, list):
            self.node_length = list(self.node_length)
        while to_do:
            current = to_do.pop()
            cycle = np.array(
                self.lT.get_successors(current, end_time=self.end_time)
            )
            if cycle.size > 0:
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
                if _next and self.lT.time[_next[0]] <= self.end_time:
                    to_do.extend(_next)
                    self.out_dict[current] = _next
                else:
                    self.out_dict[current] = []

        return self.out_dict, self.times

    def get_norm(self):
        return len(self.lT.get_sub_tree(self.root))#sum(self.times.values())

    def delta(self, x, y, corres1, corres2, times1, times2):
        return super().delta(x, y, corres1, corres2, times1, times2)


class full_tree(abstract_trees):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class tree_style(Enum):
    mini = mini_tree
    simple = simple_tree
    fragmented = fragmented_tree
    # full = full_tree

    @classmethod
    def list_names(self):
        return [style.name for style in self]


#   #####################################
#       def get_simple_tree(
#       self, r: int, end_time: int = None, time_resolution: int = 1
#   ) -> tuple:
#       """
#       Get a "simple" version of the tree spawned by the node `r`
#       This simple version is just one node per cell (as opposed to
#       one node per cell per time-point). The life time duration of
#       a cell `c` is stored in `self.cycle_time` and return by this
#       function
#
#       Args:
#           r (int): root of the tree to spawn
#           end_time (int): the last time point to consider
#           time_resolution (float): the time between two consecutive time points
#
#       Returns:
#           (dict) {m (int): [d1 (int), d2 (int)]}: a adjacency dictionnary
#               where the ids are the ids of the cells in the original tree
#               at their first time point (except for the cell `r` if it was
#               not the first time point).
#           (dict) {m (int): duration (float)}: life time duration of the cell `m`
#           (dict) {m (int): duration (float)}: life time duration of the cell `m`
#       """
#       if end_time is None:
#           end_time = self.t_e
#       out_dict = {}
#       time = {}
#       to_do = [r]
#       while to_do:
#           current = to_do.pop()
#           cycle = np.array(self.get_successors(current))
#           cycle_times = np.array([self.time[c] for c in cycle])
#           cycle = cycle[cycle_times <= end_time]
#           if cycle.size:
#               _next = self[cycle[-1]]
#               if len(_next) > 1:
#                   out_dict[current] = _next
#                   to_do.extend(_next)
#               else:
#                   out_dict[current] = []
#           time[current] = len(cycle) * time_resolution
#       return out_dict, time
#
#   def get_fragmented_tree(
#       self, r: int = 0, node_lengths=(1, 3, 5, 7), end_time: int = None
#   ) -> tuple:
#       """
#       Get a "fragmented" version of the tree spawned by the node `r`
#       The deafult version is 7 nodes per cell lifetime. The length of each node depends on the node_lengths parameter.
#       Args:
#           r (int): root of the tree to spawn
#           end_time (int): the last time point to consider
#           time_resolution (float): the time between two consecutive time points
#
#       Returns:
#           (dict) {m (int): [d1 (int), d2 (int)]}: a adjacency dictionnary
#               where the ids are the ids of the cells in the original tree
#               at their first time point (except for the cell `r` if it was
#               not the first time point).
#           (dict) {m (int): duration (float)}: life time duration of the cell `m`
#       """
#       if end_time is None:
#           end_time = self.t_e
#       out_dict = {}
#       times = {}
#       to_do = [r]
#       if not isinstance(node_lengths, list):
#           node_lengths = list(node_lengths)
#       while to_do:
#           current = to_do.pop()
#           cycle = np.array(self.get_successors(current, end_time=end_time))
#           if cycle.size > 0:
#               cumul_sum_of_nodes = np.cumsum(node_lengths) * 2 + 1
#               max_number_fragments = len(
#                   cumul_sum_of_nodes[cumul_sum_of_nodes < len(cycle)]
#               )
#               if max_number_fragments > 0:
#                   current_node_lengths = node_lengths[
#                       :max_number_fragments
#                   ].copy()
#                   length_middle_node = (
#                       len(cycle) - sum(current_node_lengths) * 2
#                   )
#                   times_tmp = (
#                       current_node_lengths
#                       + [length_middle_node]
#                       + current_node_lengths[::-1]
#                   )
#                   pos_all_nodes = np.concatenate(
#                       [[0], np.cumsum(times_tmp[:-1])]
#                   )
#                   track = cycle[pos_all_nodes]
#                   out_dict.update(
#                       {k: [v] for k, v in zip(track[:-1], track[1:])}
#                   )
#                   times.update(zip(track, times_tmp))
#               else:
#                   for i, cell in enumerate(cycle[:-1]):
#                       out_dict[cell] = [cycle[i + 1]]
#                       times[cell] = 1
#               current = cycle[-1]
#               _next = self[current]
#               times[current] = 1
#               if _next and self.time[_next[0]] <= end_time:
#                   to_do.extend(_next)
#                   out_dict[current] = _next
#               else:
#                   out_dict[current] = []
#
#       return out_dict, times
#
#   @staticmethod
#   def _edist_format(adj_dict: dict):
#       inv_adj = {vi: k for k, v in adj_dict.items() for vi in v}
#       roots = set(adj_dict).difference(inv_adj)
#       nid2list = {}
#       list2nid = {}
#       nodes = []
#       adj_list = []
#       curr_id = 0
#       for r in roots:
#           to_do = [r]
#           while to_do:
#               curr = to_do.pop(0)
#               nid2list[curr] = curr_id
#               list2nid[curr_id] = curr
#               nodes.append(curr_id)
#               to_do = adj_dict.get(curr, []) + to_do
#               curr_id += 1
#           adj_list = [
#               [nid2list[d] for d in adj_dict.get(list2nid[_id], [])]
#               for _id in nodes
#           ]
#       return nodes, adj_list, list2nid
#
