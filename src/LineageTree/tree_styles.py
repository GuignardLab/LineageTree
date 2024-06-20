from abc import ABC, abstractmethod


class abstract_trees(ABC):

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
    def get_norm(self):
        """
        Returns the correct norm method to be used for tree edit distances, because different trees need diferent methods.
        """
        pass

    @staticmethod
    def _edist_format(adj_dict: dict):
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


