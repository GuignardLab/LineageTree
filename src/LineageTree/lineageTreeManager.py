import os
import pickle as pkl
from functools import partial
from LineageTree import lineageTree
from edist.uted import uted
import numpy as np

class LineageTreeManager:

    def __init__(self):
        self.lineagetrees = {}
        self.classification = {"Wt": {},"Ptb": {}}
        self.lineageTree_counter = 0
        self.registered = {}

    def get_next_tree(self):
        self.lineageTree_counter+= 1
        return self.lineageTree_counter-1

    def add(self,other_tree, name: str = None, type:str = None):
        for tree in self.lineagetrees.values():
            if tree == other_tree:
                return False
        if isinstance(other_tree, lineageTree):
            if name:
                self.lineagetrees[name] = other_tree
            else:
                name = other_tree.name#f"LineageTree  {self.get_next_tree()}"
                self.lineagetrees[name] = other_tree
        if type in  ("Wt" ,"Ptb"):
            self.classification[type] = {name:other_tree }

    def __add__(self,other):
        self.add(self,other)

    def classify_existing(self,key,type:str):
        if type in ("Wt", "Ptb"):
            self.classification[type] = {key: self.lineagetrees[key]}
        else:
            return False

    def write(self, fname: str):
        if os.path.splitext(fname)[-1] != ".ltM":
            fname = os.path.extsep.join((fname, "ltM"))
        with open(fname, "bw") as f:
            pkl.dump(self, f)
            f.close()

    def remove_embryo(self, key):
        if key in self.lineagetrees.keys():
            del self.lineagetrees[key]
            if key in self.classification["Wt"]:
                del self.classification["Wt"][key]
            if key in self.classification["Ptb"]:
                del self.classification["Ptb"][key]
        else:
            raise Exception("No such lineage")


    @classmethod
    def load(cls,fname: str):
        """
        Loading a lineage tree Manager from a ".ltm" file.

        Args:
            fname (str): path to and name of the file to read

        Returns:
            (lineageTree): loaded file
        """
        with open(fname, "br") as f:
            ltm = pkl.load(f)
            f.close()
        return ltm

    def edist_format(self,adj_dict: dict):
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
    
    def cross_embryo_edit_distance(self, n1:int, embryo_1:str , end_time1:int, n2:int, embryo_2:str, end_time2:int, delta:callable = None, registration = None):

        def delta(x, y, corres1, corres2, times1, times2):
            if x is None and y is None:
                return 0
            if x is None:
                return times2[corres2[y]]
            if y is None:
                return times1[corres1[x]]
            len_x = times1[corres1[x]]
            len_y = times2[corres2[y]]
            return np.abs(len_x - len_y)


        def norm(x,y):
            return max(sum(times1.values()),
                        sum(times2.values()))

            # def norm(x,adj1,y,adj2,delta):
            #     return max(uted(x,adj1,[],[],delta = delta),
            #                uted(y,adj2,[],[],delta = delta))

        simple_tree_1, times1 = self.lineagetrees[embryo_1].get_comp_tree(n1, end_time=end_time1)
        simple_tree_2, times2 = self.lineagetrees[embryo_2].get_comp_tree(n2, end_time=end_time2)
        nodes1, adj1, corres1 = self.lineagetrees[embryo_1]._edist_format(simple_tree_1)
        nodes2, adj2, corres2 = self.lineagetrees[embryo_2]._edist_format(simple_tree_2)
        if len(nodes1) == len(nodes2) == 0:
            return 0

        delta_tmp = partial(
            delta, corres1=corres1, times1= times1, corres2=corres2, times2 = times2
        )
        return uted(nodes1, adj1, nodes2, adj2, delta=delta_tmp) / norm(
            n1,n2
         )
