import os
import pickle as pkl

from LineageTree import lineageTree


class LineageTreeManager:

    def __init__(self):
        self.lineagetrees = {}
        self.classification = {"Wt":{},"Ptb":{}}
        self.lineageTree_counter = 0
        self.registered = {}

    def get_next_tree(self):
        self.lineageTree_counter+=1
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

    def cross_embryo_comp(self, nodes_of_lineage1, lineage1, nodes_of_lineage2, lineage2, registration = None):
        
