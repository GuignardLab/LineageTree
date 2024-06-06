import os
import pickle as pkl
from functools import partial

import numpy as np
from edist.uted import uted

from LineageTree import lineageTree


class LineageTreeManager:
    def __init__(self):
        self.lineagetrees = {}
        # self.classification = {"Wt": {}, "Ptb": {}}
        self.lineageTree_counter = 0
        self.registered = {}

    def __next__(self):
        self.lineageTree_counter += 1
        return self.lineageTree_counter - 1

    def add(
        self, other_tree: lineageTree, name: str = "", classification: str = ""
    ):
        """Function that adds a new lineagetree object to the class.
        Can be added either by .add or by using the + operator. If a name is
        specified it will also add it as this specific name, otherwise it will
        use the already existing name of the lineagetree.

        Args:
            other_tree (lineageTree): Thelineagetree to be added.
            name (str, optional): Then name of. Defaults to "".
           

        Returns:
            _type_: _description_
        """
        for tree in self.lineagetrees.values():
            if tree == other_tree:
                return False
        if isinstance(other_tree, lineageTree):
            if name:
                self.lineagetrees[name] = other_tree
            else:
                if hasattr(other_tree, "name"):
                    name = other_tree.name
                    self.lineagetrees[name] = other_tree
                else:
                    self.lineagetrees[
                        f"Lineagetree {next(self)}"
                    ] = other_tree
                # try:
                #     name = other_tree.name
                #     self.lineagetrees[name] = other_tree
                # except:
                #     self.lineagetrees[
                #         f"Lineagetree {next(self)}"
                #     ] = other_tree
        # if classification in ("Wt", "Ptb"):
        #     self.classification[type] = {name: other_tree}

    def __add__(self, other):
        self.add(other)

    # def classify_existing(self, key, classification: str):
    #     if classification in ("Wt", "Ptb"):
    #         self.classification[classification] = {key: self.lineagetrees[key]}
    #     else:
    #         return False

    def write(self, fname: str):
        """Saves the manager

        Args:
            fname (str): The path and name of the file that is to be saved.
        """
        if os.path.splitext(fname)[-1] != ".ltM":
            fname = os.path.extsep.join((fname, "ltM"))
        with open(fname, "bw") as f:
            pkl.dump(self, f)
            f.close()

    def remove_embryo(self, key):
        """Removes the embryo from the manager.

        Args:
            key (str): The name of the lineagetree to be removed

        Raises:
            Exception: If there is not such a lineagetree
        """
        self.lineagetrees.pop(key,None)
        

    @classmethod
    def load(cls, fname: str):
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

    def cross_lineage_edit_distance(
        self,
        n1: int,
        embryo_1: str,
        end_time1: int,
        n2: int,
        embryo_2: str,
        end_time2: int,
        registration=None,
    ):
        """Compute the unordered tree edit distance from Zhang 1996 between the trees spawned
        by two nodes `n1` from lineagetree1 and `n2` lineagetree2. The topology of the trees
        are compared and the matching cost is given by the function delta (see edist doc for
        more information).The distance is normed by the function norm that takes the two list
        of nodes spawned by the trees `n1` and `n2`.

        Args:
            n1 (int): Node of the first Lineagetree
            embryo_1 (str): The key/name of the first Lineagetree
            end_time1 (int): End time of first Lineagetree
            n2 (int): The key/name of the first Lineagetree
            embryo_2 (str): Node of the second Lineagetree
            end_time2 (int): End time of second lineagetree
            registration (_type_, optional): _description_. Defaults to None.
        """

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

        def norm(times1, times2):
            return max(sum(times1.values()), sum(times2.values()))

        simple_tree_1, times1 = self.lineagetrees[embryo_1].get_comp_tree(
            n1, end_time=end_time1
        )
        simple_tree_2, times2 = self.lineagetrees[embryo_2].get_comp_tree(
            n2, end_time=end_time2
        )
        nodes1, adj1, corres1 = self.lineagetrees[embryo_1]._edist_format(
            simple_tree_1
        )
        nodes2, adj2, corres2 = self.lineagetrees[embryo_2]._edist_format(
            simple_tree_2
        )
        if len(nodes1) == len(nodes2) == 0:
            return 0

        delta_tmp = partial(
            delta,
            corres1=corres1,
            times1=times1,
            corres2=corres2,
            times2=times2,
        )
        return uted(nodes1, adj1, nodes2, adj2, delta=delta_tmp) / norm(
            times1, times2
        )
