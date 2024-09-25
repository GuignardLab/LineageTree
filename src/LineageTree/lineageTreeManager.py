import os
import pickle as pkl
from functools import partial

try:
    from edist import uted
except ImportError:
    warnings.warn(
        "No edist installed therefore you will not be able to compute the tree edit distance."
    )
from LineageTree import lineageTree

from .tree_styles import tree_style


class lineageTreeManager:
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

        """
        if isinstance(other_tree, lineageTree):
            for tree in self.lineagetrees.values():
                if tree == other_tree:
                    return False
            if name:
                self.lineagetrees[name] = other_tree
            else:
                if hasattr(other_tree, "name"):
                    name = other_tree.name
                    self.lineagetrees[name] = other_tree
                else:
                    name = f"Lineagetree {next(self)}"
                    self.lineagetrees[name] = other_tree
                    self.lineagetrees[name].name = name
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
        self.lineagetrees.pop(key, None)

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
        embryo_1,
        end_time1: int,
        n2: int,
        embryo_2,
        end_time2: int,
        style="fragmented",
        node_lengths: tuple = (1, 5, 7),
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

        tree = tree_style[style].value
        tree1 = tree(
            lT=self.lineagetrees[embryo_1],
            node_length=node_lengths,
            end_time=end_time1,
            root=n1,
        )
        tree2 = tree(
            lT=self.lineagetrees[embryo_2],
            node_length=node_lengths,
            end_time=end_time2,
            root=n2,
        )
        delta = tree1.delta
        _, times1 = tree1.tree
        _, times2 = tree2.tree
        nodes1, adj1, corres1 = tree1.edist
        nodes2, adj2, corres2 = tree2.edist
        if len(nodes1) == len(nodes2) == 0:
            return 0

        delta_tmp = partial(
            delta,
            corres1=corres1,
            times1=times1,
            corres2=corres2,
            times2=times2,
        )
        return uted.uted(nodes1, adj1, nodes2, adj2, delta=delta_tmp) / max(
            tree1.get_norm(), tree2.get_norm()
        )
