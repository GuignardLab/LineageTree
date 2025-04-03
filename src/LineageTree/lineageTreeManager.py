import os
import pickle as pkl
import warnings
from collections.abc import Callable
from functools import partial
from typing import Literal, Union

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import colormaps

from .tree_styles import tree_style

try:
    from edist import uted
except ImportError:
    warnings.warn(
        "No edist installed therefore you will not be able to compute the tree edit distance.",
        stacklevel=2,
    )
import matplotlib.pyplot as plt

try:
    from edist import uted
except ImportError:
    warnings.warn(
        "No edist installed therefore you will not be able to compute the tree edit distance.",
        stacklevel=2,
    )
from LineageTree import lineageTree


class lineageTreeManager:
    def __init__(self):
        self.lineagetrees = {}
        self.lineageTree_counter = 0
        self.registered = {}
        self._comparisons = {}

    def __next__(self):
        self.lineageTree_counter += 1
        return self.lineageTree_counter - 1

    def __len__(self):
        return len(self.lineagetrees)

    def __iter__(
        self,
    ):
        yield from self.lineagetrees.items()

    def __getitem__(self, key):
        if key in self.lineagetrees:
            return self.lineagetrees[key]
        else:
            raise KeyError(f"'{key}' not found in the manager")

    @property
    def gcd(self):
        if len(self.lineagetrees) >= 1:
            all_time_res = [
                embryo._time_resolution
                for embryo in self.lineagetrees.values()
            ]
            return np.gcd.reduce(all_time_res)

    def add(self, other_tree: lineageTree, name: str = ""):
        """Function that adds a new lineagetree object to the class.
        Can be added either by .add or by using the + operator. If a name is
        specified it will also add it as this specific name, otherwise it will
        use the already existing name of the lineagetree.

        Parameters
        ----------
            other_tree : LineageTree
                Thelineagetree to be added.
            name : str, default=""
                Then name of the lineagetree to be added.

        """
        if isinstance(other_tree, lineageTree) and other_tree.time_resolution:
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
        else:
            raise Exception(
                "Please add a LineageTree object or add time resolution to the LineageTree added."
            )

    def __add__(self, other):
        self.add(other)

    def write(self, fname: str):
        """Saves the manager

        Parameters
        ----------
            fname : str
                The path and name of the file that is to be saved.
        """
        if os.path.splitext(fname)[-1] != ".ltM":
            fname = os.path.extsep.join((fname, "ltM"))
        with open(fname, "bw") as f:
            pkl.dump(self, f)
            f.close()

    def remove_embryo(self, key):
        """Removes the embryo from the manager.

        Parameters
        ----------
            key : str
                The name of the lineagetree to be removed

        Raises
        ------
            IndexError
                If there is not such a lineagetree
        """
        self.lineagetrees.pop(key, None)

    @classmethod
    def load(cls, fname: str):
        """
        Loading a lineage tree Manager from a ".ltm" file.

        Parameters
        ----------
            fname : str
                path to and name of the file to read

        Returns
        -------
            lineageTree
                loaded file
        """
        with open(fname, "br") as f:
            ltm = pkl.load(f)
            f.close()
        return ltm

    def __cross_lineage_edit_backtrace(
        self,
        n1: int,
        embryo_1: str,
        end_time1: int,
        n2: int,
        embryo_2: str,
        end_time2: int,
        style="simple",
        norm: Literal["max", "sum"] | None = "max",
        downsample: int = 2,
        registration=None,  # will be added as a later feature
    ):
        """Compute the unordered tree edit distance from Zhang 1996 between the trees spawned
        by two nodes `n1` from lineagetree1 and `n2` lineagetree2. The topology of the trees
        are compared and the matching cost is given by the function delta (see edist doc for
        more information).The distance is normed by the function norm that takes the two list
        of nodes spawned by the trees `n1` and `n2`.

        Parameters
        ----------
            n1 : int
                Node of the first Lineagetree
            embryo_1 : str
                The key/name of the first Lineagetree
            end_time1 : int
                End time of first Lineagetree
            n2 : int
                The key/name of the first Lineagetree
            embryo_2 : str
                Node of the second Lineagetree
            end_time2 : int
                End time of second lineagetree
            registration : _type_, default=None
                _description_. Defaults to None.
        """
        parameters = {
            k: v
            for k, v in locals().items()
            if k
            in (
                "n1",
                "n2",
                "end_time1",
                "end_time2",
                "embryo_1",
                "embryo_2",
                "norm",
                "style",
                "downsample",
            )
        }
        tree = tree_style[style].value
        lcm = (
            self.lineagetrees[embryo_1]._time_resolution
            * self.lineagetrees[embryo_2]._time_resolution
        ) / self.gcd
        if style == "downsampled":
            if downsample % (lcm / 10) != 0:
                raise Exception(
                    f"Use a valid downsampling rate (multiple of {lcm/10})"
                )
            time_res = [
                downsample / self.lineagetrees[embryo_2].time_resolution,
                downsample / self.lineagetrees[embryo_1].time_resolution,
            ]
        elif style == "full":
            time_res = [
                lcm / 10 / self.lineagetrees[embryo_2].time_resolution,
                lcm / 10 / self.lineagetrees[embryo_1].time_resolution,
            ]
        else:
            time_res = [
                self.lineagetrees[embryo_1]._time_resolution,
                self.lineagetrees[embryo_2]._time_resolution,
            ]
            time_res = [i / self.gcd for i in time_res]
        tree1 = tree(
            lT=self.lineagetrees[embryo_1],
            downsample=downsample,
            end_time=end_time1,
            root=n1,
            time_scale=time_res[0],
        )
        tree2 = tree(
            lT=self.lineagetrees[embryo_2],
            downsample=downsample,
            end_time=end_time2,
            root=n2,
            time_scale=time_res[1],
        )
        delta = tree1.delta
        _, times1 = tree1.tree
        _, times2 = tree2.tree

        nodes1, adj1, corres1 = tree1.edist
        nodes2, adj2, corres2 = tree2.edist
        if len(nodes1) == len(nodes2) == 0:
            self._comparisons[hash(frozenset(parameters.values()))] = {
                "alignment": (),
                "trees": (),
            }
            return self._comparisons[hash(frozenset(parameters.values()))]
        delta_tmp = partial(
            delta,
            corres1=corres1,
            times1=times1,
            corres2=corres2,
            times2=times2,
        )
        btrc = uted.uted_backtrace(nodes1, adj1, nodes2, adj2, delta=delta_tmp)

        self._comparisons[hash(frozenset(parameters.values()))] = {
            "alignment": btrc,
            "trees": (tree1, tree2),
        }
        return self._comparisons[hash(frozenset(parameters.values()))]

    def __calculate_distance_of_sub_tree(
        self,
        node1,
        lT1,
        node2,
        lT2,
        alignment,
        corres1,
        corres2,
        delta_tmp,
        norm: Callable,
        norm1,
        norm2,
    ):
        """Private method that calculates the distance of all subtrees in a specific mapping."""
        sub_tree_1 = set(lT1.get_sub_tree(node1))
        sub_tree_2 = set(lT2.get_sub_tree(node2))
        res = 0
        for m in alignment:
            if (
                corres1.get(m._left, -1) in sub_tree_1
                or corres2.get(m._right, -1) in sub_tree_2
            ):
                res += delta_tmp(
                    m._left if m._left != -1 else None,
                    m._right if m._right != -1 else None,
                )
        return res / norm([norm1, norm2])

    def clear_comparisons(self):
        self._comparisons.clear()

    def cross_lineage_edit_distance(
        self,
        n1: int,
        embryo_1: str,
        end_time1: int,
        n2: int,
        embryo_2: str,
        end_time2: int,
        norm: tuple["max", "sum","None"] | None = "max",
        style="simple",
        downsample: int = 2,
        return_norms:bool = False
    ) -> float | tuple[float, tuple[float, float]]:
        """
        Compute the unordered tree edit backtrace from Zhang 1996 between the trees spawned
        by two nodes `n1` and `n2`. The topology of the trees are compared and the matching
        cost is given by the function delta (see edist doc for more information).

        Parameters
        ----------
        n1 : int
            id of the first node to compare
        n2 : int
            id of the second node to compare
        end_time : int
            The final time point the comparison algorithm will take into account.
            If None all nodes will be taken into account.
        norm : {"max", "sum"}, default="max"
            The normalization method to use.
        style : {"simple", "full", "downsampled"}, default="simple"
            Which tree approximation is going to be used for the comparisons.
        downsample : int, default=2
            The downsample factor for the downsampled tree approximation.
            Used only when `style="downsampled"`.

        Returns
        -------
        Alignment
            The alignment between the nodes by the subtrees spawned by the nodes n1,n2 and the normalization function.`
        tuple(tree,2)
            The two trees that have been mapped to each other.
        """

        parameters = {
            k: v
            for k, v in locals().items()
            if k
            in (
                "n1",
                "n2",
                "end_time1",
                "end_time2",
                "embryo_1",
                "embryo_2",
                "norm",
                "style",
                "downsample",
            )
        }
        if hash(frozenset(parameters.values())) in self._comparisons:
            tmp = self._comparisons[hash(frozenset(parameters.values()))]
        else:
            tmp = self.__cross_lineage_edit_backtrace(**parameters)
        if len(self._comparisons) > 100:
            warnings.warn(
                "More than 100 comparisons are saved, use clear_comparisons() to delete them.",
                stacklevel=2,
            )
        btrc = tmp["alignment"]
        tree1, tree2 = tmp["trees"]
        _, times1 = tree1.tree
        _, times2 = tree2.tree
        (
            nodes1,
            adj1,
            corres1,
        ) = tree1.edist
        (
            nodes2,
            adj2,
            corres2,
        ) = tree2.edist
        if len(nodes1) == len(nodes2) == 0:
            self._comparisons[hash(frozenset(parameters.values()))] = {
                "alignment": (),
                "trees": (),
            }
            return self._comparisons[hash(frozenset(parameters.values()))]
        delta_tmp = partial(
            tree1.delta,
            corres1=corres1,
            corres2=corres2,
            times1=times1,
            times2=times2,
        )
        norm_dict = {"max": max, "sum": sum, None: lambda x: 1}
        if norm not in norm_dict:
            raise ValueError("Select a viable normalization method (max, sum, None)")
        cost = btrc.cost(nodes1, nodes2, delta_tmp)
        norm_values = (tree1.get_norm(n1), tree2.get_norm(n2))
        if return_norms:
            return cost, norm_values
        return cost / norm_dict[norm](norm_values)


    def plot_tree_distance_graphs(
        self,
        n1: int,
        embryo_1,
        end_time1,
        n2: int,
        embryo_2,
        end_time2,
        norm: Literal["max", "sum"] | None = "max",
        style="simple",
        downsample: int = 2,
        colormap: str = "cool",
        default_color: str = "black",
        size: float = 10,
        lw: float = 0.3,
        ax: list[plt.Axes, plt.Axes] = None,
    ) -> tuple[plt.figure, plt.Axes]:
        """
        Plots the distance graphs of 2 nodes compared.
        !!!TODO make documentation!!!

        Parameters
        ----------
        n1 : int
            id of the first node to compare
        n2 : int
            id of the second node to compare
        end_time : int
            The final time point the comparison algorithm will take into account.
            If None all nodes will be taken into account.
        norm : {"max", "sum"}, default="max"
            The normalization method to use.
        style : {"simple", "full", "downsampled"}, default="simple"
            Which tree approximation is going to be used for the comparisons.
        downsample : int, default=2
            The downsample factor for the downsampled tree approximation.
            Used only when `style="downsampled"`.

        Returns
        -------
        plt.figure
            The figure of the tree distance graph
        plt.Axes
            The axes of the tree distance graph
        """

        parameters = {
            k: v
            for k, v in locals().items()
            if k
            in (
                "n1",
                "n2",
                "end_time1",
                "end_time2",
                "embryo_1",
                "embryo_2",
                "norm",
                "style",
                "downsample",
            )
        }
        if hash(frozenset(parameters.values())) in self._comparisons:
            tmp = self._comparisons[hash(frozenset(parameters.values()))]
        else:
            tmp = self.__cross_lineage_edit_backtrace(**parameters)
        btrc = tmp["alignment"]
        tree1, tree2 = tmp["trees"]
        _, times1 = tree1.tree
        _, times2 = tree2.tree
        (
            *_,
            corres1,
        ) = tree1.edist
        (
            *_,
            corres2,
        ) = tree2.edist
        delta_tmp = partial(
            tree1.delta,
            corres1=corres1,
            corres2=corres2,
            times1=times1,
            times2=times2,
        )
        norm_dict = {"max": max, "sum": sum, "None": lambda x: 1}
        if norm is None:
            norm = "None"
        if norm not in norm_dict:
            raise Warning(
                "Select a viable normalization method (max, sum, None)"
            )
        matched_right = []
        matched_left = []
        unmatched_node = []
        colors = {}
        if style not in ("full", "downsampled"):
            for m in btrc:
                if m._left != -1 and m._right != -1:
                    cyc1 = tree1.lT.get_cycle(corres1[m._left])
                    if len(cyc1) > 1:
                        node_1, *_, l_node_1 = cyc1
                        matched_left.append(node_1)
                        matched_left.append(l_node_1)
                    elif len(cyc1) == 1:
                        node_1 = l_node_1 = cyc1.pop()
                        matched_left.append(node_1)

                    cyc2 = tree2.lT.get_cycle(corres2[m._right])
                    if len(cyc2) > 1:
                        node_2, *_, l_node_2 = cyc2
                        matched_right.append(node_2)
                        matched_right.append(l_node_2)

                    elif len(cyc2) == 1:
                        node_2 = l_node_2 = cyc2.pop()
                        matched_right.append(node_2)

                    colors[node_1] = self.__calculate_distance_of_sub_tree(
                        node_1,
                        tree1.lT,
                        node_2,
                        tree2.lT,
                        btrc,
                        corres1,
                        corres2,
                        delta_tmp,
                        norm_dict[norm],
                        tree1.get_norm(node_1),
                        tree2.get_norm(node_2),
                    )
                    colors[node_2] = colors[node_1]
                    colors[l_node_1] = colors[node_1]
                    colors[l_node_2] = colors[node_2]

                else:
                    if m._left != -1:
                        node_1 = tree1.lT.get_cycle(corres1.get(m._left, "-"))[
                            0
                        ]
                    else:
                        node_1 = tree2.lT.get_cycle(
                            corres2.get(m._right, "-")
                        )[0]
                    unmatched_node.append(node_1)
        else:
            for m in btrc:
                if m._left != -1 and m._right != -1:
                    node_1 = tree1.lT.get_cycle(corres1[m._left])[0]
                    node_2 = tree2.lT.get_cycle(corres2[m._right])[0]
                    if tree1.lT.get_cycle(node_1)[0] == node_1 or  tree2.lT.get_cycle(node_2)[0] == node_2 or node_1 not in colors:
                            matched_left.append(node_1)
                            matched_right.append(node_2)
                            colors[node_1] = self.__calculate_distance_of_sub_tree(
                                node_1,
                                tree1.lT,
                                node_2,
                                tree2.lT,
                                btrc,
                                corres1,
                                corres2,
                                delta_tmp,
                                norm_dict[norm],
                                tree1.get_norm(node_1),
                                tree2.get_norm(node_2),
                            )
                            colors[node_2] = colors[node_1]
                            colors[tree1.lT.get_cycle(node_1)[-1]] = colors[node_1]
                            colors[tree2.lT.get_cycle(node_2)[-1]] = colors[node_2]

                            if tree1.lT.get_cycle(node_1)[-1]!=node_1:
                                matched_left.append( tree1.lT.get_cycle(node_1)[-1])
                            if tree2.lT.get_cycle(node_2)[-1]!=node_2:
                                matched_right.append( tree2.lT.get_cycle(node_2)[-1])
                else:
                    if m._left != -1:
                        node_1 = tree1.lT.get_cycle(corres1.get(m._left, "-"))[
                            0
                        ]
                    else:
                        node_1 = tree2.lT.get_cycle(
                            corres2.get(m._right, "-")
                        )[0]
                    unmatched_node.append(node_1)
                # for br in tree1.lT.get_all_branches_of_node(n1):
                #     col = [colors[node] for node in br if node in colors]
                #     if col:
                #         colors[br[0]] = np.average(col)
                #         matched_left.append(br[0])
                #         colors[br[-1]] = np.average(col)
                #         matched_left.append(br[-1])

                # for br in tree2.lT.get_all_branches_of_node(n2):
                #     col = [colors[node] for node in br if node in colors]
                #     if col:
                #         colors[br[0]] = np.average(col)
                #         matched_right.append(br[0])
                #         colors[br[-1]] = colors[br[0]]
                #         matched_right.append(br[-1])
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=2)
        cmap = colormaps[colormap]
        c_norm = mcolors.Normalize(0, 1)
        colors = {c: cmap(c_norm(v)) for c, v in colors.items()}
        tree1.lT.plot_node(
            tree1.lT.get_ancestor_at_t(n1),
            end_time=end_time1,
            size=size,
            selected_nodes=matched_left,
            color_of_nodes=colors,
            selected_edges=matched_left,
            color_of_edges=colors,
            default_color=default_color,
            lw=lw,
            ax=ax[0],
        )
        tree2.lT.plot_node(
            tree2.lT.get_ancestor_at_t(n2),
            end_time=end_time2,
            size=size,
            selected_nodes=matched_right,
            color_of_nodes=colors,
            selected_edges=matched_right,
            color_of_edges=colors,
            default_color=default_color,
            lw=lw,
            ax=ax[1],
        )
        return ax[0].get_figure(), ax

    def labelled_mappings(
        self,
        n1: int,
        embryo_1,
        end_time1,
        n2: int,
        embryo_2,
        end_time2,
        norm: Literal["max", "sum"] | None = "max",
        style="simple",
        downsample: int = 2,
        colormap: str = "cool",
        default_color: str = "black",
        size: float = 10,
        ax: list[plt.Axes, plt.Axes] = None,
    ) -> dict[str, list]:
        """
        Plots the distance graphs of 2 nodes compared.
        !!!TODO make documentation!!!

        Parameters
        ----------
        n1 : int
            id of the first node to compare
        n2 : int
            id of the second node to compare
        end_time : int
            The final time point the comparison algorithm will take into account.
            If None all nodes will be taken into account.
        norm : {"max", "sum"}, default="max"
            The normalization method to use.
        style : {"simple", "full", "downsampled"}, default="simple"
            Which tree approximation is going to be used for the comparisons.
        downsample : int, default=2
            The downsample factor for the downsampled tree approximation.
            Used only when `style="downsampled"`.

        Returns
        -------
        Alignment
            The alignment between the nodes of of the subtrees  spawned by the nodes n1,n2 .`
        """

        if ax:
            assert len(ax) == 2
            assert isinstance(ax[0], plt.Axes)
        parameters = {
            k: v
            for k, v in locals().items()
            if k
            in (
                "n1",
                "n2",
                "end_time1",
                "end_time2",
                "embryo_1",
                "embryo_2",
                "norm",
                "style",
                "downsample",
            )
        }
        if hash(frozenset(parameters.values())) in self._comparisons:
            tmp = self._comparisons[hash(frozenset(parameters.values()))]
        else:
            tmp = self.__cross_lineage_edit_backtrace(**parameters)
        btrc = tmp["alignment"]
        tree1, tree2 = tmp["trees"]
        _, times1 = tree1.tree
        _, times2 = tree2.tree
        (
            *_,
            corres1,
        ) = tree1.edist
        (
            *_,
            corres2,
        ) = tree2.edist
        delta_tmp = partial(
            tree1.delta,
            corres1=corres1,
            corres2=corres2,
            times1=times1,
            times2=times2,
        )
        norm_dict = {"max": max, "sum": sum, "None": lambda x: 1}
        if norm is None:
            norm = "None"
        if norm not in norm_dict:
            raise Warning(
                "Select a viable normalization method (max, sum, None)"
            )
        matched = []
        unmatched = []
        colors = {}
        if style not in ("full", "downsampled"):
            for m in btrc:
                if m._left != -1 and m._right != -1:
                    cyc1 = tree1.lT.get_cycle(corres1[m._left])
                    if len(cyc1) > 1:
                        node_1, *_, l_node_1 = cyc1
                    elif len(cyc1) == 1:
                        node_1 = l_node_1 = cyc1.pop()

                    cyc2 = tree2.lT.get_cycle(corres2[m._right])
                    if len(cyc2) > 1:
                        node_2, *_, l_node_2 = cyc2

                    elif len(cyc2) == 1:
                        node_2 = l_node_2 = cyc2.pop()

                    matched.append(
                        (
                            tree1.lT.labels.get(node_1, node_1),
                            tree2.lT.labels.get(node_2, node_2),
                        )
                    )
                else:
                    if m._left != -1:
                        tmp_node = tree1.lT.get_cycle(
                            corres1.get(m._left, "-")
                        )[0]
                        node_1 = (
                            tree1.lT.labels.get(tmp_node, tmp_node),
                            tree1.lT.name,
                        )
                    else:
                        tmp_node = tree2.lT.get_cycle(
                            corres2.get(m._right, "-")
                        )[0]
                        node_1 = (
                            tree2.lT.labels.get(tmp_node, tmp_node),
                            tree2.lT.name,
                        )
                    unmatched.append(node_1)
        else:
            for m in btrc:
                if m._left != -1 and m._right != -1:
                    node_1 = corres1[m._left]
                    node_2 = corres2[m._right]
                    matched.append(
                        (
                            tree1.lT.labels.get(node_1, node_1),
                            tree2.lT.labels.get(node_2, node_2),
                        )
                    )
                else:
                    if m._left != -1:
                        tmp_node = tree1.lT.get_cycle(
                            corres1.get(m._left, "-")
                        )[0]
                        node_1 = (
                            tree1.lT.labels.get(tmp_node, tmp_node),
                            tree1.lT.name,
                        )
                    else:
                        tmp_node = tree2.lT.get_cycle(
                            corres2.get(m._right, "-")
                        )[0]
                        node_1 = (
                            tree2.lT.labels.get(tmp_node, tmp_node),
                            tree2.lT.name,
                        )
                    unmatched.append(node_1)
        return {"matched": matched, "unmatched": unmatched}
