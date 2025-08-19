from __future__ import annotations

import os
import pickle as pkl
import warnings
from collections.abc import Callable, Generator, Iterable
from functools import partial
from typing import TYPE_CHECKING, Literal

import matplotlib.colors as mcolors
import numpy as np
from matplotlib import colormaps

from .lineage_tree import LineageTree
from .tree_approximation import tree_style

try:
    from edist import uted
except ImportError:
    warnings.warn(
        "No edist installed therefore you will not be able to compute the tree edit distance.",
        stacklevel=2,
    )
import matplotlib.pyplot as plt
from edist import uted

from .tree_approximation import TreeApproximationTemplate
from ._core.utils import convert_style_to_number

if TYPE_CHECKING:
    from edist.alignment import Alignment


class LineageTreeManager:
    norm_dict = {"max": max, "sum": sum, None: lambda x: 1}

    def __init__(self, lineagetree_list: Iterable[LineageTree] = ()):
        """Creates a LineageTreeManager
        :TODO: write the docstring

        Parameters
        ----------
        lineagetree_list: Iterable of LineageTree
            List of lineage trees to be in the LineageTreeManager
        """
        self.lineagetrees: dict[str, LineageTree] = {}
        self.lineageTree_counter: int = 0
        self._comparisons: dict = {}
        for lT in lineagetree_list:
            self.add(lT)

    def __next__(self) -> int:
        self.lineageTree_counter += 1
        return self.lineageTree_counter - 1

    def __len__(self) -> int:
        """Returns how many lineagetrees are in the manager.

        Returns
        -------
        int
            The number of trees inside the manager
        """
        return len(self.lineagetrees)

    def __iter__(self) -> Generator[tuple[str, LineageTree]]:
        yield from self.lineagetrees.items()

    def __getitem__(self, key: str) -> LineageTree:
        if key in self.lineagetrees:
            return self.lineagetrees[key]
        else:
            raise KeyError(f"'{key}' not found in the manager")

    @property
    def gcd(self) -> int:
        """Calculates the greatesτ common divisor between all lineagetree resolutions in the manager.

        Returns
        -------
        int
            The overall greatest common divisor.
        """
        if len(self) > 1:
            all_time_res = [
                embryo._time_resolution
                for embryo in self.lineagetrees.values()
            ]
            return np.gcd.reduce(all_time_res)
        elif len(self):
            return 1
        else:
            raise ValueError(
                "You cannot calculate the greatest common divisor of time resolutions with an empty manager."
            )

    def add(self, other_tree: LineageTree, name: str = ""):
        """Function that adds a new lineagetree object to the class.
        Can be added either by .add or by using the + operator. If a name is
        specified it will also add it as this specific name, otherwise it will
        use the already existing name of the lineagetree.

        Parameters
        ----------
        other_tree : LineageTree
            Thelineagetree to be added.
        name : str, default=""
            Then name of the lineagetree to be added, defaults to ''.
            (Usually lineageTrees have the name of the path they are read from,
            so this is going to be the name most of the times.)
        """
        if isinstance(other_tree, LineageTree):
            for tree in self.lineagetrees.values():
                if tree == other_tree:
                    return False
            if name:
                self.lineagetrees[name] = other_tree
            else:
                if other_tree.name:
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

    def __add__(self, other: LineageTree):
        self.add(other)

    def write(self, fname: str):
        """Saves the manager

        Parameters
        ----------
        fname : str
            The path and name of the file that is to be saved.
        """
        if os.path.splitext(fname)[-1].upper() != ".LTM":
            fname = os.path.extsep.join((fname, "lTM"))
        for _, lT in self:
            if hasattr(lT, "_protected_predecessor"):
                del lT._protected_predecessor
            if hasattr(lT, "_protected_successor"):
                del lT._protected_successor
            if hasattr(lT, "_protected_time"):
                del lT._protected_time
        with open(fname, "bw") as f:
            pkl.dump(self, f)
            f.close()

    def remove_embryo(self, key: str):
        """Removes the embryo from the manager.

        Parameters
        ----------
        key : str
            The name of the lineagetree to be removed

        Raises
        ------
        IndexError
            If there is no such lineagetree
        """
        self.lineagetrees.pop(key, None)

    @classmethod
    def load(cls, fname: str) -> LineageTreeManager:
        """Loading a lineage tree Manager from a ".ltm" file.

        Parameters
        ----------
        fname : str
            path to and name of the file to read

        Returns
        -------
        LineageTreeManager
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
        n2: int,
        embryo_2: str,
        end_time1: int | None = None,
        end_time2: int | None = None,
        style: (
            Literal["simple", "normalized_simple", "full", "downsampled"]
            | type[TreeApproximationTemplate]
        ) = "simple",
        norm: Literal["max", "sum", None] = "max",
        downsample: int = 2,
    ) -> dict[
        str,
        Alignment
        | tuple[TreeApproximationTemplate, TreeApproximationTemplate],
    ]:
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
        n2 : int
            The key/name of the first Lineagetree
        embryo_2 : str
            Node of the second Lineagetree
        end_time1 : int, optional
            The final time point the comparison algorithm will take into account for the first dataset.
            If None or not provided all nodes will be taken into account.
        end_time2 : int, optional
             The final time point the comparison algorithm will take into account for the second dataset.
            If None or not provided all nodes will be taken into account.
        style : {"simple", "normalized_simple", "full", "downsampled"} or TreeApproximationTemplate subclass, default="simple"
            The approximation used to calculate the tree.
        norm : {"max","sum", "None"}, default="max"
            The normalization method used (Not important for this function)
        downsample : int, default==2
            The downsample factor for the downsampled tree approximation.
            Used only when `style="downsampled"`.

        Returns
        -------
        dict mapping str to Alignment or tuple of [TreeApproximationTemplate, TreeApproximationTemplate]
            - 'alignment'
                The alignment between the nodes by the subtrees spawned by the nodes n1,n2 and the normalization function.`
            - 'trees'
                A list of the two trees that have been mapped to each other.
        """
        if (
            self[embryo_1].time_resolution <= 0
            or self[embryo_2].time_resolution <= 0
        ):
            raise Warning("Resolution cannot be <=0 ")
        parameters = (
            (end_time1, end_time2),
            convert_style_to_number(style, downsample),
        )
        n1_embryo, n2_embryo = sorted(
            ((n1, embryo_1), (n2, embryo_2)), key=lambda x: x[0]
        )
        self._comparisons.setdefault(parameters, {})
        if isinstance(style, str):
            tree = tree_style[style].value
        elif issubclass(style, TreeApproximationTemplate):
            tree = style
        else:
            raise Warning("Use a valid approximation.")
        time_res = tree.handle_resolutions(
            time_resolution1=self.lineagetrees[embryo_1]._time_resolution,
            time_resolution2=self.lineagetrees[embryo_2]._time_resolution,
            gcd=self.gcd,
            downsample=downsample,
        )
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
            self._comparisons[parameters][(n1_embryo, n2_embryo)] = {
                "alignment": (),
                "trees": (),
            }
            return self._comparisons[parameters][(n1_embryo, n2_embryo)]
        delta_tmp = partial(
            delta,
            corres1=corres1,
            times1=times1,
            corres2=corres2,
            times2=times2,
        )
        btrc = uted.uted_backtrace(nodes1, adj1, nodes2, adj2, delta=delta_tmp)

        self._comparisons[parameters][(n1_embryo, n2_embryo)] = {
            "alignment": btrc,
            "trees": (tree1, tree2),
        }
        return self._comparisons[parameters][(n1_embryo, n2_embryo)]

    def __calculate_distance_of_sub_tree(
        self,
        node1: int,
        lT1: LineageTree,
        node2: int,
        lT2: LineageTree,
        alignment: Alignment,
        corres1: dict,
        corres2: dict,
        delta_tmp: Callable,
        norm: Callable,
        norm1: int | float,
        norm2: int | float,
    ) -> float:
        """Calculates the distance of the subtree of a node matched in a comparison.
        DOES NOT CALCULATE THE DISTANCE FROM SCRATCH BUT USING THE ALIGNMENT.

        TODO ITS BOUND TO CHANGE

        Parameters
        ----------
        node1 : int
            The root of the first subtree
        lT1 : LineageTree
            The dataset the first lineage exists
        node2 : int
            The root of the first subtree
        lT2 : LineageTree
            The dataset the second lineage exists
        alignment : Alignment
            The alignment of the subtree
        corres1 : dict
            The correspndance dictionary of the first lineage
        corres2 : dict
            The correspondance dictionary of the second lineage
        delta_tmp : Callable
            The delta function for the comparisons
        norm : Callable
            How should the lineages be normalized
        norm1 : int or float
            The result of the normalization of the first tree
        norm2 : int or float
            The result of the normalization of the second tree

        Returns
        -------
        float
            The result of the comparison of the subtree
        """
        sub_tree_1 = set(lT1.get_subtree_nodes(node1))
        sub_tree_2 = set(lT2.get_subtree_nodes(node2))
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
        n2: int,
        embryo_2: str,
        end_time1: int | None = None,
        end_time2: int | None = None,
        norm: Literal["max", "sum", None] = "max",
        style: (
            Literal["simple", "normalized_simple", "full", "downsampled"]
            | type[TreeApproximationTemplate]
        ) = "simple",
        downsample: int = 2,
        return_norms: bool = False,
    ) -> float | tuple[float, tuple[float, float]]:
        """
        Compute the unordered tree edit backtrace from Zhang 1996 between the trees spawned
        by two nodes `n1` and `n2`. The topology of the trees are compared and the matching
        cost is given by the function delta (see edist doc for more information). There are
        5 styles available (tree approximations) and the user may add their own.

        Parameters
        ----------
        n1 : int
            id of the first node to compare
        embryo_1 : str
            the name of the first embryo to be used. (from lTm.lineagetrees.keys())
        n2 : int
            id of the second node to compare
        embryo_2 : str
            the name of the second embryo to be used. (from lTm.lineagetrees.keys())
        end_time_1 : int, optional
            the final time point the comparison algorithm will take into account for the first dataset.
            If None or not provided all nodes will be taken into account.
        end_time_2 : int, optional
            the final time point the comparison algorithm will take into account for the second dataset.
            If None or not provided all nodes will be taken into account.
        norm : {"max", "sum"}, default="max"
            The normalization method to use, defaults to 'max'.
        style : {"simple", "normalized_simple", "full", "downsampled"} or TreeApproximationTemplate subclass, default="simple"
            Which tree approximation is going to be used for the comparisons, defaults to 'simple'.
        downsample : int, default=2
            The downsample factor for the downsampled tree approximation.
            Used only when `style="downsampled"`.
        return_norms : bool
            Decide if the norms will be returned explicitly (mainly used for the napari plugin)

        Returns
        -------
        Alignment
            The alignment between the nodes by the subtrees spawned by the nodes n1,n2 and the normalization function.`
        tuple(tree,tree), optional
            The two trees that have been mapped to each other.
            Returned if `return_norms` is `True`
        """

        parameters = (
            (end_time1, end_time2),
            convert_style_to_number(style, downsample),
        )
        n1_embryo, n2_embryo = sorted(
            ((n1, embryo_1), (n2, embryo_2)), key=lambda x: x[0]
        )
        self._comparisons.setdefault(parameters, {})
        if self._comparisons[parameters].get((n1, n2)):
            tmp = self._comparisons[parameters][(n1_embryo, n2_embryo)]
        else:
            tmp = self.__cross_lineage_edit_backtrace(
                n1,
                embryo_1,
                n2,
                embryo_2,
                end_time1,
                end_time2,
                style,
                norm,
                downsample,
            )
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
            self._comparisons[hash(frozenset(parameters))] = {
                "alignment": (),
                "trees": (),
            }
            return self._comparisons[hash(frozenset(parameters))]
        delta_tmp = partial(
            tree1.delta,
            corres1=corres1,
            corres2=corres2,
            times1=times1,
            times2=times2,
        )
        if norm not in self.norm_dict:
            raise ValueError(
                "Select a viable normalization method (max, sum, None)"
            )
        cost = btrc.cost(nodes1, nodes2, delta_tmp)
        norm_values = (tree1.get_norm(n1), tree2.get_norm(n2))
        if return_norms:
            return cost, norm_values
        return cost / self.norm_dict[norm](norm_values)

    def plot_tree_distance_graphs(
        self,
        n1: int,
        embryo_1: str,
        n2: int,
        embryo_2: str,
        end_time1: int | None = None,
        end_time2: int | None = None,
        norm: Literal["max", "sum"] | None = "max",
        style: (
            Literal["simple", "normalized_simple", "full", "downsampled"]
            | type[TreeApproximationTemplate]
        ) = "simple",
        downsample: int = 2,
        colormap: str = "cool",
        default_color: str = "black",
        size: float = 10,
        lw: float = 0.3,
        ax: np.ndarray | None = None,
    ) -> tuple[plt.figure, plt.Axes]:
        """
        Plots the subtrees compared and colors them according to the quality of the matching of their subtree.

        Parameters
        ----------
        n1 : int
            id of the first node to compare
        embryo_1 : str
            the name of the first embryo
        n2 : int
            id of the second node to compare
        embryo_2 : str
            the name of the second embryo
        end_time1 : int, optional
            the final time point the comparison algorithm will take into account for the first dataset.
            If None or not provided all nodes will be taken into account.
        end_time2 : int, optional
            the final time point the comparison algorithm will take into account for the second dataset.
            If None or not provided all nodes will be taken into account.
        norm : {"max", "sum"}, default="max"
            The normalization method to use.
        style : {"simple", "normalized_simple", "full", "downsampled"} or TreeApproximationTemplate subclass, default="simple"
            Which tree approximation is going to be used for the comparisons.
        downsample : int, default=2
            The downsample factor for the downsampled tree approximation.
            Used only when `style="downsampled"`.
        colormap : str, default="cool"
            The colormap used for matched nodes, defaults to "cool"
        default_color : str
            The color of the unmatched nodes, defaults to "black"
        size : float
            The size of the nodes, defaults to 10
        lw : float
            The width of the edges, defaults to 0.3
        ax : np.ndarray, optional
            The axes used, if not provided another set of axes is produced, defaults to None

        Returns
        -------
        plt.Figure
            The matplotlib figure
        plt.Axes
            The matplotlib axes
        """

        parameters = (
            (end_time1, end_time2),
            convert_style_to_number(style, downsample),
        )
        n1_embryo, n2_embryo = sorted(
            ((n1, embryo_1), (n2, embryo_2)), key=lambda x: x[0]
        )
        self._comparisons.setdefault(parameters, {})
        if self._comparisons[parameters].get((n1, n2)):
            tmp = self._comparisons[parameters][(n1_embryo, n2_embryo)]
        else:
            tmp = self.__cross_lineage_edit_backtrace(
                n1,
                embryo_1,
                n2,
                embryo_2,
                end_time1,
                end_time2,
                style,
                norm,
                downsample,
            )
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
        if norm not in self.norm_dict:
            raise Warning(
                "Select a viable normalization method (max, sum, None)"
            )
        matched_right = []
        matched_left = []
        colors1 = {}
        colors2 = {}
        if style not in ("full", "downsampled"):
            for m in btrc:
                if m._left != -1 and m._right != -1:
                    cyc1 = tree1.lT.get_chain_of_node(corres1[m._left])
                    if len(cyc1) > 1:
                        node_1, *_, l_node_1 = cyc1
                        matched_left.append(node_1)
                        matched_left.append(l_node_1)
                    elif len(cyc1) == 1:
                        node_1 = l_node_1 = cyc1.pop()
                        matched_left.append(node_1)

                    cyc2 = tree2.lT.get_chain_of_node(corres2[m._right])
                    if len(cyc2) > 1:
                        node_2, *_, l_node_2 = cyc2
                        matched_right.append(node_2)
                        matched_right.append(l_node_2)

                    elif len(cyc2) == 1:
                        node_2 = l_node_2 = cyc2.pop()
                        matched_right.append(node_2)

                    colors1[node_1] = self.__calculate_distance_of_sub_tree(
                        node_1,
                        tree1.lT,
                        node_2,
                        tree2.lT,
                        btrc,
                        corres1,
                        corres2,
                        delta_tmp,
                        self.norm_dict[norm],
                        tree1.get_norm(node_1),
                        tree2.get_norm(node_2),
                    )
                    colors2[node_2] = colors1[node_1]
                    colors1[l_node_1] = colors1[node_1]
                    colors2[l_node_2] = colors2[node_2]

        else:
            for m in btrc:
                if m._left != -1 and m._right != -1:
                    node_1 = tree1.lT.get_chain_of_node(corres1[m._left])[0]
                    node_2 = tree2.lT.get_chain_of_node(corres2[m._right])[0]
                    if (
                        tree1.lT.get_chain_of_node(node_1)[0] == node_1
                        or tree2.lT.get_chain_of_node(node_2)[0] == node_2
                        and (node_1 not in colors1 or node_2 not in colors2)
                    ):
                        matched_left.append(node_1)
                        l_node_1 = tree1.lT.get_chain_of_node(node_1)[-1]
                        matched_left.append(l_node_1)
                        matched_right.append(node_2)
                        l_node_2 = tree2.lT.get_chain_of_node(node_2)[-1]
                        matched_right.append(l_node_2)
                        colors1[node_1] = (
                            self.__calculate_distance_of_sub_tree(
                                node_1,
                                tree1.lT,
                                node_2,
                                tree2.lT,
                                btrc,
                                corres1,
                                corres2,
                                delta_tmp,
                                self.norm_dict[norm],
                                tree1.get_norm(node_1),
                                tree2.get_norm(node_2),
                            )
                        )
                        colors2[node_2] = colors1[node_1]
                        colors1[tree1.lT.get_chain_of_node(node_1)[-1]] = (
                            colors1[node_1]
                        )
                        colors2[tree2.lT.get_chain_of_node(node_2)[-1]] = (
                            colors2[node_2]
                        )

                        if tree1.lT.get_chain_of_node(node_1)[-1] != node_1:
                            matched_left.append(
                                tree1.lT.get_chain_of_node(node_1)[-1]
                            )
                        if tree2.lT.get_chain_of_node(node_2)[-1] != node_2:
                            matched_right.append(
                                tree2.lT.get_chain_of_node(node_2)[-1]
                            )
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=2)
        cmap = colormaps[colormap]
        c_norm = mcolors.Normalize(0, 1)
        colors1 = {c: cmap(c_norm(v)) for c, v in colors1.items()}
        colors2 = {c: cmap(c_norm(v)) for c, v in colors2.items()}
        tree1.lT.plot_subtree(
            tree1.lT.get_ancestor_at_t(n1),
            end_time=end_time1,
            size=size,
            color_of_nodes=colors1,
            color_of_edges=colors1,
            default_color=default_color,
            lw=lw,
            ax=ax[0],
        )
        tree2.lT.plot_subtree(
            tree2.lT.get_ancestor_at_t(n2),
            end_time=end_time2,
            size=size,
            color_of_nodes=colors2,
            color_of_edges=colors2,
            default_color=default_color,
            lw=lw,
            ax=ax[1],
        )
        return ax[0].get_figure(), ax

    def labelled_mappings(
        self,
        n1: int,
        embryo_1: str,
        n2: int,
        embryo_2: str,
        end_time1: int | None = None,
        end_time2: int | None = None,
        norm: Literal["max", "sum", None] = "max",
        style: (
            Literal["simple", "normalized_simple", "full", "downsampled"]
            | type[TreeApproximationTemplate]
        ) = "simple",
        downsample: int = 2,
    ) -> dict[str, list[str]]:
        """
        Returns the labels or IDs of all the nodes in the subtrees compared.

        Parameters
        ----------
        n1 : int
            id of the first node to compare
        embryo_1 : str
            the name of the first lineage
        n2 : int
            id of the second node to compare
        embryo_2: str
            the name of the second lineage
        end_time1 : int, optional
            the final time point the comparison algorithm will take into account for the first dataset.
            If None or not provided all nodes will be taken into account.
        end_time2 : int, optional
            the final time point the comparison algorithm will take into account for the first dataset.
            If None or not provided all nodes will be taken into account.
        norm : {"max", "sum"}, default="max"
            The normalization method to use.
        style : {"simple", "normalized_simple", "full", "downsampled"} or TreeApproximationTemplate subclass, default="simple"
            Which tree approximation is going to be used for the comparisons.
        downsample : int, default=2
            The downsample factor for the downsampled tree approximation.
            Used only when `style="downsampled"`.

        Returns
        -------
        dict mapping str to lists of str
            - 'matched' The labels of the matched nodes of the alignment.
            - 'unmatched' The labels of the unmatched nodes of the alginment.
        """

        parameters = (
            (end_time1, end_time2),
            convert_style_to_number(style, downsample),
        )
        n1_embryo, n2_embryo = sorted(
            ((n1, embryo_1), (n2, embryo_2)), key=lambda x: x[0]
        )
        self._comparisons.setdefault(parameters, {})
        if self._comparisons[parameters].get((n1, n2)):
            tmp = self._comparisons[parameters][(n1_embryo, n2_embryo)]
        else:
            tmp = self.__cross_lineage_edit_backtrace(
                n1,
                embryo_1,
                n2,
                embryo_2,
                end_time1,
                end_time2,
                style,
                norm,
                downsample,
            )
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
        if norm not in self.norm_dict:
            raise Warning(
                "Select a viable normalization method (max, sum, None)"
            )
        matched = []
        unmatched = []
        if style not in ("full", "downsampled"):
            for m in btrc:
                if m._left != -1 and m._right != -1:
                    cyc1 = tree1.lT.get_chain_of_node(corres1[m._left])
                    if len(cyc1) > 1:
                        node_1, *_ = cyc1
                    elif len(cyc1) == 1:
                        node_1 = cyc1.pop()

                    cyc2 = tree2.lT.get_chain_of_node(corres2[m._right])
                    if len(cyc2) > 1:
                        node_2, *_ = cyc2

                    elif len(cyc2) == 1:
                        node_2 = cyc2.pop()

                    matched.append(
                        (
                            tree1.lT.labels.get(node_1, node_1),
                            tree2.lT.labels.get(node_2, node_2),
                        )
                    )
                else:
                    if m._left != -1:
                        tmp_node = tree1.lT.get_chain_of_node(
                            corres1.get(m._left, "-")
                        )[0]
                        node_1 = (
                            tree1.lT.labels.get(tmp_node, tmp_node),
                            tree1.lT.name,
                        )
                    else:
                        tmp_node = tree2.lT.get_chain_of_node(
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
                        tmp_node = tree1.lT.get_chain_of_node(
                            corres1.get(m._left, "-")
                        )[0]
                        node_1 = (
                            tree1.lT.labels.get(tmp_node, tmp_node),
                            tree1.lT.name,
                        )
                    else:
                        tmp_node = tree2.lT.get_chain_of_node(
                            corres2.get(m._right, "-")
                        )[0]
                        node_1 = (
                            tree2.lT.labels.get(tmp_node, tmp_node),
                            tree2.lT.name,
                        )
                    unmatched.append(node_1)
        return {"matched": matched, "unmatched": unmatched}
