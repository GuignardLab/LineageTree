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
    """Container for several lineage trees, to compare them.

    Holds named [`LineageTree`][lineagetree.LineageTree] objects, typically
    one per embryo, and compares lineages across them with
    [`cross_lineage_edit_distance`][lineagetree.LineageTreeManager.cross_lineage_edit_distance].

    Trees are stored in ``self.lineagetrees``, keyed by name. If a tree has
    no name, one is generated (``"Lineagetree 0"``, ``"Lineagetree 1"``, …).
    Every tree must have its
    [`time_resolution`][lineagetree.LineageTree.time_resolution] set before
    comparisons, so that datasets recorded at different frame rates can be
    compared.

    Attributes
    ----------
    norm_dict : dict
        Mapping from normalization name to function.
    lineagetrees : dict of {str: LineageTree}
        The managed trees, keyed by name.

    Examples
    --------
    >>> from lineagetree import LineageTree, LineageTreeManager
    >>> lT1 = LineageTree(successor={0: [1, 2], 1: [], 2: []}, name="embryo1")
    >>> lT2 = LineageTree(
    ...     successor={0: [1, 2], 1: [3], 2: [], 3: []}, name="embryo2"
    ... )
    >>> lT1.time_resolution = lT2.time_resolution = 1
    >>> lTm = LineageTreeManager([lT1, lT2])
    >>> float(lTm.cross_lineage_edit_distance(0, "embryo1", 0, "embryo2"))
    0.25
    """

    norm_dict = {"max": max, "sum": sum, None: lambda x: 1}

    def __init__(self, lineagetree_list: Iterable[LineageTree] = ()):
        """Create a LineageTreeManager.

        Parameters
        ----------
        lineagetree_list : Iterable of LineageTree, default=()
            Optional initial collection of lineage trees to add.
        """
        self.lineagetrees: dict[str, LineageTree] = {}
        self.lineageTree_counter: int = 0
        self._comparisons: dict = {}
        for lT in lineagetree_list:
            self.add(lT)

    def __next__(self) -> int:
        """Return the next auto-increment counter value and advance it.

        Used internally to generate unique fallback names for unnamed trees.

        Returns
        -------
        int
            Current counter value (before incrementing).
        """
        self.lineageTree_counter += 1
        return self.lineageTree_counter - 1

    def __len__(self) -> int:
        """Return how many lineage trees are in the manager.

        Returns
        -------
        int
            The number of trees inside the manager.
        """
        return len(self.lineagetrees)

    def __iter__(self) -> Generator[tuple[str, LineageTree], None, None]:
        """Iterate over ``(name, LineageTree)`` pairs.

        Yields
        ------
        tuple of (str, LineageTree)
            Name and corresponding tree for each entry in the manager.
        """
        yield from self.lineagetrees.items()

    def __getitem__(self, key: str) -> LineageTree:
        """Return the :class:`~lineagetree.LineageTree` stored under ``key``.

        Parameters
        ----------
        key : str
            Name of the tree to retrieve.

        Returns
        -------
        LineageTree
            The tree associated with ``key``.

        Raises
        ------
        KeyError
            If ``key`` is not present in the manager.
        """
        if key in self.lineagetrees:
            return self.lineagetrees[key]
        else:
            raise KeyError(f"'{key}' not found in the manager")

    @property
    def gcd(self) -> int:
        """Greatest common divisor of the time resolutions of all the trees.

        Computed on ``lT._time_resolution``, i.e. ten times
        ``lT.time_resolution``. 1 when the manager holds a single tree.

        Raises
        ------
        ValueError
            If the manager is empty.
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
        """Add a lineage tree to the manager.

        A tree equal to one already in the manager (same topology and times)
        is not added again. A tree added under an existing name replaces the
        tree stored under that name.

        Parameters
        ----------
        other_tree : LineageTree
            The lineage tree to add.
        name : str, default=""
            The name to store the tree under. If empty, the tree's own name
            is used (loaders name trees after their file), or one is
            generated and given to the tree.

        Returns
        -------
        bool or None
            False if an equal tree is already in the manager, None otherwise.

        Raises
        ------
        Exception
            If ``other_tree`` is not a ``LineageTree``.
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
        """Add ``other`` to the manager in place; see ``add``.

        Returns None, so write ``lTm + lT`` as a statement, not
        ``lTm = lTm + lT``.
        """
        self.add(other)

    def write(self, fname: str):
        """Save the manager and all its trees to a ``.lTM`` file.

        Parameters
        ----------
        fname : str
            The path and name of the file to save. The ``.lTM`` extension is
            added if missing.
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
        """Remove a lineage tree from the manager.

        Parameters
        ----------
        key : str
            The name of the lineage tree to remove. Unknown names are
            ignored.
        """
        self.lineagetrees.pop(key, None)

    @classmethod
    def load(cls, fname: str) -> LineageTreeManager:
        """Load a lineage tree manager from a ``.lTM`` file.

        Parameters
        ----------
        fname : str
            Path to and name of the file to read.

        Returns
        -------
        LineageTreeManager
            The loaded manager.

        Warnings
        --------
        ``.lTM`` files are pickles: only load files from sources you trust.
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
            Literal[
                "simple", "normalized_simple", "full", "downsampled", "mini"
            ]
            | type[TreeApproximationTemplate]
        ) = "simple",
        norm: Literal["max", "sum", None] = "max",
        downsample: int = 2,
    ) -> dict[
        str,
        Alignment
        | tuple[TreeApproximationTemplate, TreeApproximationTemplate],
    ]:
        """Compute the unordered tree edit alignment between two lineages.

        The trees spawned by node ``n1`` of ``embryo_1`` and node ``n2`` of
        ``embryo_2`` are compared with the unordered tree edit distance of
        Zhang (1996). The result is cached in ``self._comparisons``.

        Parameters
        ----------
        n1 : int
            Node of the first lineage tree.
        embryo_1 : str
            The key/name of the first lineage tree.
        n2 : int
            Node of the second lineage tree.
        embryo_2 : str
            The key/name of the second lineage tree.
        end_time1 : int, optional
            The final time point the comparison algorithm takes into account for
            the first dataset. If None, all nodes are taken into account.
        end_time2 : int, optional
            The final time point the comparison algorithm takes into account for
            the second dataset. If None, all nodes are taken into account.
        style : {"simple", "normalized_simple", "full", "downsampled", "mini"} or TreeApproximationTemplate subclass, default="simple"
            The approximation used to calculate the tree.
        norm : {"max", "sum", None}, default="max"
            Not used.
        downsample : int, default=2
            The downsample factor for the downsampled tree approximation.
            Used only when ``style="downsampled"``.

        Returns
        -------
        dict of {str: Alignment or tuple of TreeApproximationTemplate}
            A dictionary with two keys:

            - ``'alignment'``: the alignment between the nodes of the subtrees
              spawned by ``n1`` and ``n2``,
            - ``'trees'``: the two trees that have been mapped to each other.
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
        """Calculate the distance of the subtree of a node matched in a comparison.

        This does not calculate the distance from scratch but reuses the
        existing alignment.

        Parameters
        ----------
        node1 : int
            The root of the first subtree.
        lT1 : LineageTree
            The dataset the first lineage belongs to.
        node2 : int
            The root of the second subtree.
        lT2 : LineageTree
            The dataset the second lineage belongs to.
        alignment : Alignment
            The alignment of the subtree.
        corres1 : dict
            The correspondence dictionary of the first lineage.
        corres2 : dict
            The correspondence dictionary of the second lineage.
        delta_tmp : Callable
            The delta function for the comparisons.
        norm : Callable
            How the lineages should be normalized.
        norm1 : int or float
            The result of the normalization of the first tree.
        norm2 : int or float
            The result of the normalization of the second tree.

        Returns
        -------
        float
            The result of the comparison of the subtree.
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

    def clear_comparisons(self) -> None:
        """Clear all cached cross-lineage tree-edit-distance comparisons.

        Frees memory by erasing the cached alignment results stored in
        ``self._comparisons``. Call this when the cache grows too large or
        after modifying trees in the manager, since cached alignments are not
        updated.
        """
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
            Literal[
                "simple", "normalized_simple", "full", "downsampled", "mini"
            ]
            | type[TreeApproximationTemplate]
        ) = "simple",
        downsample: int = 2,
        return_norms: bool = False,
    ) -> float | tuple[float, tuple[float, float]]:
        """Compute the unordered tree edit distance between two lineages.

        This is the cross-dataset version of
        [`LineageTree.unordered_tree_edit_distance`][lineagetree.LineageTree.unordered_tree_edit_distance]:
        the subtree spawned by ``n1`` in ``embryo_1`` is compared with the
        subtree spawned by ``n2`` in ``embryo_2``. Durations are rescaled
        with the time resolution of each tree, which must be set.

        The alignment is cached; see
        [`clear_comparisons`][lineagetree.LineageTreeManager.clear_comparisons].

        Parameters
        ----------
        n1 : int
            The node spawning the first subtree.
        embryo_1 : str
            The name of the tree ``n1`` belongs to (a key of
            ``lTm.lineagetrees``).
        n2 : int
            The node spawning the second subtree.
        embryo_2 : str
            The name of the tree ``n2`` belongs to.
        end_time1 : int, optional
            The final time point the comparison algorithm takes into account for
            the first dataset. If None, all nodes are taken into account.
        end_time2 : int, optional
            The final time point the comparison algorithm takes into account for
            the second dataset. If None, all nodes are taken into account.
        norm : {"max", "sum", None}, default="max"
            How the cost is normalised: ``"max"`` divides it by the larger of
            the two tree norms, ``"sum"`` by their sum, and None does not
            normalise.
        style : {"simple", "normalized_simple", "full", "downsampled", "mini"} or TreeApproximationTemplate subclass, default="simple"
            The tree approximation used for the comparison; see
            [`tree_style`][lineagetree.tree_approximation.tree_style].
        downsample : int, default=2
            The downsample factor for the downsampled tree approximation.
            Used only when ``style="downsampled"``.
        return_norms : bool, default=False
            If True, return the cost before normalisation together with the
            norms of the two trees, instead of the normalised distance
            (mainly used by the napari plugin).

        Returns
        -------
        float or tuple of (float, tuple of (float, float))
            The normalised distance between the two subtrees. With
            ``return_norms=True``, the tuple ``(cost, (norm1, norm2))``, where
            ``cost`` is not normalised.

        Raises
        ------
        Warning
            If the time resolution of either tree is not set.
        ValueError
            If ``norm`` is not one of the values above.
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
            Literal[
                "simple", "normalized_simple", "full", "downsampled", "mini"
            ]
            | type[TreeApproximationTemplate]
        ) = "simple",
        downsample: int = 2,
        colormap: str = "cool",
        default_color: str = "black",
        size: float = 10,
        lw: float = 0.3,
        ax: np.ndarray | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> tuple[plt.Figure, np.ndarray]:
        """Plot two compared lineages, coloured by how well they match.

        The comparison is the one of
        [`cross_lineage_edit_distance`][lineagetree.LineageTreeManager.cross_lineage_edit_distance].
        Each matched chain is coloured by the normalised distance between the
        subtrees spawned by the two chains it is matched with.

        Parameters
        ----------
        n1 : int
            The node spawning the first subtree.
        embryo_1 : str
            The name of the tree ``n1`` belongs to.
        n2 : int
            The node spawning the second subtree.
        embryo_2 : str
            The name of the tree ``n2`` belongs to.
        end_time1 : int, optional
            The final time point the comparison algorithm takes into account for
            the first dataset. If None, all nodes are taken into account.
        end_time2 : int, optional
            The final time point the comparison algorithm takes into account for
            the second dataset. If None, all nodes are taken into account.
        norm : {"max", "sum", None}, default="max"
            The normalization method; see
            [`cross_lineage_edit_distance`][lineagetree.LineageTreeManager.cross_lineage_edit_distance].
        style : {"simple", "normalized_simple", "full", "downsampled", "mini"} or TreeApproximationTemplate subclass, default="simple"
            The tree approximation used for the comparison.
        downsample : int, default=2
            The downsample factor for the downsampled tree approximation.
            Used only when ``style="downsampled"``.
        colormap : str, default="cool"
            Name of the matplotlib colormap used for matched nodes.
        default_color : str, default="black"
            The colour of the unmatched nodes.
        size : float, default=10
            The size of the nodes.
        lw : float, default=0.3
            The width of the edges.
        ax : numpy.ndarray of plt.Axes, optional
            Two axes, one per lineage. If None, a new figure with two axes is
            created.
        vmin, vmax : float, optional
            Distances in ``[vmin, vmax]`` are mapped linearly onto the
            colormap. ``vmin`` defaults to the 5th percentile and ``vmax`` to
            the 95th percentile of the distances.

        Returns
        -------
        plt.Figure
            The matplotlib figure.
        numpy.ndarray of plt.Axes
            The two axes.
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
        if vmin is None:
            vmin = np.percentile(list(colors1.values()), 5)
        if vmax is None:
            vmax = np.percentile(list(colors1.values()), 95)
        c_norm = mcolors.Normalize(vmin, vmax)
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
            Literal[
                "simple", "normalized_simple", "full", "downsampled", "mini"
            ]
            | type[TreeApproximationTemplate]
        ) = "simple",
        downsample: int = 2,
    ) -> dict[str, list]:
        """List which nodes are matched when comparing two lineages.

        The comparison is the one of
        [`cross_lineage_edit_distance`][lineagetree.LineageTreeManager.cross_lineage_edit_distance].
        Nodes are reported by their label or, when they have none, by their
        id. With chain-based styles, each chain is reported by its first
        node.

        Parameters
        ----------
        n1 : int
            The node spawning the first subtree.
        embryo_1 : str
            The name of the tree ``n1`` belongs to.
        n2 : int
            The node spawning the second subtree.
        embryo_2 : str
            The name of the tree ``n2`` belongs to.
        end_time1 : int, optional
            The final time point the comparison algorithm takes into account for
            the first dataset. If None, all nodes are taken into account.
        end_time2 : int, optional
            The final time point the comparison algorithm takes into account for
            the second dataset. If None, all nodes are taken into account.
        norm : {"max", "sum", None}, default="max"
            The normalization method.
        style : {"simple", "normalized_simple", "full", "downsampled", "mini"} or TreeApproximationTemplate subclass, default="simple"
            The tree approximation used for the comparison.
        downsample : int, default=2
            The downsample factor for the downsampled tree approximation.
            Used only when ``style="downsampled"``.

        Returns
        -------
        dict
            A dictionary with two keys:

            - ``'matched'``: list of ``(node_of_tree1, node_of_tree2)``
              pairs,
            - ``'unmatched'``: list of ``(node, tree_name)`` pairs for the
              nodes of either tree that have no match.
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
