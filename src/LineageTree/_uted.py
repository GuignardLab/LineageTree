from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import partial
from itertools import combinations
from typing import TYPE_CHECKING, Literal

import matplotlib.colors as mcolors
from edist import uted
from matplotlib import colormaps
from matplotlib import pyplot as plt

from .tree_approximation import TreeApproximationTemplate, tree_style
from .utils import convert_style_to_number

if TYPE_CHECKING:
    from edist.alignment import Alignment


def unordered_tree_edit_distances_at_time_t(
    self,
    t: int,
    end_time: int | None = None,
    style: (
        Literal["simple", "full", "downsampled", "normalized_simple"]
        | type[TreeApproximationTemplate]
    ) = "simple",
    downsample: int = 2,
    norm: Literal["max", "sum", None] = "max",
    recompute: bool = False,
) -> dict[tuple[int, int], float]:
    """Compute all the pairwise unordered tree edit distances from Zhang 996 between the trees spawned at time `t`

    Parameters
    ----------
    t : int
        time to look at
    end_time : int
        The final time point the comparison algorithm will take into account.
        If None all nodes will be taken into account.
    style : {"simple", "full", "downsampled", "normalized_simple"} or TreeApproximationTemplate subclass, default="simple"
        Which tree approximation is going to be used for the comparisons.
    downsample : int, default=2
        The downsample factor for the downsampled tree approximation.
        Used only when `style="downsampled"`.
    norm : {"max", "sum"}, default="max"
        The normalization method to use.
    recompute : bool, default=False
        If True, forces to recompute the distances

    Returns
    -------
    dict mapping a tuple of tuple that contains 2 ints to float
        a dictionary that maps a pair of node ids at time `t` to their unordered tree edit distance
    """
    if not hasattr(self, "uted"):
        self.uted = {}
    elif t in self.uted and not recompute:
        return self.uted[t]
    self.uted[t] = {}
    roots = self.nodes_at_t(t=t)
    for n1, n2 in combinations(roots, 2):
        key = tuple(sorted((n1, n2)))
        self.uted[t][key] = self.unordered_tree_edit_distance(
            n1,
            n2,
            end_time=end_time,
            style=style,
            downsample=downsample,
            norm=norm,
        )
    return self.uted[t]


def __calculate_distance_of_sub_tree(
    self,
    node1: int,
    node2: int,
    alignment: Alignment,
    corres1: dict[int, int],
    corres2: dict[int, int],
    delta_tmp: Callable,
    norm: Callable,
    norm1: int | float,
    norm2: int | float,
) -> float:
    """Calculates the distance of the subtree of each node matched in a comparison.
    DOES NOT CALCULATE THE DISTANCE FROM SCRATCH BUT USING THE ALIGNMENT.
    TODO ITS BOUND TO CHANGE
    Parameters
    ----------
    node1 : int
        The root of the first subtree
    node2 : int
        The root of the second subtree
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
    sub_tree_1 = set(self.get_subtree_nodes(node1))
    sub_tree_2 = set(self.get_subtree_nodes(node2))
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


def __unordereded_backtrace(
    self,
    n1: int,
    n2: int,
    end_time: int | None = None,
    norm: Literal["max", "sum", None] = "max",
    style: (
        Literal["simple", "normalized_simple", "full", "downsampled"]
        | type[TreeApproximationTemplate]
    ) = "simple",
    downsample: int = 2,
) -> dict[
    str,
    Alignment | tuple[TreeApproximationTemplate, TreeApproximationTemplate],
]:
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
    style : {"simple", "full", "downsampled", "normalized_simple"} or TreeApproximationTemplate subclass, default="simple"
        Which tree approximation is going to be used for the comparisons.
    downsample : int, default=2
        The downsample factor for the downsampled tree approximation.
        Used only when `style="downsampled"`.

    Returns
    -------
    dict mapping str to Alignment or tuple of [TreeApproximationTemplate, TreeApproximationTemplate]
        - 'alignment'
            The alignment between the nodes by the subtrees spawned by the nodes n1,n2 and the normalization function.
        - 'trees'
            A list of the two trees that have been mapped to each other.
    """

    parameters = (
        end_time,
        convert_style_to_number(style=style, downsample=downsample),
    )
    n1, n2 = sorted([n1, n2])
    self._comparisons.setdefault(parameters, {})
    if len(self._comparisons) > 100:
        warnings.warn(
            "More than 100 comparisons are saved, use clear_comparisons() to delete them.",
            stacklevel=2,
        )
    if isinstance(style, str):
        tree = tree_style[style].value
    elif issubclass(style, TreeApproximationTemplate):
        tree = style
    else:
        raise ValueError("Please use a valid approximation.")
    tree1 = tree(
        lT=self,
        downsample=downsample,
        end_time=end_time,
        root=n1,
        time_scale=1,
    )
    tree2 = tree(
        lT=self,
        downsample=downsample,
        end_time=end_time,
        root=n2,
        time_scale=1,
    )
    delta = tree1.delta
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
        self._comparisons[parameters][(n1, n2)] = {
            "alignment": (),
            "trees": (),
        }
        return self._comparisons[parameters][(n1, n2)]
    delta_tmp = partial(
        delta,
        corres1=corres1,
        corres2=corres2,
        times1=times1,
        times2=times2,
    )
    btrc = uted.uted_backtrace(nodes1, adj1, nodes2, adj2, delta=delta_tmp)

    self._comparisons[parameters][(n1, n2)] = {
        "alignment": btrc,
        "trees": (tree1, tree2),
    }
    return self._comparisons[parameters][(n1, n2)]


def unordered_tree_edit_distance(
    self,
    n1: int,
    n2: int,
    end_time: int | None = None,
    norm: Literal["max", "sum", None] = "max",
    style: (
        Literal["simple", "normalized_simple", "full", "downsampled"]
        | type[TreeApproximationTemplate]
    ) = "simple",
    downsample: int = 2,
    return_norms: bool = False,
) -> float | tuple[float, tuple[float, float]]:
    """
    Compute the unordered tree edit distance from Zhang 1996 between the trees spawned
    by two nodes `n1` and `n2`. The topology of the trees are compared and the matching
    cost is given by the function delta (see edist doc for more information).
    The distance is normed by the function norm that takes the two list of nodes
    spawned by the trees `n1` and `n2`.

    Parameters
    ----------
    n1 : int
        id of the first node to compare
    n2 : int
        id of the second node to compare
    end_time : int, optional
        The final time point the comparison algorithm will take into account.
        If None or not provided all nodes will be taken into account.
    norm : {"max", "sum"}, default="max"
        The normalization method to use, defaults to 'max'.
    style : {"simple", "normalized_simple", "full", "downsampled"} or TreeApproximationTemplate subclass, default="simple"
        Which tree approximation is going to be used for the comparisons.
    downsample : int, default=2
        The downsample factor for the downsampled tree approximation.
        Used only when `style="downsampled"`.

    Returns
    -------
    float
        The normalized unordered tree edit distance between `n1` and `n2`
    """
    parameters = (
        end_time,
        convert_style_to_number(style=style, downsample=downsample),
    )
    n1, n2 = sorted([n1, n2])
    self._comparisons.setdefault(parameters, {})
    if self._comparisons[parameters].get((n1, n2)):
        tmp = self._comparisons[parameters][(n1, n2)]
    else:
        tmp = self.__unordereded_backtrace(
            n1, n2, end_time, norm, style, downsample
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
    n2: int,
    end_time: int | None = None,
    norm: Literal["max", "sum", None] = "max",
    style: (
        Literal["simple", "normalized_simple", "full", "downsampled"]
        | type[TreeApproximationTemplate]
    ) = "simple",
    downsample: int = 2,
    colormap: str = "cool",
    default_color: str = "black",
    size: float = 10,
    lw: float = 0.3,
    ax: list[plt.Axes] | None = None,
) -> tuple[plt.figure, plt.Axes]:
    """
    Plots the subtrees compared and colors them according to the quality of the matching of their subtree.

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
    style : {"simple", "full", "downsampled", "normalized_simple} or TreeApproximationTemplate subclass, default="simple"
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
            The figure of the plot
    plt.Axes
            The axes of the plot
    """
    parameters = (
        end_time,
        convert_style_to_number(style=style, downsample=downsample),
    )
    n1, n2 = sorted([n1, n2])
    self._comparisons.setdefault(parameters, {})
    if self._comparisons[parameters].get((n1, n2)):
        tmp = self._comparisons[parameters][(n1, n2)]
    else:
        tmp = self.__unordereded_backtrace(
            n1, n2, end_time, norm, style, downsample
        )
    btrc: Alignment = tmp["alignment"]
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
        raise Warning("Select a viable normalization method (max, sum, None)")
    matched_right = []
    matched_left = []
    colors = {}
    if style not in ("full", "downsampled"):
        for m in btrc:
            if m._left != -1 and m._right != -1:
                cyc1 = self.get_chain_of_node(corres1[m._left])
                if len(cyc1) > 1:
                    node_1, *_, l_node_1 = cyc1
                    matched_left.append(node_1)
                    matched_left.append(l_node_1)
                elif len(cyc1) == 1:
                    node_1 = l_node_1 = cyc1.pop()
                    matched_left.append(node_1)

                cyc2 = self.get_chain_of_node(corres2[m._right])
                if len(cyc2) > 1:
                    node_2, *_, l_node_2 = cyc2
                    matched_right.append(node_2)
                    matched_right.append(l_node_2)

                elif len(cyc2) == 1:
                    node_2 = l_node_2 = cyc2.pop()
                    matched_right.append(node_2)

                colors[node_1] = self.__calculate_distance_of_sub_tree(
                    node_1,
                    node_2,
                    btrc,
                    corres1,
                    corres2,
                    delta_tmp,
                    self.norm_dict[norm],
                    tree1.get_norm(node_1),
                    tree2.get_norm(node_2),
                )
                colors[node_2] = colors[node_1]
                colors[l_node_1] = colors[node_1]
                colors[l_node_2] = colors[node_2]
    else:
        for m in btrc:
            if m._left != -1 and m._right != -1:
                node_1 = corres1[m._left]
                node_2 = corres2[m._right]

                if (
                    self.get_chain_of_node(node_1)[0] == node_1
                    or self.get_chain_of_node(node_2)[0] == node_2
                    and (node_1 not in colors or node_2 not in colors)
                ):
                    matched_left.append(node_1)
                    l_node_1 = self.get_chain_of_node(node_1)[-1]
                    matched_left.append(l_node_1)
                    matched_right.append(node_2)
                    l_node_2 = self.get_chain_of_node(node_2)[-1]
                    matched_right.append(l_node_2)
                    colors[node_1] = self.__calculate_distance_of_sub_tree(
                        node_1,
                        node_2,
                        btrc,
                        corres1,
                        corres2,
                        delta_tmp,
                        self.norm_dict[norm],
                        tree1.get_norm(node_1),
                        tree2.get_norm(node_2),
                    )
                    colors[l_node_1] = colors[node_1]
                    colors[node_2] = colors[node_1]
                    colors[l_node_2] = colors[node_1]
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
    cmap = colormaps[colormap]
    c_norm = mcolors.Normalize(0, 1)
    colors = {c: cmap(c_norm(v)) for c, v in colors.items()}
    self.plot_subtree(
        self.get_ancestor_at_t(n1),
        end_time=end_time,
        size=size,
        selected_nodes=matched_left,
        color_of_nodes=colors,
        selected_edges=matched_left,
        color_of_edges=colors,
        default_color=default_color,
        lw=lw,
        ax=ax[0],
    )
    self.plot_subtree(
        self.get_ancestor_at_t(n2),
        end_time=end_time,
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
