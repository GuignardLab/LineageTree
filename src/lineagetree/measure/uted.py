"""Unordered tree edit distance (UTED) between the subtrees of a dataset.

The distance is the constrained unordered tree edit distance of Zhang (1996),
computed by ``edist``. The subtrees are first simplified with a tree style
(see [`tree_style`][lineagetree.tree_approximation.tree_style]), and the
alignments are cached on the tree so that the distance, its plot and its
mappings are computed only once per pair of nodes.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import partial
from itertools import combinations
from typing import TYPE_CHECKING, Literal

import numpy as np
import matplotlib.colors as mcolors
from edist import uted
from matplotlib import colormaps
from matplotlib import pyplot as plt

from ..tree_approximation import TreeApproximationTemplate, tree_style
from .._core.utils import convert_style_to_number

if TYPE_CHECKING:
    from edist.alignment import Alignment
    from ..lineage_tree import LineageTree


norm_dict = {"max": max, "sum": sum, None: lambda x: 1}


def unordered_tree_edit_distances_at_time_t(
    lT: LineageTree,
    t: int,
    end_time: int | None = None,
    style: (
        Literal["simple", "full", "downsampled", "normalized_simple", "mini"]
        | type[TreeApproximationTemplate]
    ) = "simple",
    downsample: int = 2,
    norm: Literal["max", "sum", None] = "max",
    recompute: bool = False,
) -> dict[tuple[int, int], float]:
    """Compute the tree edit distance between every pair of nodes at time ``t``.

    Each pair of nodes at time ``t`` is compared with
    [`unordered_tree_edit_distance`][lineagetree.LineageTree.unordered_tree_edit_distance].
    The result is cached in ``lT.uted[t]``.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    t : int
        The time point whose nodes are compared.
    end_time : int, optional
        The last time point taken into account. If None, the whole subtrees
        are compared.
    style : {"simple", "normalized_simple", "full", "downsampled", "mini"} or TreeApproximationTemplate subclass, default="simple"
        The tree approximation used for the comparison; see
        [`tree_style`][lineagetree.tree_approximation.tree_style].
    downsample : int, default=2
        The downsample factor for the downsampled tree approximation.
        Used only when ``style="downsampled"``.
    norm : {"max", "sum", None}, default="max"
        The normalization method; see
        [`unordered_tree_edit_distance`][lineagetree.LineageTree.unordered_tree_edit_distance].
    recompute : bool, default=False
        If True, recompute the distances even if ``lT.uted[t]`` exists.
        The cache does not depend on the other parameters, so use it after
        changing any of them.

    Returns
    -------
    dict of {tuple of (int, int): float}
        Maps each pair of nodes at time ``t``, as a sorted tuple, to their
        distance.
    """
    if not hasattr(lT, "uted"):
        lT.uted = {}
    elif t in lT.uted and not recompute:
        return lT.uted[t]
    lT.uted[t] = {}
    roots = lT.time_nodes[t]
    for n1, n2 in combinations(roots, 2):
        key = tuple(sorted((n1, n2)))
        lT.uted[t][key] = lT.unordered_tree_edit_distance(
            n1,
            n2,
            end_time=end_time,
            style=style,
            downsample=downsample,
            norm=norm,
        )
    return lT.uted[t]


def __calculate_distance_of_sub_tree(
    lT: LineageTree,
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
    """Calculate the distance of the subtree of a node matched in a comparison.

    This does not calculate the distance from scratch but reuses the
    existing alignment.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    node1 : int
        The root of the first subtree
    node2 : int
        The root of the second subtree
    alignment : Alignment
        The alignment of the subtree
    corres1 : dict
        The correspondence dictionary of the first lineage
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
    sub_tree_1 = set(lT.get_subtree_nodes(node1))
    sub_tree_2 = set(lT.get_subtree_nodes(node2))
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


def clear_comparisons(lT: LineageTree) -> None:
    """Clear all cached tree-edit-distance comparisons.

    Comparisons between subtrees are stored in ``lT._comparisons`` keyed by
    ``(end_time, style_id)`` to avoid redundant recomputation. This function
    empties the cache. Call it when memory usage is a concern, or after
    modifying the tree, since cached alignments are not updated.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance whose comparison cache should be cleared.
    """
    lT._comparisons.clear()


def __unordereded_backtrace(
    lT: LineageTree,
    n1: int,
    n2: int,
    end_time: int | None = None,
    norm: Literal["max", "sum", None] = "max",
    style: (
        Literal["simple", "normalized_simple", "full", "downsampled", "mini"]
        | type[TreeApproximationTemplate]
    ) = "simple",
    downsample: int = 2,
) -> dict[
    str,
    Alignment | tuple[TreeApproximationTemplate, TreeApproximationTemplate],
]:
    """Compute the unordered tree edit alignment between two subtrees.

    The trees spawned by ``n1`` and ``n2`` are compared with the unordered
    tree edit distance of Zhang (1996). The result is cached in
    ``lT._comparisons``.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    n1 : int
        id of the first node to compare
    n2 : int
        id of the second node to compare
    end_time : int, optional
        The last time point taken into account. If None, the whole subtrees
        are compared.
    norm : {"max", "sum", None}, default="max"
        Not used.
    style : {"simple", "normalized_simple", "full", "downsampled", "mini"} or TreeApproximationTemplate subclass, default="simple"
        The tree approximation used for the comparison.
    downsample : int, default=2
        The downsample factor for the downsampled tree approximation.
        Used only when ``style="downsampled"``.

    Returns
    -------
    dict
        A dictionary with two keys:

        - ``'alignment'``: the ``edist`` alignment between the two subtrees,
        - ``'trees'``: the two tree approximations that were aligned.

        Both are empty tuples when the two subtrees are empty.
    """

    parameters = (
        end_time,
        convert_style_to_number(style=style, downsample=downsample),
    )
    n1, n2 = sorted([n1, n2])
    lT._comparisons.setdefault(parameters, {})
    if len(lT._comparisons[parameters]) > 100:
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
        lT=lT,
        downsample=downsample,
        end_time=end_time,
        root=n1,
        time_scale=1,
    )
    tree2 = tree(
        lT=lT,
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
        lT._comparisons[parameters][(n1, n2)] = {
            "alignment": (),
            "trees": (),
        }
        return lT._comparisons[parameters][(n1, n2)]
    delta_tmp = partial(
        delta,
        corres1=corres1,
        corres2=corres2,
        times1=times1,
        times2=times2,
    )
    btrc = uted.uted_backtrace(nodes1, adj1, nodes2, adj2, delta=delta_tmp)

    lT._comparisons[parameters][(n1, n2)] = {
        "alignment": btrc,
        "trees": (tree1, tree2),
    }
    return lT._comparisons[parameters][(n1, n2)]


def unordered_tree_edit_distance(
    lT: LineageTree,
    n1: int,
    n2: int,
    end_time: int | None = None,
    norm: Literal["max", "sum", None] = "max",
    style: (
        Literal["simple", "normalized_simple", "full", "downsampled", "mini"]
        | type[TreeApproximationTemplate]
    ) = "simple",
    downsample: int = 2,
    return_norms: bool = False,
) -> float | tuple[float, tuple[float, float]]:
    """Compute the unordered tree edit distance between two subtrees.

    The subtrees spawned by ``n1`` and ``n2`` are simplified according to
    ``style``, then compared with the constrained unordered tree edit
    distance of Zhang (1996): the minimal total cost of the node insertions,
    deletions and substitutions that turn one tree into the other, where
    the order of the successors does not matter. The cost of each operation
    is given by the ``delta`` method of the style. The cost is then divided
    by a norm computed from both trees, so that trees of different sizes can
    be compared.

    The alignment is cached; see
    [`clear_comparisons`][lineagetree.LineageTree.clear_comparisons].

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    n1 : int
        The node spawning the first subtree.
    n2 : int
        The node spawning the second subtree.
    end_time : int, optional
        The last time point taken into account. If None, the whole subtrees
        are compared.
    norm : {"max", "sum", None}, default="max"
        How the cost is normalised, from the norms of the two trees (see the
        ``get_norm`` method of the style): ``"max"`` divides by the larger
        one, ``"sum"`` by their sum, and None does not normalise.
    style : {"simple", "normalized_simple", "full", "downsampled", "mini"} or TreeApproximationTemplate subclass, default="simple"
        The tree approximation used for the comparison; see
        [`tree_style`][lineagetree.tree_approximation.tree_style].
    downsample : int, default=2
        The downsample factor for the downsampled tree approximation.
        Used only when ``style="downsampled"``.
    return_norms : bool, default=False
        If True, return the cost before normalisation together with the
        norms of the two trees, instead of the normalised distance.

    Returns
    -------
    float or tuple of (float, tuple of (float, float))
        The normalised distance between the subtrees of ``n1`` and ``n2``.
        With ``return_norms=True``, the tuple ``(cost, (norm1, norm2))``,
        where ``cost`` is not normalised.

    Raises
    ------
    ValueError
        If ``norm`` is not one of the values above.

    Examples
    --------
    >>> from lineagetree import LineageTree
    >>> lT = LineageTree(successor={0: [1, 2], 1: [], 2: [3], 3: []})
    >>> float(lT.unordered_tree_edit_distance(1, 2, norm="max"))
    0.5
    """
    parameters = (
        end_time,
        convert_style_to_number(style=style, downsample=downsample),
    )
    n1, n2 = sorted([n1, n2])
    lT._comparisons.setdefault(parameters, {})
    if lT._comparisons[parameters].get((n1, n2)):
        tmp = lT._comparisons[parameters][(n1, n2)]
    else:
        tmp = __unordereded_backtrace(
            lT, n1, n2, end_time, norm, style, downsample
        )
    if not tmp["trees"]:
        return (0, (0, 0)) if return_norms else 0
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

    if norm not in lT._norm_dict:
        raise ValueError(
            "Select a viable normalization method (max, sum, None)"
        )
    cost = btrc.cost(nodes1, nodes2, delta_tmp)
    norm_values = (tree1.get_norm(n1), tree2.get_norm(n2))
    if return_norms:
        return cost, norm_values
    return cost / lT._norm_dict[norm](norm_values)


def plot_tree_distance_graphs(
    lT: LineageTree,
    n1: int,
    n2: int,
    end_time: int | None = None,
    norm: Literal["max", "sum", None] = "max",
    style: (
        Literal["simple", "normalized_simple", "full", "downsampled", "mini"]
        | type[TreeApproximationTemplate]
    ) = "simple",
    downsample: int = 2,
    colormap: str = "cool",
    default_color: str = "black",
    size: float = 10,
    lw: float = 0.3,
    ax: list[plt.Axes] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plot two compared subtrees, coloured by how well they match.

    The comparison is the one of
    [`unordered_tree_edit_distance`][lineagetree.LineageTree.unordered_tree_edit_distance].
    Each matched chain is coloured by the normalised distance between the
    subtrees spawned by the two chains it is matched with; unmatched chains
    are drawn in ``default_color``. The whole lineages containing ``n1`` and
    ``n2`` are drawn, from their ancestor at the first time point.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    n1 : int
        The node spawning the first subtree.
    n2 : int
        The node spawning the second subtree.
    end_time : int, optional
        The last time point taken into account. If None, the whole subtrees
        are compared.
    norm : {"max", "sum", None}, default="max"
        The normalization method; see
        [`unordered_tree_edit_distance`][lineagetree.LineageTree.unordered_tree_edit_distance].
    style : {"simple", "normalized_simple", "full", "downsampled", "mini"} or TreeApproximationTemplate subclass, default="simple"
        The tree approximation used for the comparison; see
        [`tree_style`][lineagetree.tree_approximation.tree_style].
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
    ax : list of plt.Axes, optional
        Two axes, one per subtree. If None, a new figure with two axes is
        created.
    vmin, vmax : float, optional
        Distances in ``[vmin, vmax]`` are mapped linearly onto the colormap.
        ``vmin`` defaults to the 5th percentile and ``vmax`` to the 95th
        percentile of the distances.

    Returns
    -------
    plt.Figure
        The figure of the plot.
    list of plt.Axes
        The two axes of the plot.
    """
    parameters = (
        end_time,
        convert_style_to_number(style=style, downsample=downsample),
    )
    n1, n2 = sorted([n1, n2])
    lT._comparisons.setdefault(parameters, {})
    if lT._comparisons[parameters].get((n1, n2)):
        tmp = lT._comparisons[parameters][(n1, n2)]
    else:
        tmp = __unordereded_backtrace(
            lT, n1, n2, end_time, norm, style, downsample
        )
    if not tmp["trees"]:
        fig, ax = (
            plt.subplots(nrows=1, ncols=2, sharey=True)
            if ax is None
            else (ax[0].get_figure(), ax)
        )
        return fig, ax
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

    if norm not in lT._norm_dict:
        raise ValueError(
            "Select a viable normalization method (max, sum, None)"
        )
    matched_right = []
    matched_left = []
    colors = {}
    if style not in ("full", "downsampled"):
        for m in btrc:
            if m._left != -1 and m._right != -1:
                cyc1 = lT.get_chain_of_node(corres1[m._left])
                if len(cyc1) > 1:
                    node_1, *_, l_node_1 = cyc1
                    matched_left.append(node_1)
                    matched_left.append(l_node_1)
                elif len(cyc1) == 1:
                    node_1 = l_node_1 = cyc1.pop()
                    matched_left.append(node_1)

                cyc2 = lT.get_chain_of_node(corres2[m._right])
                if len(cyc2) > 1:
                    node_2, *_, l_node_2 = cyc2
                    matched_right.append(node_2)
                    matched_right.append(l_node_2)

                elif len(cyc2) == 1:
                    node_2 = l_node_2 = cyc2.pop()
                    matched_right.append(node_2)

                colors[node_1] = __calculate_distance_of_sub_tree(
                    lT,
                    node_1,
                    node_2,
                    btrc,
                    corres1,
                    corres2,
                    delta_tmp,
                    lT._norm_dict[norm],
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
                    lT.get_chain_of_node(node_1)[0] == node_1
                    or lT.get_chain_of_node(node_2)[0] == node_2
                ) and (node_1 not in colors or node_2 not in colors):
                    matched_left.append(node_1)
                    l_node_1 = lT.get_chain_of_node(node_1)[-1]
                    matched_left.append(l_node_1)
                    matched_right.append(node_2)
                    l_node_2 = lT.get_chain_of_node(node_2)[-1]
                    matched_right.append(l_node_2)
                    colors[node_1] = __calculate_distance_of_sub_tree(
                        lT,
                        node_1,
                        node_2,
                        btrc,
                        corres1,
                        corres2,
                        delta_tmp,
                        lT._norm_dict[norm],
                        tree1.get_norm(node_1),
                        tree2.get_norm(node_2),
                    )
                    colors[l_node_1] = colors[node_1]
                    colors[node_2] = colors[node_1]
                    colors[l_node_2] = colors[node_1]
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)
    cmap = colormaps[colormap]
    if vmin is None:
        vmin = np.percentile(list(colors.values()), 5)
    if vmax is None:
        vmax = np.percentile(list(colors.values()), 95)
    c_norm = mcolors.Normalize(vmin, vmax)
    colors = {c: cmap(c_norm(v)) for c, v in colors.items()}
    lT.plot_subtree(
        lT.get_ancestor_at_t(n1),
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
    lT.plot_subtree(
        lT.get_ancestor_at_t(n2),
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


def labelled_mappings(
    lT: LineageTree,
    n1: int,
    n2: int,
    end_time: int | None = None,
    norm: Literal["max", "sum", None] = "max",
    style: (
        Literal["simple", "normalized_simple", "full", "downsampled", "mini"]
        | type[TreeApproximationTemplate]
    ) = "simple",
    downsample: int = 2,
) -> dict[str, list]:
    """List which nodes are matched when comparing two subtrees.

    The comparison is the one of
    [`unordered_tree_edit_distance`][lineagetree.LineageTree.unordered_tree_edit_distance].
    Nodes are reported by their label (see
    [`labels`][lineagetree.LineageTree.labels]) or, when they have none, by
    their id. With chain-based styles, each chain is reported by its first
    node.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    n1 : int
        The node spawning the first subtree.
    n2 : int
        The node spawning the second subtree.
    end_time : int, optional
        The last time point taken into account. If None, the whole subtrees
        are compared.
    norm : {"max", "sum", None}, default="max"
        The normalization method.
    style : {"simple", "normalized_simple", "full", "downsampled", "mini"} or TreeApproximationTemplate subclass, default="simple"
        The tree approximation used for the comparison; see
        [`tree_style`][lineagetree.tree_approximation.tree_style].
    downsample : int, default=2
        The downsample factor for the downsampled tree approximation.
        Used only when `style="downsampled"`.

    Returns
    -------
    dict
        A dictionary with two keys:

        - ``'matched'``: list of ``(node_of_tree1, node_of_tree2)`` pairs,
        - ``'unmatched'``: list of the nodes of either tree that have no
          match.
    """
    parameters = (
        end_time,
        convert_style_to_number(style=style, downsample=downsample),
    )
    n1, n2 = sorted([n1, n2])
    lT._comparisons.setdefault(parameters, {})
    if lT._comparisons[parameters].get((n1, n2)):
        tmp = lT._comparisons[parameters][(n1, n2)]
    else:
        tmp = __unordereded_backtrace(
            lT, n1, n2, end_time, norm, style, downsample
        )
    if not tmp["trees"]:
        return {"matched": [], "unmatched": []}
    btrc = tmp["alignment"]
    tree1, tree2 = tmp["trees"]

    (
        *_,
        corres1,
    ) = tree1.edist
    (
        *_,
        corres2,
    ) = tree2.edist

    if norm not in lT._norm_dict:
        raise ValueError(
            "Select a viable normalization method (max, sum, None)"
        )
    matched = []
    unmatched = []
    if style not in ("full", "downsampled"):
        for m in btrc:
            if m._left != -1 and m._right != -1:
                cyc1 = lT.get_chain_of_node(corres1[m._left])
                if len(cyc1) > 1:
                    node_1, *_ = cyc1
                elif len(cyc1) == 1:
                    node_1 = cyc1.pop()
                cyc2 = lT.get_chain_of_node(corres2[m._right])
                if len(cyc2) > 1:
                    node_2, *_ = cyc2
                elif len(cyc2) == 1:
                    node_2 = cyc2.pop()
                matched.append(
                    (
                        lT.labels.get(node_1, node_1),
                        lT.labels.get(node_2, node_2),
                    )
                )

            else:
                if m._left != -1:
                    node_1 = lT.get_chain_of_node(corres1.get(m._left, "-"))[0]
                else:
                    node_1 = lT.get_chain_of_node(corres2.get(m._right, "-"))[
                        0
                    ]
                unmatched.append(lT.labels.get(node_1, node_1))
    else:
        for m in btrc:
            if m._left != -1 and m._right != -1:
                node_1 = corres1[m._left]
                node_2 = corres2[m._right]
                matched.append(
                    (
                        lT.labels.get(node_1, node_1),
                        lT.labels.get(node_2, node_2),
                    )
                )
            else:
                if m._left != -1:
                    node_1 = corres1[m._left]
                else:
                    node_1 = corres2[m._right]
                unmatched.append(lT.labels.get(node_1, node_1))
    return {"matched": matched, "unmatched": unmatched}
