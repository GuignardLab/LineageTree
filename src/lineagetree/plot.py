from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Literal, TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from ._core.utils import create_links_and_chains, hierarchical_pos

if TYPE_CHECKING:
    from .lineage_tree import LineageTree


def __plot_nodes(
    hier: dict,
    selected_nodes: set,
    color: str | dict | list,
    size: int | float,
    ax: plt.Axes,
    default_color: str = "black",
    **kwargs,
) -> None:
    """
    Private method that plots the nodes of the tree.
    """

    if isinstance(color, dict):
        color = [color.get(k, default_color) for k in hier]
    elif isinstance(color, str | list):
        color = [
            color if node in selected_nodes else default_color for node in hier
        ]
    hier_pos = np.array(list(hier.values()))
    ax.scatter(*hier_pos.T, s=size, zorder=10, color=color, **kwargs)


def __plot_edges(
    hier: dict,
    lnks_tms: dict,
    selected_edges: Iterable,
    color: str | dict | list,
    lw: float,
    ax: plt.Axes,
    default_color: str = "black",
    **kwargs,
) -> None:
    """
    Private method that plots the edges of the tree.
    """
    if isinstance(color, dict):
        selected_edges = color.keys()
    lines = []
    c = []
    for pred, succs in lnks_tms["links"].items():
        for suc in succs:
            lines.append(
                [
                    [hier[suc][0], hier[suc][1]],
                    [hier[pred][0], hier[pred][1]],
                ]
            )
            if pred in selected_edges:
                if isinstance(color, str | list):
                    c.append(color)
                elif isinstance(color, dict):
                    c.append(color[pred])
            else:
                c.append(default_color)
    lc = LineCollection(lines, colors=c, linewidth=lw, **kwargs)
    ax.add_collection(lc)


def draw_tree_graph(
    lT: LineageTree,
    hier: dict[int, tuple[int, int]],
    lnks_tms: dict[str, dict[int, list | int]],
    selected_nodes: list | set | None = None,
    selected_edges: list | set | None = None,
    color_of_nodes: str | dict = "magenta",
    color_of_edges: str | dict = "magenta",
    size: int | float = 10,
    lw: float = 0.3,
    ax: plt.Axes | None = None,
    default_color: str = "black",
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Function to plot the tree graph.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    hier : dict mapping int to tuple of int
        Dictionary that contains the positions of all nodes.
    lnks_tms : dict mapping string to dictionaries mapping int to list or int
        - 'links' : conatains the hierarchy of the nodes (only start and end of each chain)
        - 'times' : contains the distance between the  start and the end of each chain.
    selected_nodes : list or set, optional
        Which nodes are to be selected (Painted with a different color, according to 'color_'of_nodes')
    selected_edges : list or set, optional
        Which edges are to be selected (Painted with a different color, according to 'color_'of_edges')
    color_of_nodes : str, default="magenta"
        Color of selected nodes
    color_of_edges : str, default="magenta"
        Color of selected edges
    size : int, default=10
        Size of the nodes, defaults to 10
    lw : float, default=0.3
        The width of the edges of the tree graph, defaults to 0.3
    ax : plt.Axes, optional
        Plot the graph on existing ax. If not provided or None a new ax is going to be created.
    default_color : str, default="black"
        Default color of nodes

    Returns
    -------
    plt.Figure
        The matplotlib figure
    plt.Axes
        The matplotlib ax
    """
    if selected_nodes is None:
        selected_nodes = []
    if selected_edges is None:
        selected_edges = []
    if ax is None:
        _, ax = plt.subplots()
    else:
        ax.clear()
    if not isinstance(selected_nodes, set):
        selected_nodes = set(selected_nodes)
    if not isinstance(selected_edges, set):
        selected_edges = set(selected_edges)
    if 0 < size:
        __plot_nodes(
            hier,
            selected_nodes,
            color_of_nodes,
            size=size,
            ax=ax,
            default_color=default_color,
            **kwargs,
        )
    if not color_of_edges:
        color_of_edges = color_of_nodes
    __plot_edges(
        hier,
        lnks_tms,
        selected_edges,
        color_of_edges,
        lw,
        ax,
        default_color=default_color,
        **kwargs,
    )
    ax.autoscale()
    plt.draw()
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    return ax.get_figure(), ax


def _create_dict_of_plots(
    lT: LineageTree,
    node: int | Iterable[int] | None = None,
    start_time: int | None = None,
    end_time: int | None = None,
) -> dict[int, dict]:
    """Generates a dictionary of graphs where the keys are the index of the graph and
    the values are the graphs themselves which are produced by `create_links_and_chains`

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    node : int or Iterable of int, optional
        The id of the node/nodes to produce the simple graphs, if not provided or None will
        calculate the dicts for every root that starts before 'start_time'
    start_time : int, optional
        Important only if there are no nodes it will produce the graph of every
        root that starts before or at start time. If not provided or None the 'start_time' defaults to the start of the dataset.
    end_time : int, optional
        The last timepoint to be considered, if not provided or None the last timepoint of the
        dataset (t_e) is considered.

    Returns
    -------
    dict mapping int to dict
        The keys are just index values 0-n and the values are the graphs produced.
    """
    if start_time is None:
        start_time = lT.t_b
    if end_time is None:
        end_time = lT.t_e
    if node is None:
        mothers = [root for root in lT.roots if lT._time[root] <= start_time]
    elif isinstance(node, Iterable):
        mothers = node
    else:
        mothers = [node]
    return {
        i: create_links_and_chains(lT, mother, end_time=end_time)
        for i, mother in enumerate(mothers)
    }


def plot_all_lineages(
    lT: LineageTree,
    nodes: list | None = None,
    last_time_point_to_consider: int | None = None,
    nrows: int = 2,
    figsize: tuple[int, int] = (10, 15),
    dpi: int = 100,
    fontsize: int = 15,
    axes: plt.Axes | None = None,
    vert_gap: int = 1,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes, dict[plt.Axes, int]]:
    """Plots all lineages.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    nodes : list, optional
        The nodes spawning the graphs to be plotted.
    last_time_point_to_consider : int, optional
        Which timepoints and upwards are the graphs to be plotted.
        For example if start_time is 10, then all trees that begin
        on tp 10 or before are calculated. Defaults to None, where
        it will plot all the roots that exist on `lT.t_b`.
    nrows : int, default=2
        How many rows of plots should be printed.
    figsize : tuple, default=(10, 15)
        The size of the figure.
    dpi : int, default=100
        The dpi of the figure.
    fontsize : int, default=15
        The fontsize of the labels.
    axes : plt.Axes, optional
        The axes to plot the graphs on. If None or not provided new axes are going to be created.
    vert_gap : int, default=1
        space between the nodes, defaults to 1
    **kwargs:
        kwargs accepted by matplotlib.pyplot.plot, matplotlib.pyplot.scatter

    Returns
    -------
    plt.Figure
        The figure
    plt.Axes
        The axes
    dict of plt.Axes to int
        A dictionary that maps the axes to the root of the tree.
    """
    nrows = int(nrows)
    if last_time_point_to_consider is None:
        last_time_point_to_consider = lT.t_b
    if nrows < 1 or not nrows:
        nrows = 1
        raise Warning("Number of rows has to be at least 1")
    if nodes:
        graphs = {
            i: lT._create_dict_of_plots(node) for i, node in enumerate(nodes)
        }
    else:
        graphs = lT._create_dict_of_plots(
            start_time=last_time_point_to_consider
        )
    pos = {
        i: hierarchical_pos(
            g,
            g["root"],
            ycenter=-int(lT._time[g["root"]]),
            vert_gap=vert_gap,
        )
        for i, g in graphs.items()
    }
    if axes is None:
        ncols = int(len(graphs) // nrows) + (+np.sign(len(graphs) % nrows))
        figure, axes = plt.subplots(
            figsize=figsize, nrows=nrows, ncols=ncols, dpi=dpi, sharey=True
        )
    else:
        figure, axes = axes.flatten()[0].get_figure(), axes
        if len(axes.flatten()) < len(graphs):
            raise Exception(
                f"Not enough axes, they should be at least {len(graphs)}."
            )
    flat_axes = axes.flatten()
    ax2root = {}
    min_width, min_height = float("inf"), float("inf")
    for ax in flat_axes:
        bbox = ax.get_window_extent().transformed(
            figure.dpi_scale_trans.inverted()
        )
        min_width = min(min_width, bbox.width)
        min_height = min(min_height, bbox.height)

    adjusted_fontsize = fontsize * min(min_width, min_height) / 5
    for i, graph in graphs.items():
        lT.draw_tree_graph(
            hier=pos[i], lnks_tms=graph, ax=flat_axes[i], **kwargs
        )
        root = graph["root"]
        ax2root[flat_axes[i]] = root
        label = lT.labels.get(root, "Unlabeled")
        xlim = flat_axes[i].get_xlim()
        ylim = flat_axes[i].get_ylim()
        x_pos = (xlim[0] + xlim[1]) / 2
        y_pos = ylim[1] * 0.8
        flat_axes[i].text(
            x_pos,
            y_pos,
            label,
            fontsize=adjusted_fontsize,
            color="black",
            ha="center",
            va="center",
            bbox={
                "facecolor": "white",
                "alpha": 0.5,
                "edgecolor": "green",
            },
        )
    [figure.delaxes(ax) for ax in axes.flatten() if not ax.has_data()]
    return axes.flatten()[0].get_figure(), axes, ax2root


def plot_subtree(
    lT: LineageTree,
    node: int,
    end_time: int | None = None,
    figsize: tuple[int, int] = (4, 7),
    dpi: int = 150,
    vert_gap: int = 2,
    selected_nodes: list | None = None,
    selected_edges: list | None = None,
    color_of_nodes: str | dict = "magenta",
    color_of_edges: str | dict = "magenta",
    size: int | float = 10,
    lw: float = 0.1,
    default_color: str = "black",
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots the subtree spawn by a node.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    node : int
        The id of the node that is going to be plotted.
    end_time : int, optional
        The last timepoint to be considered, if None or not provided the last timepoint of the dataset (t_e) is considered.
    figsize : tuple of 2 ints, default=(4,7)
        The size of the figure, deafults to (4,7)
    vert_gap : int, default=2
        The verical gap of a node when it divides, defaults to 2.
    dpi : int, default=150
        The dpi of the figure, defaults to 150
    selected_nodes : list, optional
        The nodes that are selected by the user to be colored in a different color, defaults to None
    selected_edges : list, optional
        The edges that are selected by the user to be colored in a different color, defaults to None
    color_of_nodes : str, default="magenta"
        The color of the nodes to be colored, except the default colored ones, defaults to "magenta"
    color_of_edges : str, default="magenta"
        The color of the edges to be colored, except the default colored ones, defaults to "magenta"
    size : int, default=10
        The size of the nodes, defaults to 10
    lw : float, default=0.1
        The widthe of the edges of the tree graph, defaults to 0.1
    default_color : str, default="black"
        The default color of nodes and edges, defaults to "black"
    ax : plt.Axes, optional
        The ax where the plot is going to be applied, if not provided or None new axes will be created.

    Returns
    -------
    plt.Figure
        The matplotlib figure
    plt.Axes
        The matplotlib axes

    Raises
    ------
    Warning
        If more than one nodes are received
    """
    graph = lT._create_dict_of_plots(node, end_time=end_time)
    if len(graph) > 1:
        raise Warning(
            "Please use lT.plot_all_lineages(nodes) for plotting multiple nodes."
        )
    graph = graph[0]
    if not ax:
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=dpi)
    lT.draw_tree_graph(
        hier=hierarchical_pos(
            graph,
            graph["root"],
            vert_gap=vert_gap,
            ycenter=-int(lT._time[node]),
        ),
        selected_edges=selected_edges,
        selected_nodes=selected_nodes,
        color_of_edges=color_of_edges,
        color_of_nodes=color_of_nodes,
        default_color=default_color,
        size=size,
        lw=lw,
        lnks_tms=graph,
        ax=ax,
    )
    return ax.get_figure(), ax


def plot_dtw_heatmap(
    lT: LineageTree,
    nodes1: int,
    nodes2: int,
    threshold: int = 1000,
    regist: bool = True,
    start_d: int = 0,
    back_d: int = 0,
    fast: bool = False,
    w: int = 0,
    centered_band: bool = True,
) -> tuple[float, plt.Figure]:
    """
    Plot DTW cost matrix between two chains in heatmap format

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    nodes1 : int
        node to compare distance
    nodes2 : int
        node to compare distance
    threshold : int, default=1000
        set a maximum number of points a chain can have
    regist : bool, default=True
        Rotate and translate trajectories
    start_d : int, default=0
        start delay
    back_d : int, default=0
        end delay
    fast : bool, default=False
        if `True`, the algorithm will use a faster version but might not find the optimal alignment
    w : int, default=0
        window size
    centered_band : bool, default=True
        when running the fast algorithm, `True` if the windown is centered

    Returns
    -------
    float
        DTW distance
    plt.Figure
        Heatmap of cost matrix with opitimal path
    """
    cost, path, cost_mat, pos_chain1, pos_chain2 = lT.calculate_dtw(
        nodes1,
        nodes2,
        threshold,
        regist,
        start_d,
        back_d,
        fast,
        w,
        centered_band,
        cost_mat_p=True,
    )

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(
        cost_mat, cmap="viridis", origin="lower", interpolation="nearest"
    )
    plt.colorbar(im)
    ax.set_title("Heatmap of DTW Cost Matrix")
    ax.set_xlabel("Tree 1")
    ax.set_ylabel("tree 2")
    x_path, y_path = zip(*path, strict=True)
    ax.plot(y_path, x_path, color="black")

    return cost, fig


def __plot_2d(
    pos_chain1: np.ndarray,
    pos_chain2: np.ndarray,
    nodes1: list[int],
    nodes2: list[int],
    ax: plt.Axes,
    x_idx: list[int],
    y_idx: list[int],
    x_label: str,
    y_label: str,
) -> None:
    ax.plot(
        pos_chain1[:, x_idx],
        pos_chain1[:, y_idx],
        "-",
        label=f"root = {nodes1}",
    )
    ax.plot(
        pos_chain2[:, x_idx],
        pos_chain2[:, y_idx],
        "-",
        label=f"root = {nodes2}",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def plot_dtw_trajectory(
    lT: LineageTree,
    nodes1: int,
    nodes2: int,
    threshold: int = 1000,
    regist: bool = True,
    start_d: int = 0,
    back_d: int = 0,
    fast: bool = False,
    w: int = 0,
    centered_band: bool = True,
    projection: Literal["3d", "xy", "xz", "yz", "pca", None] = None,
    alig: bool = False,
) -> tuple[float, plt.Figure]:
    """
    Plots DTW trajectories aligment between two chains in 2D or 3D

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    nodes1 : int
        node to compare distance
    nodes2 : int
        node to compare distance
    threshold : int, default=1000
        set a maximum number of points a chain can have
    regist : bool, default=True
        Rotate and translate trajectories
    start_d : int, default=0
        start delay
    back_d : int, default=0
        end delay
    w : int, default=0
        window size
    fast : bool, default=False
        True if the user wants to run the fast algorithm with window restrains
    centered_band : bool, default=True
        if running the fast algorithm, True if the windown is centered
    projection : {"3d", "xy", "xz", "yz", "pca"}, optional
        specify which 2D to plot ->
        "3d" : for the 3d visualization
        "xy" or None (default) : 2D projection of axis x and y
        "xz" : 2D projection of axis x and z
        "yz" : 2D projection of axis y and z
        "pca" : PCA projection
    alig : bool
        True to show alignment on plot

    Returns
    -------
    float
        DTW distance
    figure
        Trajectories Plot
    """
    (
        distance,
        alignment,
        cost_mat,
        pos_chain1,
        pos_chain2,
    ) = lT.calculate_dtw(
        nodes1,
        nodes2,
        threshold,
        regist,
        start_d,
        back_d,
        fast,
        w,
        centered_band,
        cost_mat_p=True,
    )

    fig = plt.figure(figsize=(10, 6))

    if projection == "3d":
        ax = fig.add_subplot(1, 1, 1, projection="3d")
    else:
        ax = fig.add_subplot(1, 1, 1)

    if projection == "3d":
        ax.plot(
            pos_chain1[:, 0],
            pos_chain1[:, 1],
            pos_chain1[:, 2],
            "-",
            label=f"root = {nodes1}",
        )
        ax.plot(
            pos_chain2[:, 0],
            pos_chain2[:, 1],
            pos_chain2[:, 2],
            "-",
            label=f"root = {nodes2}",
        )
        ax.set_ylabel("y position")
        ax.set_xlabel("x position")
        ax.set_zlabel("z position")
    else:
        if projection == "xy" or projection == "yx" or projection is None:
            __plot_2d(
                pos_chain1,
                pos_chain2,
                nodes1,
                nodes2,
                ax,
                0,
                1,
                "x position",
                "y position",
            )
        elif projection == "xz" or projection == "zx":
            __plot_2d(
                pos_chain1,
                pos_chain2,
                nodes1,
                nodes2,
                ax,
                0,
                2,
                "x position",
                "z position",
            )
        elif projection == "yz" or projection == "zy":
            __plot_2d(
                pos_chain1,
                pos_chain2,
                nodes1,
                nodes2,
                ax,
                1,
                2,
                "y position",
                "z position",
            )
        elif projection == "pca":
            try:
                from sklearn.decomposition import PCA
            except ImportError:
                Warning(
                    "scikit-learn is not installed, the PCA orientation cannot be used."
                    "You can install scikit-learn with pip install"
                )

            # Apply PCA
            pca = PCA(n_components=2)
            pca.fit(np.vstack([pos_chain1, pos_chain2]))
            pos_chain1_2d = pca.transform(pos_chain1)
            pos_chain2_2d = pca.transform(pos_chain2)

            ax.plot(
                pos_chain1_2d[:, 0],
                pos_chain1_2d[:, 1],
                "-",
                label=f"root = {nodes1}",
            )
            ax.plot(
                pos_chain2_2d[:, 0],
                pos_chain2_2d[:, 1],
                "-",
                label=f"root = {nodes2}",
            )

            # Set axis labels
            axes = ["x", "y", "z"]
            x_label = axes[np.argmax(np.abs(pca.components_[0]))]
            y_label = axes[np.argmax(np.abs(pca.components_[1]))]
            x_percent = 100 * (
                np.max(np.abs(pca.components_[0]))
                / np.sum(np.abs(pca.components_[0]))
            )
            y_percent = 100 * (
                np.max(np.abs(pca.components_[1]))
                / np.sum(np.abs(pca.components_[1]))
            )
            ax.set_xlabel(f"{x_percent:.0f}% of {x_label} position")
            ax.set_ylabel(f"{y_percent:.0f}% of {y_label} position")
        else:
            raise ValueError(
                """Error: available projections are:
                    '3d' : for the 3d visualization
                    'xy' or None (default) : 2D projection of axis x and y
                    'xz' : 2D projection of axis x and z
                    'yz' : 2D projection of axis y and z
                    'pca' : PCA projection"""
            )

    connections = [[pos_chain1[i], pos_chain2[j]] for i, j in alignment]

    for connection in connections:
        xyz1 = connection[0]
        xyz2 = connection[1]
        x_pos = [xyz1[0], xyz2[0]]
        y_pos = [xyz1[1], xyz2[1]]
        z_pos = [xyz1[2], xyz2[2]]

        if alig and projection != "pca":
            if projection == "3d":
                ax.plot(x_pos, y_pos, z_pos, "k--", color="grey")
            else:
                ax.plot(x_pos, y_pos, "k--", color="grey")

    ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout()

    if alig and projection == "pca":
        warnings.warn(
            "Error: not possible to show alignment in PCA projection !",
            UserWarning,
            stacklevel=2,
        )

    return distance, fig
