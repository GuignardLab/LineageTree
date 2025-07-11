#!python
# This file is subject to the terms and conditions defined in
# file 'LICENCE', which is part of this source code package.
# Author: Leo Guignard (leo.guignard...@AT@...gmail.com)

from __future__ import annotations

import importlib.metadata
import warnings
from collections.abc import Iterable, Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.spatial import distance

from ._properties import (
    all_chains,
    depth,
    edges,
    labels,
    leaves,
    nodes,
    number_of_nodes,
    parenting,
    predecessor,
    roots,
    successor,
    t_b,
    t_e,
    time,
    time_nodes,
    time_resolution,
)
from .tree_approximation import TreeApproximationTemplate
from .utils import (
    convert_style_to_number,
    create_links_and_chains,
    hierarchical_pos,
)


class LineageTree:
    norm_dict = {"max": max, "sum": sum, None: lambda x: 1}

    # The properties are defined in the `_properties.py` file
    # and assigned to the class here.
    successor = successor
    predecessor = predecessor
    time = time
    t_b = t_b
    t_e = t_e
    nodes = nodes
    number_of_nodes = number_of_nodes
    depth = depth
    roots = roots
    leaves = leaves
    edges = edges
    labels = labels
    time_resolution = time_resolution
    all_chains = all_chains
    time_nodes = time_nodes
    parenting = parenting

    def __check_cc_cycles(self, n: int) -> tuple[bool, set[int]]:
        """Check if the connected component of a given node `n` has a cycle.

        Returns
        -------
        bool
            True if the tree has cycles, False otherwise.
        set of int
            The set of nodes that have been checked.
        """
        to_do = [n]
        no_cycle = True
        already_done = set()
        while to_do and no_cycle:
            current = to_do.pop(-1)
            if current not in already_done:
                already_done.add(current)
            else:
                no_cycle = False
            to_do.extend(self._successor[current])
        to_do = list(self._predecessor[n])
        while to_do and no_cycle:
            current = to_do.pop(-1)
            if current not in already_done:
                already_done.add(current)
            else:
                no_cycle = False
            to_do.extend(self._predecessor[current])
        return not no_cycle, already_done

    def __check_for_cycles(self) -> bool:
        """Check if the tree has cycles.

        Returns
        -------
        bool
            True if the tree has cycles, False otherwise.
        """
        to_do = set(self.nodes)
        found_cycle = False
        while to_do and not found_cycle:
            current = to_do.pop()
            found_cycle, done = self.__check_cc_cycles(current)
            to_do.difference_update(done)
        return found_cycle

    def __eq__(self, other) -> bool:
        if isinstance(other, LineageTree):
            return (
                other._successor == self._successor
                and other._predecessor == self._predecessor
                and other._time == self._time
            )
        else:
            return False

    def __setstate__(self, state):
        if "_successor" not in state:
            state["_successor"] = state["successor"]
        if "_predecessor" not in state:
            state["_predecessor"] = state["predecessor"]
        if "_time" not in state:
            state["_time"] = state["time"]
        self.__dict__.update(state)

    def get_predecessors(
        self,
        x: int,
        depth: int | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[int]:
        """Computes the predecessors of the node `x` up to
        `depth` predecessors or the begining of the life of `x`.
        The ordered list of ids is returned.

        Parameters
        ----------
        x : int
            id of the node to compute
        depth : int
            maximum number of predecessors to return

        Returns
        -------
        list of int
            list of ids, the last id is `x`
        """
        if not start_time:
            start_time = self.t_b
        if not end_time:
            end_time = self.t_e
        unconstrained_chain = [x]
        chain = [x] if start_time <= self._time[x] <= end_time else []
        acc = 0
        while (
            acc != depth
            and start_time < self._time[unconstrained_chain[0]]
            and (
                self._predecessor[unconstrained_chain[0]] != ()
                and (  # Please dont change very important even if it looks weird.
                    len(
                        self._successor[
                            self._predecessor[unconstrained_chain[0]][0]
                        ]
                    )
                    == 1
                )
            )
        ):
            unconstrained_chain.insert(
                0, self._predecessor[unconstrained_chain[0]][0]
            )
            acc += 1
            if start_time <= self._time[unconstrained_chain[0]] <= end_time:
                chain.insert(0, unconstrained_chain[0])

        return chain

    def get_successors(
        self, x: int, depth: int | None = None, end_time: int | None = None
    ) -> list[int]:
        """Computes the successors of the node `x` up to
        `depth` successors or the end of the life of `x`.
        The ordered list of ids is returned.

        Parameters
        ----------
        x : int
            id of the node to compute
        depth : int, optional
            maximum number of predecessors to return
        end_time : int, optional
            maximum time to consider

        Returns
        -------
        list of int
            list of ids, the first id is `x`
        """
        if end_time is None:
            end_time = self.t_e
        chain = [x]
        acc = 0
        while (
            len(self._successor[chain[-1]]) == 1
            and acc != depth
            and self._time[chain[-1]] < end_time
        ):
            chain += self._successor[chain[-1]]
            acc += 1

        return chain

    def get_chain_of_node(
        self,
        x: int,
        depth: int | None = None,
        depth_pred: int | None = None,
        depth_succ: int | None = None,
        end_time: int | None = None,
    ) -> list[int]:
        """Computes the predecessors and successors of the node `x` up to
        `depth_pred` predecessors plus `depth_succ` successors.
        If the value `depth` is provided and not None,
        `depth_pred` and `depth_succ` are overwriten by `depth`.
        The ordered list of ids is returned.
        If all `depth` are None, the full chain is returned.

        Parameters
        ----------
        x : int
            id of the node to compute
        depth : int, optional
            maximum number of predecessors and successor to return
        depth_pred : int, optional
            maximum number of predecessors to return
        depth_succ : int, optional
            maximum number of successors to return

        Returns
        -------
        list of int
            list of node ids
        """
        if end_time is None:
            end_time = self.t_e
        if depth is not None:
            depth_pred = depth_succ = depth
        return self.get_predecessors(x, depth_pred, end_time=end_time)[
            :-1
        ] + self.get_successors(x, depth_succ, end_time=end_time)

    def get_all_chains_of_subtree(
        self, node: int, end_time: int | None = None
    ) -> list[list[int]]:
        """Computes all the chains of the subtree spawn by a given node.
        Similar to get_all_chains().

        Parameters
        ----------
        node : int
            The node from which we want to get its chains.
        end_time : int, optional
            The time at which we want to stop the chains.

        Returns
        -------
        list of list of int
            list of chains
        """
        if not end_time:
            end_time = self.t_e
        chains = [self.get_successors(node)]
        to_do = list(self._successor[chains[0][-1]])
        while to_do:
            current = to_do.pop()
            chain = self.get_successors(current, end_time=end_time)
            if self._time[chain[-1]] <= end_time:
                chains += [chain]
                to_do += self._successor[chain[-1]]
        return chains

    def _compute_all_chains(self) -> list[list[int]]:
        """Computes all the chains of a given lineage tree,
        stores it in `self.all_chains` and returns it.

        Returns
        -------
        list of list of int
            list of chains
        """
        all_chains = []
        to_do = sorted(self.roots, key=self.time.get, reverse=True)
        while len(to_do) != 0:
            current = to_do.pop()
            chain = self.get_chain_of_node(current)
            all_chains += [chain]
            to_do.extend(self._successor[chain[-1]])
        return all_chains

    def __get_chains(  # TODO: Probably should be removed, might be used by DTW. Might also be a @dynamic_property
        self, nodes: Iterable | int | None = None
    ) -> dict[int, list[list[int]]]:
        """Returns all the chains in the subtrees spawned by each of the given nodes.

        Parameters
        ----------
        nodes : Iterable or int, optional
            id or Iterable of ids of the nodes to be computed, if `None` all roots are used

        Returns
        -------
        dict mapping int to list of Chain
            dictionary mapping the node ids to a list of chains
        """
        all_chains = self.all_chains
        if nodes is None:
            nodes = self.roots
        if not isinstance(nodes, Iterable):
            nodes = [nodes]
        output_chains = {}
        for n in nodes:
            starting_node = self.get_predecessors(n)[0]
            found = False
            done = False
            starting_time = self.time[n]
            i = 0
            current_chain = []
            while not done and i < len(all_chains):
                curr_found = all_chains[i][0] == starting_node
                found = found or curr_found
                if found:
                    done = (
                        self.time[all_chains[i][0]] <= starting_time
                    ) and not curr_found
                    if not done:
                        if curr_found:
                            current_chain.append(self.get_successors(n))
                        else:
                            current_chain.append(all_chains[i])
                i += 1
            output_chains[n] = current_chain
        return output_chains

    def find_leaves(self, roots: int | Iterable) -> set[int]:
        """Finds the leaves of a tree spawned by one or more nodes.

        Parameters
        ----------
        roots : int or Iterable
            The roots of the trees spawning the leaves

        Returns
        -------
        set
            The leaves of one or more trees.
        """
        if not isinstance(roots, Iterable):
            to_do = [roots]
        elif isinstance(roots, Iterable):
            to_do = list(roots)
        leaves = set()
        while to_do:
            curr = to_do.pop()
            succ = self._successor[curr]
            if not succ:
                leaves.add(curr)
            to_do += succ
        return leaves

    def get_subtree_nodes(
        self,
        x: int | Iterable,
        end_time: int | None = None,
        preorder: bool = False,
    ) -> list[int]:
        """Computes the list of nodes from the subtree spawned by *x*
        The default output order is Breadth First Traversal.
        Unless preorder is `True` in that case the order is
        Depth First Traversal (DFT) preordered.

        Parameters
        ----------
        x : int
            id of root node
        preorder : bool, default=False
            if True the output preorder is DFT

        Returns
        -------
        list of int
            the ordered list of node ids
        """
        if not end_time:
            end_time = self.t_e
        if not isinstance(x, Iterable):
            to_do = [x]
        elif isinstance(x, Iterable):
            to_do = list(x)
        subtree = []
        while to_do:
            curr = to_do.pop()
            succ = self._successor[curr]
            if succ and end_time < self._time.get(curr, end_time):
                succ = []
                continue
            if preorder:
                to_do = succ + to_do
            else:
                to_do += succ
                subtree += [curr]
        return subtree

    def get_ancestor_at_t(self, n: int, time: int | None = None) -> int:
        """Find the id of the ancestor of a give node `n`
        at a given time `time`.

        If there is no ancestor, returns `None`
        If time is None return the root of the subtree that spawns
        the node n.

        Parameters
        ----------
        n : int
            node for which to look the ancestor
        time : int, optional
            time at which the ancestor has to be found.
            If `None` the ancestor at the first time point
            will be found.

        Returns
        -------
        int
            the id of the ancestor at time `time`,
            `-1` if there is no ancestor.
        """
        if n not in self.nodes:
            return -1
        if time is None:
            time = self.t_b
        ancestor = n
        while (
            time < self._time.get(ancestor, self.t_b - 1)
            and self._predecessor[ancestor]
        ):
            ancestor = self._predecessor[ancestor][0]
        if self._time.get(ancestor, self.t_b - 1) == time:
            return ancestor
        else:
            return -1

    def get_labelled_ancestor(self, node: int) -> int:
        """Finds the first labelled ancestor and returns its ID otherwise returns -1

        Parameters
        ----------
        node : int
            The id of the node

        Returns
        -------
        int
            Returns the first ancestor found that has a label otherwise `-1`.
        """
        if node not in self.nodes:
            return -1
        ancestor = node
        while (
            self.t_b <= self._time.get(ancestor, self.t_b - 1)
            and ancestor != -1
        ):
            if ancestor in self.labels:
                return ancestor
            ancestor = self._predecessor.get(ancestor, [-1])[0]
        return -1

    def get_ancestor_with_attribute(self, node: int, attribute: str) -> int:
        """General purpose function to help with searching the first ancestor that has an attribute.
        Similar to get_labeled_ancestor and may make it redundant.

        Parameters
        ----------
        node : int
            The id of the node

        Returns
        -------
        int
            Returns the first ancestor found that has an attribute otherwise `-1`.
        """
        attr_dict = self.__getattribute__(attribute)
        if not isinstance(attr_dict, dict):
            raise ValueError("Please select a dict attribute")
        if node not in self.nodes:
            return -1
        if node in attr_dict:
            return node
        if node in self.roots:
            return -1
        ancestor = (node,)
        while ancestor and ancestor != [-1]:
            ancestor = ancestor[0]
            if ancestor in attr_dict:
                return ancestor
            ancestor = self._predecessor.get(ancestor, [-1])
        return -1

    def labelled_mappings(
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
    ) -> dict[str, list[str]]:
        """
        Returns the labels or IDs of all the nodes in the subtrees compared.


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
        style : {"simple", "full", "downsampled", "normalized_simple} or TreeApproximationTemplate subclass, default="simple"
            Which tree approximation is going to be used for the comparisons, defaults to 'simple'.
        downsample : int, default=2
            The downsample factor for the downsampled tree approximation.
            Used only when `style="downsampled"`.

        Returns
        -------
        dict mapping str to list of str
            - 'matched' The labels of the matched nodes of the alignment.
            - 'unmatched' The labels of the unmatched nodes of the alginment.
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
                    cyc1 = self.get_chain_of_node(corres1[m._left])
                    if len(cyc1) > 1:
                        node_1, *_ = cyc1
                    elif len(cyc1) == 1:
                        node_1 = cyc1.pop()
                    cyc2 = self.get_chain_of_node(corres2[m._right])
                    if len(cyc2) > 1:
                        node_2, *_ = cyc2
                    elif len(cyc2) == 1:
                        node_2 = cyc2.pop()
                    matched.append(
                        (
                            self.labels.get(node_1, node_1),
                            self.labels.get(node_2, node_2),
                        )
                    )

                else:
                    if m._left != -1:
                        node_1 = self.get_chain_of_node(
                            corres1.get(m._left, "-")
                        )[0]
                    else:
                        node_1 = self.get_chain_of_node(
                            corres2.get(m._right, "-")
                        )[0]
                    unmatched.append(self.labels.get(node_1, node_1))
        else:
            for m in btrc:
                if m._left != -1 and m._right != -1:
                    node_1 = corres1[m._left]
                    node_2 = corres2[m._right]
                    matched.append(
                        (
                            self.labels.get(node_1, node_1),
                            self.labels.get(node_2, node_2),
                        )
                    )
                else:
                    if m._left != -1:
                        node_1 = corres1[m._left]
                    else:
                        node_1 = corres2[m._right]
                    unmatched.append(self.labels.get(node_1, node_1))
        return {"matched": matched, "unmatched": unmatched}

    @staticmethod
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
                color if node in selected_nodes else default_color
                for node in hier
            ]
        hier_pos = np.array(list(hier.values()))
        ax.scatter(*hier_pos.T, s=size, zorder=10, color=color, **kwargs)

    @staticmethod
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
        self,
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
            self.__plot_nodes(
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
        self.__plot_edges(
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
        self,
        node: int | Iterable[int] | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> dict[int, dict]:
        """Generates a dictionary of graphs where the keys are the index of the graph and
        the values are the graphs themselves which are produced by `create_links_and_chains`

        Parameters
        ----------
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
            start_time = self.t_b
        if end_time is None:
            end_time = self.t_e
        if node is None:
            mothers = [
                root for root in self.roots if self._time[root] <= start_time
            ]
        elif isinstance(node, Iterable):
            mothers = node
        else:
            mothers = [node]
        return {
            i: create_links_and_chains(self, mother, end_time=end_time)
            for i, mother in enumerate(mothers)
        }

    def plot_all_lineages(
        self,
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
        nodes : list, optional
            The nodes spawning the graphs to be plotted.
        last_time_point_to_consider : int, optional
            Which timepoints and upwards are the graphs to be plotted.
            For example if start_time is 10, then all trees that begin
            on tp 10 or before are calculated. Defaults to None, where
            it will plot all the roots that exist on `self.t_b`.
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
            last_time_point_to_consider = self.t_b
        if nrows < 1 or not nrows:
            nrows = 1
            raise Warning("Number of rows has to be at least 1")
        if nodes:
            graphs = {
                i: self._create_dict_of_plots(node)
                for i, node in enumerate(nodes)
            }
        else:
            graphs = self._create_dict_of_plots(
                start_time=last_time_point_to_consider
            )
        pos = {
            i: hierarchical_pos(
                g,
                g["root"],
                ycenter=-int(self._time[g["root"]]),
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
            self.draw_tree_graph(
                hier=pos[i], lnks_tms=graph, ax=flat_axes[i], **kwargs
            )
            root = graph["root"]
            ax2root[flat_axes[i]] = root
            label = self.labels.get(root, "Unlabeled")
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
        self,
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
        graph = self._create_dict_of_plots(node, end_time=end_time)
        if len(graph) > 1:
            raise Warning(
                "Please use lT.plot_all_lineages(nodes) for plotting multiple nodes."
            )
        graph = graph[0]
        if not ax:
            _, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=dpi)
        self.draw_tree_graph(
            hier=hierarchical_pos(
                graph,
                graph["root"],
                vert_gap=vert_gap,
                ycenter=-int(self._time[node]),
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

    def nodes_at_t(
        self,
        t: int,
        r: int | Iterable[int] | None = None,
    ) -> list[int]:
        """
        Returns the list of nodes at time `t` that are spawn by the node(s) `r`.

        Parameters
        ----------
        t : int
            target time, if `None` goes as far as possible
        r : int or Iterable of int, optional
            id or list of ids of the spawning node

        Returns
        -------
        list of int
            list of ids of the nodes at time `t` spawned by `r`
        """
        if not r and r != 0:
            r = {root for root in self.roots if self.time[root] <= t}
        if isinstance(r, int):
            r = [r]
        if t is None:
            t = self.t_e
        to_do = list(r)
        final_nodes = []
        while len(to_do) > 0:
            curr = to_do.pop()
            for _next in self._successor[curr]:
                if self._time[_next] < t:
                    to_do.append(_next)
                elif self._time[_next] == t:
                    final_nodes.append(_next)
        if not final_nodes:
            return list(r)
        return final_nodes

    @staticmethod
    def __calculate_diag_line(dist_mat: np.ndarray) -> tuple[float, float]:
        """
        Calculate the line that centers the band w.

        Parameters
        ----------
        dist_mat : np.ndarray
            distance matrix obtained by the function calculate_dtw

        Returns
        -------
        float
            The slope of the curve
        float
            The intercept of the curve
        """
        i, j = dist_mat.shape
        x1 = max(0, i - j) / 2
        x2 = (i + min(i, j)) / 2
        y1 = max(0, j - i) / 2
        y2 = (j + min(i, j)) / 2
        slope = (y1 - y2) / (x1 - x2)
        intercept = y1 - slope * x1
        return slope, intercept

    # Reference: https://github.com/kamperh/lecture_dtw_notebook/blob/main/dtw.ipynb
    def __dp(
        self,
        dist_mat: np.ndarray,
        start_d: int = 0,
        back_d: int = 0,
        fast: bool = False,
        w: int = 0,
        centered_band: bool = True,
    ) -> tuple[list[int], np.ndarray, float]:
        """
        Find DTW minimum cost between two series using dynamic programming.

        Parameters
        ----------
        dist_mat : np.ndarray
            distance matrix obtained by the function calculate_dtw
        start_d : int, default=0
            start delay
        back_d : int, default=0
            end delay
        fast : bool, default=False
            if `True`, the algorithm will use a faster version but might not find the optimal alignment
        w : int, default=0
            window constrain
        centered_band : bool, default=True
            if `True`, the band will be centered around the diagonal

        Returns
        -------
        tuple of tuples of int
            Aligment path
        np.ndarray
            cost matrix
        float
            optimal cost
        """
        N, M = dist_mat.shape
        w_limit = max(w, abs(N - M))  # Calculate the Sakoe-Chiba band width

        if centered_band:
            slope, intercept = self.__calculate_diag_line(dist_mat)
            square_root = np.sqrt((slope**2) + 1)

        # Initialize the cost matrix
        cost_mat = np.full((N + 1, M + 1), np.inf)
        cost_mat[0, 0] = 0

        # Fill the cost matrix while keeping traceback information
        traceback_mat = np.zeros((N, M))

        cost_mat[: start_d + 1, 0] = 0
        cost_mat[0, : start_d + 1] = 0

        cost_mat[N - back_d :, M] = 0
        cost_mat[N, M - back_d :] = 0

        for i in range(N):
            for j in range(M):
                if fast and not centered_band:
                    condition = abs(i - j) <= w_limit
                elif fast:
                    condition = (
                        abs(slope * i - j + intercept) / square_root <= w_limit
                    )
                else:
                    condition = True

                if condition:
                    penalty = [
                        cost_mat[i, j],  # match (0)
                        cost_mat[i, j + 1],  # insertion (1)
                        cost_mat[i + 1, j],  # deletion (2)
                    ]
                    i_penalty = np.argmin(penalty)
                    cost_mat[i + 1, j + 1] = (
                        dist_mat[i, j] + penalty[i_penalty]
                    )
                    traceback_mat[i, j] = i_penalty

        min_index1 = np.argmin(cost_mat[N - back_d :, M])
        min_index2 = np.argmin(cost_mat[N, M - back_d :])

        if (
            cost_mat[N, M - back_d + min_index2]
            < cost_mat[N - back_d + min_index1, M]
        ):
            i = N - 1
            j = M - back_d + min_index2 - 1
            final_cost = cost_mat[i + 1, j + 1]
        else:
            i = N - back_d + min_index1 - 1
            j = M - 1
            final_cost = cost_mat[i + 1, j + 1]

        path = [(i, j)]

        while (
            start_d != 0
            and ((start_d < i and j > 0) or (i > 0 and start_d < j))
        ) or (start_d == 0 and (i > 0 or j > 0)):
            tb_type = traceback_mat[i, j]
            if tb_type == 0:
                # Match
                i -= 1
                j -= 1
            elif tb_type == 1:
                # Insertion
                i -= 1
            elif tb_type == 2:
                # Deletion
                j -= 1

            path.append((i, j))

        # Strip infinity edges from cost_mat before returning
        cost_mat = cost_mat[1:, 1:]
        return path[::-1], cost_mat, final_cost

    # Reference: https://github.com/nghiaho12/rigid_transform_3D
    @staticmethod
    def __rigid_transform_3D(A, B):
        assert A.shape == B.shape

        num_rows, num_cols = A.shape
        if num_rows != 3:
            raise Exception(
                f"matrix A is not 3xN, it is {num_rows}x{num_cols}"
            )

        num_rows, num_cols = B.shape
        if num_rows != 3:
            raise Exception(
                f"matrix B is not 3xN, it is {num_rows}x{num_cols}"
            )

        # find mean column wise
        centroid_A = np.mean(A, axis=1)
        centroid_B = np.mean(B, axis=1)

        # ensure centroids are 3x1
        centroid_A = centroid_A.reshape(-1, 1)
        centroid_B = centroid_B.reshape(-1, 1)

        # subtract mean
        Am = A - centroid_A
        Bm = B - centroid_B

        H = Am @ np.transpose(Bm)

        # find rotation
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # special reflection case
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T

        t = -R @ centroid_A + centroid_B

        return R, t

    def __interpolate(
        self, chain1: list, chain2: list, threshold: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Interpolate two series that have different lengths

        Parameters
        ----------
        chain1 : list of int
            list of nodes of the first chain to compare
        chain2 : list of int
            list of nodes of the second chain to compare
        threshold : int
            set a maximum number of points a chain can have

        Returns
        -------
        list of np.ndarray
            `x`, `y`, `z` postions for `chain1`
        list of np.ndarray
            `x`, `y`, `z` postions for `chain2`
        """
        inter1_pos = []
        inter2_pos = []

        chain1_pos = np.array([self.pos[c_id] for c_id in chain1])
        chain2_pos = np.array([self.pos[c_id] for c_id in chain2])

        # Both chains have the same length and size below the threshold - nothing is done
        if len(chain1) == len(chain2) and (
            len(chain1) <= threshold or len(chain2) <= threshold
        ):
            return chain1_pos, chain2_pos
        # Both chains have the same length but one or more sizes are above the threshold
        elif len(chain1) > threshold or len(chain2) > threshold:
            sampling = threshold
        # chains have different lengths and the sizes are below the threshold
        else:
            sampling = max(len(chain1), len(chain2))

        for pos in range(3):
            chain1_interp = InterpolatedUnivariateSpline(
                np.linspace(0, 1, len(chain1_pos[:, pos])),
                chain1_pos[:, pos],
                k=1,
            )
            inter1_pos.append(chain1_interp(np.linspace(0, 1, sampling)))

            chain2_interp = InterpolatedUnivariateSpline(
                np.linspace(0, 1, len(chain2_pos[:, pos])),
                chain2_pos[:, pos],
                k=1,
            )
            inter2_pos.append(chain2_interp(np.linspace(0, 1, sampling)))

        return np.column_stack(inter1_pos), np.column_stack(inter2_pos)

    def calculate_dtw(
        self,
        nodes1: int,
        nodes2: int,
        threshold: int = 1000,
        regist: bool = True,
        start_d: int = 0,
        back_d: int = 0,
        fast: bool = False,
        w: int = 0,
        centered_band: bool = True,
        cost_mat_p: bool = False,
    ) -> (
        tuple[float, tuple, np.ndarray, np.ndarray, np.ndarray]
        | tuple[float, tuple]
    ):
        """
        Calculate DTW distance between two chains

        Parameters
        ----------
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
        cost_mat_p : bool, default=False
            True if print the not normalized cost matrix

        Returns
        -------
        float
            DTW distance
        tuple of tuples
            Aligment path
        matrix
            Cost matrix
        list of lists
            rotated and translated trajectories positions
        list of lists
            rotated and translated trajectories positions
        """
        nodes1_chain = self.get_chain_of_node(nodes1)
        nodes2_chain = self.get_chain_of_node(nodes2)

        interp_chain1, interp_chain2 = self.__interpolate(
            nodes1_chain, nodes2_chain, threshold
        )

        pos_chain1 = np.array([self.pos[c_id] for c_id in nodes1_chain])
        pos_chain2 = np.array([self.pos[c_id] for c_id in nodes2_chain])

        if regist:
            R, t = self.__rigid_transform_3D(
                np.transpose(interp_chain1), np.transpose(interp_chain2)
            )
            pos_chain1 = np.transpose(np.dot(R, pos_chain1.T) + t)

        dist_mat = distance.cdist(pos_chain1, pos_chain2, "euclidean")

        path, cost_mat, final_cost = self.__dp(
            dist_mat,
            start_d,
            back_d,
            w=w,
            fast=fast,
            centered_band=centered_band,
        )
        cost = final_cost / len(path)

        if cost_mat_p:
            return cost, path, cost_mat, pos_chain1, pos_chain2
        else:
            return cost, path

    def plot_dtw_heatmap(
        self,
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
        cost, path, cost_mat, pos_chain1, pos_chain2 = self.calculate_dtw(
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

    @staticmethod
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
        self,
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
        ) = self.calculate_dtw(
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
                self.__plot_2d(
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
                self.__plot_2d(
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
                self.__plot_2d(
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

    def get_subtree(self, node_list: set[int]) -> LineageTree:
        new_successors = {
            n: tuple(vi for vi in self.successor[n] if vi in node_list)
            for n in node_list
        }
        return LineageTree(
            successor=new_successors,
            time=self._time,
            pos=self.pos,
            name=self.name,
            root_leaf_value=[
                (),
            ],
            **{
                name: self.__dict__[name]
                for name in self._custom_property_list
            },
        )

    def __init__(
        self,
        *,
        successor: dict[int, Sequence] | None = None,
        predecessor: dict[int, int | Sequence] | None = None,
        time: dict[int, int] | None = None,
        starting_time: int | None = None,
        pos: dict[int, Iterable] | None = None,
        name: str | None = None,
        root_leaf_value: Sequence | None = None,
        **kwargs,
    ):
        """Create a LineageTree object from minimal information, without reading from a file.
        Either `successor` or `predecessor` should be specified.

        Parameters
        ----------
        successor : dict mapping int to Iterable
            Dictionary assigning nodes to their successors.
        predecessor : dict mapping int to int or Iterable
            Dictionary assigning nodes to their predecessors.
        time : dict mapping int to int, optional
            Dictionary assigning nodes to the time point they were recorded to.
            Defaults to None, in which case all times are set to `starting_time`.
        starting_time : int, optional
            Starting time of the lineage tree. Defaults to 0.
        pos : dict mapping int to Iterable, optional
            Dictionary assigning nodes to their positions. Defaults to None.
        name : str, optional
            Name of the lineage tree. Defaults to None.
        root_leaf_value : Iterable, optional
            Iterable of values of roots' predecessors and leaves' successors in the successor and predecessor dictionaries.
            Defaults are `[None, (), [], set()]`.
        **kwargs:
            Supported keyword arguments are dictionaries assigning nodes to any custom property.
            The property must be specified for every node, and named differently from LineageTree's own attributes.
        """
        self.__version__ = importlib.metadata.version("lineagetree")
        self.name = str(name) if name is not None else None
        if successor is not None and predecessor is not None:
            raise ValueError(
                "You cannot have both successors and predecessors."
            )

        if root_leaf_value is None:
            root_leaf_value = [None, (), [], set()]
        elif not isinstance(root_leaf_value, Iterable):
            raise TypeError(
                f"root_leaf_value is of type {type(root_leaf_value)}, expected Iterable."
            )
        elif len(root_leaf_value) < 1:
            raise ValueError(
                "root_leaf_value should have at least one element."
            )
        self._successor = {}
        self._predecessor = {}
        if successor is not None:
            for pred, succs in successor.items():
                if succs in root_leaf_value:
                    self._successor[pred] = ()
                else:
                    if not isinstance(succs, Iterable):
                        raise TypeError(
                            f"Successors should be Iterable, got {type(succs)}."
                        )
                    if len(succs) == 0:
                        raise ValueError(
                            f"{succs} was not declared as a leaf but was found as a successor.\n"
                            "Please lift the ambiguity."
                        )
                    self._successor[pred] = tuple(succs)
                    for succ in succs:
                        if succ in self._predecessor:
                            raise ValueError(
                                "Node can have at most one predecessor."
                            )
                        self._predecessor[succ] = (pred,)
        elif predecessor is not None:
            for succ, pred in predecessor.items():
                if pred in root_leaf_value:
                    self._predecessor[succ] = ()
                else:
                    if isinstance(pred, Sequence):
                        if len(pred) == 0:
                            raise ValueError(
                                f"{pred} was not declared as a leaf but was found as a successor.\n"
                                "Please lift the ambiguity."
                            )
                        if 1 < len(pred):
                            raise ValueError(
                                "Node can have at most one predecessor."
                            )
                        pred = pred[0]
                    self._predecessor[succ] = (pred,)
                    self._successor.setdefault(pred, ())
                    self._successor[pred] += (succ,)
        for root in set(self._successor).difference(self._predecessor):
            self._predecessor[root] = ()
        for leaf in set(self._predecessor).difference(self._successor):
            self._successor[leaf] = ()

        if self.__check_for_cycles():
            raise ValueError(
                "Cycles were found in the tree, there should not be any."
            )

        if pos is None or len(pos) == 0:
            self.pos = {}
        else:
            if self.nodes.difference(pos) != set():
                raise ValueError("Please provide the position of all nodes.")
            self.pos = {
                node: np.array(position) for node, position in pos.items()
            }
        if "labels" in kwargs:
            self._labels = kwargs["labels"]
            kwargs.pop("labels")
        if time is None:
            if starting_time is None:
                starting_time = 0
            if not isinstance(starting_time, int):
                warnings.warn(
                    f"Attribute `starting_time` was a `{type(starting_time)}`, has been casted as an `int`.",
                    stacklevel=2,
                )
            self._time = dict.fromkeys(self.roots, starting_time)
            queue = list(self.roots)
            for node in queue:
                for succ in self._successor[node]:
                    self._time[succ] = self._time[node] + 1
                    queue.append(succ)
        else:
            if starting_time is not None:
                warnings.warn(
                    "Both `time` and `starting_time` were provided, `starting_time` was ignored.",
                    stacklevel=2,
                )
            self._time = {n: int(time[n]) for n in self.nodes}
            if self._time != time:
                if len(self._time) != len(time):
                    warnings.warn(
                        "The provided `time` dictionary had keys that were not nodes. "
                        "They have been removed",
                        stacklevel=2,
                    )
                else:
                    warnings.warn(
                        "The provided `time` dictionary had values that were not `int`. "
                        "These values have been truncated and converted to `int`",
                        stacklevel=2,
                    )
            if self.nodes.symmetric_difference(self._time) != set():
                raise ValueError(
                    "Please provide the time of all nodes and only existing nodes."
                )
            if not all(
                self._time[node] < self._time[s]
                for node, succ in self._successor.items()
                for s in succ
            ):
                raise ValueError(
                    "Provided times are not strictly increasing. Setting times to default."
                )
        # custom properties
        self._custom_property_list = []
        for name, d in kwargs.items():
            if name in self.__dict__:
                warnings.warn(
                    f"Attribute name {name} is reserved.", stacklevel=2
                )
                continue
            setattr(self, name, d)
            self._custom_property_list.append(name)
        if not hasattr(self, "_comparisons"):
            self._comparisons = {}
