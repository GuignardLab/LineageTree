#!python
# This file is subject to the terms and conditions defined in
# file 'LICENCE', which is part of this source code package.
# Author: Leo Guignard (leo.guignard...@AT@...gmail.com)

import importlib.metadata
import os
import pickle as pkl
import struct
import warnings
from collections.abc import Callable, Iterable
from functools import partial
from itertools import combinations
from numbers import Number
from pathlib import Path
from types import MappingProxyType
from typing import TextIO, Union, Literal

import svgwrite
from packaging.version import Version

from .tree_styles import tree_style

try:
    from edist import uted
except ImportError:
    warnings.warn(
        "No edist installed therefore you will not be able to compute the tree edit distance.",
        stacklevel=2,
    )
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.spatial import Delaunay, distance
from scipy.spatial import cKDTree as KDTree

from .utils import (
    create_links_and_cycles,
    hierarchical_pos,
)


class lineageTree:
    def __eq__(self, other) -> bool:
        if isinstance(other, lineageTree):
            return other.successor == self._successor
        return False

    def get_next_id(self) -> int:
        """Computes the next authorized id.

        Returns:
            int: next authorized id
        """
        if self.max_id == -1 and self.nodes:
            self.max_id = max(self.nodes)
        if self.next_id == []:
            self.max_id += 1
            return self.max_id
        else:
            return self.next_id.pop()

    def complete_lineage(self, nodes: int | set = None) -> None:
        """Makes all leaf branches longer so that they reach the last timepoint (self.t_e), useful
        for tree edit distance algorithms.

        Args:
            nodes int | set (optional): Which trees should be "completed", if None it will complete the whole dataset. Defaults to None.
        """
        if nodes is None:
            nodes = set(self.roots)
        elif isinstance(nodes, int):
            nodes = {nodes}
        specfific_nodes = self.find_leaves(nodes)
        for leaf in specfific_nodes:
            self.add_branch(
                leaf,
                (self.t_e - self.time[leaf]),
                downstream=True,
            )

    ###TODO pos can be callable and stay motionless (copy the position of the succ node, use something like optical flow)
    def add_branch(
        self,
        node: int,
        length: int,
        downstream: bool,
        pos: Union[callable, None] = None,
    ) -> int:
        """Adds branches either downstream or upstream to the node selected. Does not add nodes
        Args:
            pred (int): Id of the successor (predecessor if reverse is False)
            length (int): The length of the new branch.
            pos (np.ndarray, optional): The new position of the branch. Defaults to None.
            move_timepoints (bool): Moves the time, important only if reverse= True
            reverese (bool): If True will create a branch that goes forwards in time otherwise backwards.
        Returns:
            (int): Id of the last node added.
        """
        if length == 0:
            return node
        if length < 1:
            raise ValueError("Length cannot be <2")
        if downstream:
            for _ in range(length):
                old_node = node
                node = self._add_node(pred=[old_node])
                self._time[node] = self.time[old_node] + 1
                # if not pos:
                #     self.pos[node] = self.pos[old_node]
                # else:
                #     self.pos[node] = pos
        else:
            if self._predecessor[node]:
                raise Warning("The node already has a predecessor.")
            if self.time[node] - length < self.t_b:
                raise Warning(
                    "A node cannot created outside the lower bound of the dataset. (It is possible to change it by lT.t_b = int(...))"
                )
            for _ in range(length):
                old_node = node
                node = self._add_node(succ=[old_node])
                self._time[node] = self.time[old_node] - 1
                # if not pos:
                #     self.pos[node] = self.pos[old_node]
                # else:
                #     self.pos[node] = pos
        self._changed_leaves = True
        return node

    def cut_tree(self, root: int) -> int:
        """It transforms a lineage that has at least 2 divisions into 2 independent lineages,
        that spawn from the time point of the first node. (splits a tree into 2)

        Args:
            root (int): The id of the node, which will be cut.

        Returns:
            int: The id of the new tree
        """
        cycle = self.get_successors(root)
        last_cell = cycle[-1]
        if last_cell in self._successor:
            new_lT = self._successor[last_cell].pop()
            self._predecessor.pop(new_lT)
            label_of_root = self.labels.get(cycle[0], cycle[0])
            self.labels[cycle[0]] = f"L-Split {label_of_root}"
            new_tr = self.add_branch(new_lT, len(cycle), downstream=False)
            self.roots.add(new_tr)
            self.labels[new_tr] = f"R-Split {label_of_root}"
            return new_tr
        else:
            raise Warning("No division of the branch")

    def fuse_lineage_tree(
        self,
        l1_root: int,
        l2_root: int,
        length_l1: int = 0,
        length_l2: int = 0,
        length: int = 1,
    ) -> int:
        """Fuses 2 lineages from the lineagetree object. The 2 lineages that are to be fused can have a longer
        first node and the node of the resulting lineage can also be longer.

        Args:
            l1_root (int): Id of the first root
            l2_root (int): Id of the second root
            length_l1 (int, optional): The length of the branch that will be added on top of the first lineage. Defaults to 0, which means only one node will be added.
            length_l2 (int, optional): The length of the branch that will be added on top of the second lineage. Defaults to 0, which means only one node will be added.
            length (int, optional): The length of the branch that will be added on top of the resulting lineage. Defaults to 1.

        Returns:
            int: The id of the root of the new lineage.
        """
        if self.predecessor.get(l1_root) or self.predecessor.get(l2_root):
            raise ValueError("Please select 2 roots.")
        if self.time[l1_root] != self.time[l2_root]:
            warnings.warn(
                "Using lineagetrees that do not exist in the same timepoint. The operation will continue"
            )
        new_root1 = self.add_branch(l1_root, length_l1)
        new_root2 = self.add_branch(l2_root, length_l2)
        next_root1 = self[new_root1][0]
        self.remove_nodes(new_root1)
        self.successor[new_root2].append(next_root1)
        self.predecessor[next_root1] = [new_root2]
        new_branch = self.add_branch(
            new_root2,
            length - 1,
        )
        self.labels[new_branch] = f"Fusion of {new_root1} and {new_root2}"
        return new_branch

    def copy_lineage(self, root: int) -> int:
        """
        Copies the structure of a tree and makes a new with new nodes.
        Warning does not take into account the predecessor of the root node.

        Args:
            root (int): The root of the tree to be copied

        Returns:
            int: The root of the new tree.
        """
        new_nodes = {
            old_node: self.get_next_id()
            for old_node in self.get_sub_tree(root)
        }
        self.nodes.update(new_nodes.values())
        for old_node, new_node in new_nodes.items():
            self.time[new_node] = self.time[old_node]
            succ = self._successor.get(old_node)
            if succ:
                self._successor[new_node] = [new_nodes[n] for n in succ]
            pred = self._predecessor.get(old_node)
            if pred:
                self._predecessor[new_node] = [new_nodes[n] for n in pred]
            self.pos[new_node] = self.pos[old_node] + 0.5
        new_root = new_nodes[root]
        self.labels[new_root] = f"Copy of {root}"
        if self.time[new_root] == 0:
            self.roots.add(new_root)
        return new_root

    def add_root(self, t: int, pos: list = None):
        """Adds a root to a specific timepoint.

        Args:
            t (int): The timepoint the node is going to be added.
            pos (list): The position of the new node.
        """
        C_next = self.get_next_id()
        self._successor[C_next] = ()
        self._predecessor[C_next] = ()
        self._time[C_next] = t
        self.pos[C_next] = pos if isinstance(pos, list) else ()
        self._changed_roots = True
        return C_next

    def _add_node(
        self,
        succ: list = None,
        pred: list = None,
        pos: np.ndarray = None,
        nid: int = None,
    ) -> int:
        """Adds a node to the LineageTree object that is either a successor or a predecessor.

        Args:
            succ (int): id of the node the new node is a successor to
            pred (int): id of the node the new node is a predecessor to
            pos (bool): The position of the new nodes. ###TODO callable###
            nid (int): id value of the new node, to be used carefully,
                if None is provided the new id is automatically computed.
        Returns:
            int: id of the new node.
        """
        if not succ and not pred:
            raise Warning(
                "Please enter a successor or a predecessor, otherwise use the add_roots() function."
            )
        C_next = self.get_next_id() if nid is None else nid
        if succ:
            self._successor[C_next] = succ
            for suc in succ:
                self._predecessor[suc] = (C_next,)
        else:
            self._successor[C_next] = ()
        if pred:
            self._predecessor[C_next] = pred
            self._successor[pred[0]] = self._successor.setdefault(
                pred[0], ()
            ) + (C_next,)
        else:
            self._predecessor[C_next] = ()
        if isinstance(pos, list):
            self.pos[C_next] = pos
        return C_next

    def remove_nodes(self, group: int | set | list) -> None:
        """Removes a group of nodes from the LineageTree

        Args:
            group (set | list | int): One or more nodes that are to be removed.
        """
        if isinstance(group, int | float):
            group = {group}
        if isinstance(group, list):
            group = set(group)
        group = self.nodes.intersection(group)
        for node in group:
            for attr in self.__dict__:
                attr_value = self.__getattribute__(attr)
                if isinstance(attr_value, dict) and attr not in [
                    "successor",
                    "predecessor",
                    "_successor",
                    "_predecessor",
                    "_time",
                ]:
                    attr_value.pop(node, ())
            if self._predecessor.get(node):
                self._successor[self._predecessor[node][0]] = tuple(
                    set(
                        self._successor[self._predecessor[node][0]]
                    ).difference(group)
                )
            for p_node in self._successor.get(node, []):
                self._predecessor[p_node] = ()
            self._predecessor.pop(node, ())
            self._successor.pop(node, ())

    def modify_branch(self, node: int, new_length: int) -> None:
        """Changes the length of a branch, so it adds or removes nodes
        to make the correct length of the cycle.

        Args:
            node (int): Any node of the branch to be modified/
            new_length (int): The new length of the tree.
        """
        if new_length < 1:
            raise Warning(
                "New length should be >0, I f you want to remove a branch call remove_nodes"
            )
        cycle = self.get_cycle(node)
        length = len(cycle)
        if length == new_length:
            return
        successors = self._successor.get(cycle[-1])
        if length < new_length:
            nodes_to_change_time_vals = set(self.get_sub_tree(node))
            for _ in range(new_length - length):
                old_node = node
                node = self._add_node(
                    succ=self._successor[old_node],
                    pred=self._predecessor[old_node],
                )
        if new_length < length:
            self.remove_nodes(cycle[0 : length - new_length - 1])
            for _ in range(new_length):
                old_node = node
                node = self._add_node(
                    succ=self._successor[old_node],
                    pred=self._predecessor[old_node],
                )

    @property
    def t_b(self):
        return min(self.time.values())

    @property
    def t_e(self):
        return max(self.time.values())

    @property
    def nodes(self):
        return frozenset(self.successor.keys())

    @property
    def time(self):
        return MappingProxyType(self._time)

    @property
    def successor(self):
        return MappingProxyType(self._successor)

    @property
    def predecessor(self):
        return MappingProxyType(self._predecessor)

    @property
    def depth(self) -> dict[int, int]:
        if not hasattr(self, "_depth"):
            self._depth = {}
            for leaf in self.leaves:
                self._depth[leaf] = 1
                while leaf in self._predecessor:
                    parent = self._predecessor[leaf][0]
                    current_depth = self._depth.get(parent, 0)
                    self._depth[parent] = max(
                        self._depth[leaf] + 1, current_depth
                    )
                    leaf = parent
            for root in self.roots - set(self._depth):
                self._depth[root] = 1
        return self._depth

    @property
    def roots(self):
        if not hasattr(self, "_roots") or self._changed_roots:
            self._roots = {s for s, p in self._predecessor.items() if p == ()}
            self._changed_roots = True
        return self._roots

    @property
    def edges(self):
        return {(k, vi) for k, v in self._successor.items() for vi in v}

    @property
    def leaves(self):
        if not hasattr(self, "_leaves") or self._changed_leaves:
            self._leaves = {p for p, s in self._successor.items() if s == ()}
            self._changed_leaves = True
        return self._leaves

    @property
    def labels(self):
        if not hasattr(self, "_labels"):
            if hasattr(self, "cell_name"):
                self._labels = {
                    i: self.cell_name.get(i, "Unlabeled") for i in self.roots
                }
            else:
                self._labels = {
                    root: "Unlabeled"
                    for root in self.roots
                    for leaf in self.find_leaves(root)
                    if abs(self.time[leaf] - self.time[root])
                    >= abs(self.t_e - self.t_b) / 4
                }
        return self._labels

    @property
    def time_resolution(self):
        if not hasattr(self, "_time_resolution"):
            self.time_resolution = 1
        return self._time_resolution / 10

    @time_resolution.setter
    def time_resolution(self, time_resolution):
        if time_resolution is not None:
            self._time_resolution = int(time_resolution * 10)
        else:
            warnings.warn("Time resolution set to default 0", stacklevel=2)
            self._time_resolution = 10

    def __setstate__(self, state):
        if "_successor" not in state:
            state["_successor"] = state["successor"]
        if "_predecessor" not in state:
            state["_predecessor"] = state["predecessor"]
        if "_time" not in state:
            state["_time"] = state["time"]
        self.__dict__.update(state)

    def _get_height(self, c: int, done: dict) -> float:
        """Recursively computes the height of a cell within a tree * a space factor.
        This function is specific to the function write_to_svg.

        Args:
            c (int): id of a cell in a lineage tree from which the height will be computed from
            done ({int: [int, int]}): a dictionary that maps a cell id to its vertical and horizontal position
        Returns:
            float: the height of the cell `c`
        """
        if c in done:
            return done[c][0]
        else:
            P = np.mean(
                [self._get_height(di, done) for di in self._successor[c]]
            )
            done[c] = [P, self.vert_space_factor * self.time[c]]
            return P

    def write_to_svg(
        self,
        file_name: str,
        roots: list = None,
        draw_nodes: bool = True,
        draw_edges: bool = True,
        order_key: Callable = None,
        vert_space_factor: float = 0.5,
        horizontal_space: float = 1,
        node_size: Callable = None,
        stroke_width: Callable = None,
        factor: float = 1.0,
        node_color: Callable = None,
        stroke_color: Callable = None,
        positions: dict = None,
        node_color_map: Callable = None,
        normalize: bool = True,
    ) -> None:
        """Writes the lineage tree to an SVG file.
        Node and edges coloring and size can be provided.

        Args:
            file_name (str): filesystem filename valid for `open()`
            roots ([int, ...]): list of node ids to be drawn. If `None` all the nodes will be drawn. Default `None`
            draw_nodes (bool): wether to print the nodes or not, default `True`
            draw_edges (bool): wether to print the edges or not, default `True`
            order_key (Callable): function that would work for the attribute `key=` for the `sort`/`sorted` function
            vert_space_factor (float): the vertical position of a node is its time. `vert_space_factor` is a
                               multiplier to space more or less nodes in time
            horizontal_space (float): space between two consecutive nodes
            node_size (Callable | str): a function that maps a node id to a `float` value that will determine the
                       radius of the node. The default function return the constant value `vertical_space_factor/2.1`
                       If a string is given instead and it is a property of the tree,
                       the the size will be mapped according to the property
            stroke_width (Callable): a function that maps a node id to a `float` value that will determine the
                          width of the daughter edge.  The default function return the constant value `vertical_space_factor/2.1`
            factor (float): scaling factor for nodes positions, default 1
            node_color (Callable | str): a function that maps a node id to a triplet between 0 and 255.
                        The triplet will determine the color of the node. If a string is given instead and it is a property
                        of the tree, the the color will be mapped according to the property
            node_color_map (Callable | str): the name of the colormap to use to color the nodes, or a colormap function
            stroke_color (Callable): a function that maps a node id to a triplet between 0 and 255.
                          The triplet will determine the color of the stroke of the inward edge.
            positions ({int: [float, float], ...}): dictionary that maps a node id to a 2D position.
                       Default `None`. If provided it will be used to position the nodes.
        """

        def normalize_values(v, nodes, _range, shift, mult):
            min_ = np.percentile(v, 1)
            max_ = np.percentile(v, 99)
            values = _range * ((v - min_) / (max_ - min_)) + shift
            values_dict_nodes = dict(zip(nodes, values, strict=True))
            return lambda x: values_dict_nodes[x] * mult

        if roots is None:
            roots = self.roots
            if hasattr(self, "image_label"):
                roots = [cell for cell in roots if self.image_label[cell] != 1]

        if node_size is None:

            def node_size(x):
                return vert_space_factor / 2.1

        elif isinstance(node_size, str) and node_size in self.__dict__:
            values = np.array(
                [self._successor[node_size][c] for c in self.nodes]
            )
            node_size = normalize_values(
                values, self.nodes, 0.5, 0.5, vert_space_factor / 2.1
            )
        if stroke_width is None:

            def stroke_width(x):
                return vert_space_factor / 2.2

        if node_color is None:

            def node_color(x):
                return 0, 0, 0

        elif isinstance(node_color, str) and node_color in self.__dict__:
            if isinstance(node_color_map, str):
                from matplotlib import colormaps

                if node_color_map in colormaps:
                    node_color_map = colormaps[node_color_map]
                else:
                    node_color_map = colormaps["viridis"]
            values = np.array(
                [self._successor[node_color][c] for c in self.nodes]
            )
            normed_vals = normalize_values(values, self.nodes, 1, 0, 1)

            def node_color(x):
                return [k * 255 for k in node_color_map(normed_vals(x))[:-1]]

        coloring_edges = stroke_color is not None
        if not coloring_edges:

            def stroke_color(x):
                return 0, 0, 0

        elif isinstance(stroke_color, str) and stroke_color in self.__dict__:
            if isinstance(node_color_map, str):
                from matplotlib import colormaps

                if node_color_map in colormaps:
                    node_color_map = colormaps[node_color_map]
                else:
                    node_color_map = colormaps["viridis"]
            values = np.array(
                [self._successor[stroke_color][c] for c in self.nodes]
            )
            normed_vals = normalize_values(values, self.nodes, 1, 0, 1)

            def stroke_color(x):
                return [k * 255 for k in node_color_map(normed_vals(x))[:-1]]

        prev_x = 0
        self.vert_space_factor = vert_space_factor
        if order_key is not None:
            roots.sort(key=order_key)
        treated_cells = []

        pos_given = positions is not None
        if not pos_given:
            positions = dict(
                zip(
                    self.nodes,
                    [
                        [0.0, 0.0],
                    ]
                    * len(self.nodes),
                    strict=True,
                ),
            )
        for _i, r in enumerate(roots):
            r_leaves = []
            to_do = [r]
            while len(to_do) != 0:
                curr = to_do.pop(0)
                treated_cells += [curr]
                if not self._successor[curr]:
                    if order_key is not None:
                        to_do += sorted(self._successor[curr], key=order_key)
                    else:
                        to_do += self._successor[curr]
                else:
                    r_leaves += [curr]
            r_pos = {
                leave: [
                    prev_x + horizontal_space * (1 + j),
                    self.vert_space_factor * self.time[leave],
                ]
                for j, leave in enumerate(r_leaves)
            }
            self._get_height(r, r_pos)
            prev_x = np.max(list(r_pos.values()), axis=0)[0]
            if not pos_given:
                positions.update(r_pos)

        dwg = svgwrite.Drawing(
            file_name,
            profile="tiny",
            size=factor * np.max(list(positions.values()), axis=0),
        )
        if draw_edges and not draw_nodes and not coloring_edges:
            to_do = set(treated_cells)
            while len(to_do) > 0:
                curr = to_do.pop()
                c_cycle = self.get_cycle(curr)
                x1, y1 = positions[c_cycle[0]]
                x2, y2 = positions[c_cycle[-1]]
                dwg.add(
                    dwg.line(
                        (factor * x1, factor * y1),
                        (factor * x2, factor * y2),
                        stroke=svgwrite.rgb(0, 0, 0),
                    )
                )
                for si in self._successor[c_cycle[-1]]:
                    x3, y3 = positions[si]
                    dwg.add(
                        dwg.line(
                            (factor * x2, factor * y2),
                            (factor * x3, factor * y3),
                            stroke=svgwrite.rgb(0, 0, 0),
                        )
                    )
                to_do.difference_update(c_cycle)
        else:
            for c in treated_cells:
                x1, y1 = positions[c]
                for si in self._successor[c]:
                    x2, y2 = positions[si]
                    if draw_edges:
                        dwg.add(
                            dwg.line(
                                (factor * x1, factor * y1),
                                (factor * x2, factor * y2),
                                stroke=svgwrite.rgb(*(stroke_color(si))),
                                stroke_width=svgwrite.pt(stroke_width(si)),
                            )
                        )
            for c in treated_cells:
                x1, y1 = positions[c]
                if draw_nodes:
                    dwg.add(
                        dwg.circle(
                            (factor * x1, factor * y1),
                            node_size(c),
                            fill=svgwrite.rgb(*(node_color(c))),
                        )
                    )
        dwg.save()

    def to_tlp(
        self,
        fname: str,
        t_min: int = -1,
        t_max: int = np.inf,
        nodes_to_use: list = None,
        temporal: bool = True,
        spatial: str = None,
        write_layout: bool = True,
        node_properties: dict = None,
        Names: bool = False,
    ) -> None:
        """Write a lineage tree into an understable tulip file.

        Args:
            fname (str): path to the tulip file to create
            t_min (int): minimum time to consider, default -1
            t_max (int): maximum time to consider, default np.inf
            nodes_to_use ([int, ]): list of nodes to show in the graph,
                          default *None*, then self.nodes is used
                          (taking into account *t_min* and *t_max*)
            temporal (bool): True if the temporal links should be printed, default True
            spatial (str): Build spatial edges from a spatial neighbourhood graph.
                The graph has to be computed before running this function
                'ball': neighbours at a given distance,
                'kn': k-nearest neighbours,
                'GG': gabriel graph,
                None: no spatial edges are writen.
                Default None
            write_layout (bool): True, write the spatial position as layout,
                                   False, do not write spatial positionm
                                   default True
            node_properties ({`p_name`, [{id, p_value}, default]}): a dictionary of properties to write
                                                To a key representing the name of the property is
                                                paired a dictionary that maps a cell id to a property
                                                and a default value for this property
            Names (bool): Only works with ASTEC outputs, True to sort the cells by their names
        """

        def format_names(names_which_matter):
            """Return an ensured formated cell names"""
            tmp = {}
            for k, v in names_which_matter.items():
                tmp[k] = (
                    v.split(".")[0][0]
                    + "{:02d}".format(int(v.split(".")[0][1:]))
                    + "."
                    + "{:04d}".format(int(v.split(".")[1][:-1]))
                    + v.split(".")[1][-1]
                )
            return tmp

        def spatial_adjlist_to_set(s_g):
            s_edges = set()
            for _t, gg in s_g.items():
                for c, N in gg.items():
                    s_edges.update([tuple(sorted([c, ni])) for ni in N])
            return s_edges

        with open(fname, "w") as f:
            f.write('(tlp "2.0"\n')
            f.write("(nodes ")

            if spatial:
                if spatial.lower() == "gg" and hasattr(self, "Gabriel_graph"):
                    s_edges = spatial_adjlist_to_set(self.Gabriel_graph)
                elif spatial.lower() == "kn" and hasattr(self, "kn_graph"):
                    s_edges = spatial_adjlist_to_set(self.kn_graph)
                elif spatial.lower() == "ball" and hasattr(self, "th_edges"):
                    s_edges = spatial_adjlist_to_set(self.th_edges)

            if not nodes_to_use:
                if t_max != np.inf or t_min > -1:
                    nodes_to_use = [
                        n for n in self.nodes if t_min < self.time[n] <= t_max
                    ]
                    edges_to_use = []
                    if temporal:
                        edges_to_use += [
                            e
                            for e in self.edges
                            if t_min < self.time[e[0]] < t_max
                        ]
                    if spatial:
                        edges_to_use += [
                            e
                            for e in s_edges
                            if t_min < self.time[e[0]] < t_max
                        ]
                else:
                    nodes_to_use = list(self.nodes)
                    edges_to_use = []
                    if temporal:
                        edges_to_use += list(self.edges)
                    if spatial:
                        edges_to_use += list(s_edges)
            else:
                edges_to_use = []
                nodes_to_use = set(nodes_to_use)
                if temporal:
                    for n in nodes_to_use:
                        for d in self._successor[n]:
                            if d in nodes_to_use:
                                edges_to_use.append((n, d))
                if spatial:
                    edges_to_use += [
                        e for e in s_edges if t_min < self.time[e[0]] < t_max
                    ]
            nodes_to_use = set(nodes_to_use)
            if Names:
                names_which_matter = {
                    k: v
                    for k, v in node_properties[Names][0].items()
                    if v != "" and v != "NO" and k in nodes_to_use
                }
                names_formated = format_names(names_which_matter)
                order_on_nodes = np.array(list(names_formated.keys()))[
                    np.argsort(list(names_formated.values()))
                ]
                nodes_to_use = set(nodes_to_use).difference(order_on_nodes)
                tmp_names = {}
                for k, v in node_properties[Names][0].items():
                    if (
                        len(
                            self._successor.get(
                                self._predecessor.get(k, [-1])[0], ()
                            )
                        )
                        != 1
                        or self.time[k] == t_min + 1
                    ):
                        tmp_names[k] = v
                node_properties[Names][0] = tmp_names
                for n in order_on_nodes:
                    f.write(str(n) + " ")
            else:
                order_on_nodes = set()

            for n in nodes_to_use:
                f.write(str(n) + " ")
            f.write(")\n")

            nodes_to_use.update(order_on_nodes)

            for i, e in enumerate(edges_to_use):
                f.write(
                    "(edge "
                    + str(i)
                    + " "
                    + str(e[0])
                    + " "
                    + str(e[1])
                    + ")\n"
                )

            f.write('(property 0 int "time"\n')
            f.write('\t(default "0" "0")\n')
            for n in nodes_to_use:
                f.write(
                    "\t(node " + str(n) + ' "' + str(self.time[n]) + '")\n'
                )
            f.write(")\n")

            if write_layout:
                f.write('(property 0 layout "viewLayout"\n')
                f.write('\t(default "(0, 0, 0)" "()")\n')
                for n in nodes_to_use:
                    f.write(
                        "\t(node "
                        + str(n)
                        + ' "'
                        + str(tuple(self.pos[n]))
                        + '")\n'
                    )
                f.write(")\n")
                f.write('(property 0 double "distance"\n')
                f.write('\t(default "0" "0")\n')
                for i, e in enumerate(edges_to_use):
                    d_tmp = np.linalg.norm(self.pos[e[0]] - self.pos[e[1]])
                    f.write("\t(edge " + str(i) + ' "' + str(d_tmp) + '")\n')
                    f.write(
                        "\t(node " + str(e[0]) + ' "' + str(d_tmp) + '")\n'
                    )
                f.write(")\n")

            if node_properties:
                for p_name, (p_dict, default) in node_properties.items():
                    if isinstance(list(p_dict.values())[0], str):
                        f.write(f'(property 0 string "{p_name}"\n')
                        f.write(f"\t(default {default} {default})\n")
                    elif isinstance(list(p_dict.values())[0], Number):
                        f.write(f'(property 0 double "{p_name}"\n')
                        f.write('\t(default "0" "0")\n')
                    for n in nodes_to_use:
                        f.write(
                            "\t(node "
                            + str(n)
                            + ' "'
                            + str(p_dict.get(n, default))
                            + '")\n'
                        )
                    f.write(")\n")

            f.write(")")
            f.close()

    def to_binary(self, fname: str, starting_points: list = None) -> None:
        """Writes the lineage tree (a forest) as a binary structure
        (assuming it is a binary tree, it would not work for *n* ary tree with 2 < *n*).
        The binary file is composed of 3 sequences of numbers and
        a header specifying the size of each of these sequences.
        The first sequence, *number_sequence*, represents the lineage tree
        as a DFT preporder transversal list. -1 signifying a leaf and -2 a branching
        The second sequence, *time_sequence*, represent the starting time of each tree.
        The third sequence, *pos_sequence*, reprensent the 3D coordinates of the objects.
        The header specify the size of each of these sequences.
        Each size is stored as a long long
        The *number_sequence* is stored as a list of long long (0 -> 2^(8*8)-1)
        The *time_sequence* is stored as a list of unsigned short (0 -> 2^(8*2)-1)
        The *pos_sequence* is stored as a list of double.

        Args:
            fname (str): name of the binary file
            starting_points ([int, ]): list of the roots to be written.
                If None, all roots are written, default value, None
        """
        if starting_points is None:
            starting_points = list(self.roots)
        number_sequence = [-1]
        pos_sequence = []
        time_sequence = []
        for c in starting_points:
            time_sequence.append(self.time.get(c, 0))
            to_treat = [c]
            while to_treat:
                curr_c = to_treat.pop()
                number_sequence.append(curr_c)
                pos_sequence += list(self.pos[curr_c])
                if self._successor[curr_c] == ():
                    number_sequence.append(-1)
                elif len(self._successor[curr_c]) == 1:
                    to_treat += self._successor[curr_c]
                else:
                    number_sequence.append(-2)
                    to_treat += self._successor[curr_c]
        remaining_nodes = set(self.nodes) - set(number_sequence)

        for c in remaining_nodes:
            time_sequence.append(self.time.get(c, 0))
            number_sequence.append(c)
            pos_sequence += list(self.pos[c])
            number_sequence.append(-1)

        with open(fname, "wb") as f:
            f.write(struct.pack("q", len(number_sequence)))
            f.write(struct.pack("q", len(time_sequence)))
            f.write(struct.pack("q", len(pos_sequence)))
            f.write(struct.pack("q" * len(number_sequence), *number_sequence))
            f.write(struct.pack("H" * len(time_sequence), *time_sequence))
            f.write(struct.pack("d" * len(pos_sequence), *pos_sequence))

            f.close()

    def write(self, fname: str) -> None:
        """
        Write a lineage tree on disk as an .lT file.

        Args:
            fname (str): path to and name of the file to save
        """
        if os.path.splitext(fname)[-1] != ".lT":
            fname = os.path.extsep.join((fname, "lT"))
        with open(fname, "bw") as f:
            pkl.dump(self, f)
            f.close()

    @classmethod
    def load(clf, fname: str) -> "lineageTree":
        """
        Loading a lineage tree from a ".lT" file.

        Args:
            fname (str): path to and name of the file to read

        Returns:
            (lineageTree): loaded file
        """
        with open(fname, "br") as f:
            lT = pkl.load(f)
            f.close()
        if not hasattr(lT, "time_resolution"):
            lT.time_resolution = None
        for node in lT.nodes.difference(lT.predecessor):
            lT._predecessor[node] = ()
        for node in set(lT.predecessor).difference(lT.successor):
            lT._successor[node] = ()
        for k, v in lT.successor.items():
            lT._successor[k] = tuple(v)
        for k, v in lT.predecessor.items():
            lT._predecessor[k] = tuple(v)
        return lT

    def get_idx3d(self, t: int) -> tuple:
        """Get a 3d kdtree for the dataset at time *t* .
        The  kdtree is stored in *self.kdtrees[t]*

        Args:
            t (int): time
        Returns:
            (kdtree, [int, ]): the built kdtree and
                the correspondancy list,
                If the query in the kdtree gives you the value i,
                then it corresponds to the id in the tree to_check_self[i]
        """
        to_check_self = list(self.nodes_at_t(t=t))
        if t not in self.kdtrees:
            data_corres = {}
            data = []
            for i, C in enumerate(to_check_self):
                data.append(tuple(self.pos[C]))
                data_corres[i] = C
            idx3d = KDTree(data)
            self.kdtrees[t] = idx3d
        else:
            idx3d = self.kdtrees[t]
        return idx3d, np.array(to_check_self)

    def get_gabriel_graph(self, t: int) -> dict[int, set[int]]:
        """Build the Gabriel graph of the given graph for time point `t`
        The Garbiel graph is then stored in self.Gabriel_graph and returned
        *WARNING: the graph is not recomputed if already computed. even if nodes were added*.

        Args:
            t (int): time
        Returns:
            {int, set([int, ])}: a dictionary that maps a node to
                the set of its neighbors
        """
        if not hasattr(self, "Gabriel_graph"):
            self.Gabriel_graph = {}

        if t not in self.Gabriel_graph:
            idx3d, nodes = self.get_idx3d(t)

            data_corres = {}
            data = []
            for i, C in enumerate(nodes):
                data.append(self.pos[C])
                data_corres[i] = C

            tmp = Delaunay(data)

            delaunay_graph = {}

            for N in tmp.simplices:
                for e1, e2 in combinations(np.sort(N), 2):
                    delaunay_graph.setdefault(e1, set()).add(e2)
                    delaunay_graph.setdefault(e2, set()).add(e1)

            Gabriel_graph = {}

            for e1, neighbs in delaunay_graph.items():
                for ni in neighbs:
                    if not any(
                        np.linalg.norm((data[ni] + data[e1]) / 2 - data[i])
                        < np.linalg.norm(data[ni] - data[e1]) / 2
                        for i in delaunay_graph[e1].intersection(
                            delaunay_graph[ni]
                        )
                    ):
                        Gabriel_graph.setdefault(data_corres[e1], set()).add(
                            data_corres[ni]
                        )
                        Gabriel_graph.setdefault(data_corres[ni], set()).add(
                            data_corres[e1]
                        )

            self.Gabriel_graph[t] = Gabriel_graph

        return self.Gabriel_graph[t]

    def get_predecessors(
        self, x: int, depth: int = None, start_time: int = None, end_time=None
    ) -> list[int]:
        """Computes the predecessors of the node `x` up to
        `depth` predecessors or the begining of the life of `x`.
        The ordered list of ids is returned.

        Args:
            x (int): id of the node to compute
            depth (int): maximum number of predecessors to return
        Returns:
            [int, ]: list of ids, the last id is `x`
        """
        if not start_time:
            start_time = self.t_b
        if not end_time:
            end_time = self.t_e
        unconstrained_cycle = [x]
        cycle = [x] if start_time <= self.time[x] <= end_time else []
        acc = 0
        while (
            acc != depth
            and start_time < self.time[unconstrained_cycle[0]]
            and (
                self._predecessor[unconstrained_cycle[0]] != ()
                and (  # Please dont change very important even if it looks weird.
                    len(
                        self._successor[
                            self._predecessor[unconstrained_cycle[0]][0]
                        ]
                    )
                    == 1
                )
            )
        ):
            unconstrained_cycle.insert(
                0, self._predecessor[unconstrained_cycle[0]][0]
            )
            acc += 1
            if start_time <= self.time[unconstrained_cycle[0]] <= end_time:
                cycle.insert(0, unconstrained_cycle[0])

        return cycle

    def get_successors(
        self, x: int, depth: int = None, end_time: int = None
    ) -> list[int]:
        """Computes the successors of the node `x` up to
        `depth` successors or the end of the life of `x`.
        The ordered list of ids is returned.

        Args:
            x (int): id of the node to compute
            depth (int): maximum number of predecessors to return
        Returns:
            [int, ]: list of ids, the first id is `x`
        """
        if end_time is None:
            end_time = self.t_e
        cycle = [x]
        acc = 0
        while (
            len(self._successor[cycle[-1]]) == 1
            and acc != depth
            and self.time[cycle[-1]] < end_time
        ):
            cycle += self._successor[cycle[-1]]
            acc += 1

        return cycle

    def get_cycle(
        self,
        x: int,
        depth: int = None,
        depth_pred: int = None,
        depth_succ: int = None,
        end_time: int = None,
    ) -> list[int]:
        """Computes the predecessors and successors of the node `x` up to
        `depth_pred` predecessors plus `depth_succ` successors.
        If the value `depth` is provided and not None,
        `depth_pred` and `depth_succ` are overwriten by `depth`.
        The ordered list of ids is returned.
        If all `depth` are None, the full cycle is returned.

        Args:
            x (int): id of the node to compute
            depth (int): maximum number of predecessors and successor to return
            depth_pred (int): maximum number of predecessors to return
            depth_succ (int): maximum number of successors to return
        Returns:
            [int, ]: list of ids
        """
        if end_time is None:
            end_time = self.t_e
        if depth is not None:
            depth_pred = depth_succ = depth
        return self.get_predecessors(x, depth_pred, end_time=end_time)[
            :-1
        ] + self.get_successors(x, depth_succ, end_time=end_time)

    @property
    def all_tracks(self) -> list[list[int]]:
        if not hasattr(self, "_all_tracks"):
            self._all_tracks = self.get_all_tracks()
        return self._all_tracks

    def get_all_branches_of_node(
        self, node: int, end_time: int = None
    ) -> list[list[int]]:
        """Computes all the tracks of the subtree spawn by a given node.
        Similar to get_all_tracks().

        Args:
            node (int, optional): The node that we want to get its branches.

        Returns:
            ([[int, ...], ...]): list of lists containing track cell ids
        """
        if not end_time:
            end_time = self.t_e
        branches = [self.get_successors(node)]
        to_do = list(self._successor[branches[0][-1]])
        while to_do:
            current = to_do.pop()
            track = self.get_successors(current, end_time=end_time)
            # if len(track) != 1 or self.time[current] <= end_time:
            if self.time[track[-1]] <= end_time:
                branches += [track]
                to_do += self._successor[track[-1]]
        return branches

    def get_all_tracks(self, force_recompute: bool = False) -> list[list[int]]:
        """Computes all the tracks of a given lineage tree,
        stores it in `self.all_tracks` and returns it.

        Returns:
            ([[int, ...], ...]): list of lists containing track cell ids
        """
        if not hasattr(self, "_all_tracks") or force_recompute:
            self._all_tracks = []
            to_do = list(self.roots)
            while len(to_do) != 0:
                current = to_do.pop()
                track = self.get_cycle(current)
                self._all_tracks += [track]
                to_do.extend(self._successor[track[-1]])
        return self._all_tracks

    def get_tracks(self, roots: list = None) -> list[list[int]]:
        """Computes the tracks given by the list of nodes `roots` and returns it.

        Args:
            roots (list): list of ids of the roots to be computed
        Returns:
            ([[int, ...], ...]): list of lists containing track cell ids
        """
        if roots is None:
            return self.get_all_tracks(force_recompute=True)
        else:
            tracks = []
            to_do = list(roots)
            while len(to_do) != 0:
                current = to_do.pop()
                track = self.get_cycle(current)
                tracks.append(track)
                to_do.extend(self._successor[track[-1]])
            return tracks

    def find_leaves(self, roots: int | Iterable) -> set[int]:
        """Finds the leaves of a tree spawned by one or more nodes.

        Args:
            roots (int | Iterable): The roots of the trees.

        Returns:
            set: The leaves of one or more trees.
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

    def get_sub_tree(
        self,
        x: int | Iterable,
        end_time: int | None = None,
        preorder: bool = False,
    ) -> list[int]:
        """Computes the list of cells from the subtree spawned by *x*
        The default output order is breadth first traversal.
        Unless preorder is `True` in that case the order is
        Depth first traversal preordered.

        Args:
            x (int): id of root node
            preorder (bool): if True the output is preorder DFT
        Returns:
            ([int, ...]): the ordered list of node ids
        """
        if not end_time:
            end_time = self.t_e
        if not isinstance(x, Iterable):
            to_do = [x]
        elif isinstance(x, Iterable):
            to_do = list(x)
        sub_tree = []
        while to_do:
            curr = to_do.pop()
            succ = self._successor[curr]
            if succ and end_time < self.time.get(curr, end_time):
                succ = []
                continue
            if preorder:
                to_do = succ + to_do
            else:
                to_do += succ
                sub_tree += [curr]
        return sub_tree

    def compute_spatial_density(
        self, t_b: int = None, t_e: int = None, th: float = 50
    ) -> dict[int, float]:
        """Computes the spatial density of cells between `t_b` and `t_e`.
        The spatial density is computed as follow:
        #cell/(4/3*pi*th^3)
        The results is stored in self.spatial_density is returned.

        Args:
            t_b (int): starting time to look at, default first time point
            t_e (int): ending time to look at, default last time point
            th (float): size of the neighbourhood
        Returns:
            {int, float}: dictionary that maps a cell id to its spatial density
        """
        s_vol = 4 / 3.0 * np.pi * th**3
        time_range = set(range(self.t_b, self.t_e)).intersection(
            self.time.values()
        )
        for t in time_range:
            idx3d, nodes = self.get_idx3d(t)
            nb_ni = [
                (len(ni) - 1) / s_vol
                for ni in idx3d.query_ball_tree(idx3d, th)
            ]
            self.spatial_density.update(dict(zip(nodes, nb_ni, strict=True)))
        return self.spatial_density

    def compute_k_nearest_neighbours(self, k: int = 10) -> dict[int, set[int]]:
        """Computes the k-nearest neighbors
        Writes the output in the attribute `kn_graph`
        and returns it.

        Args:
            k (float): number of nearest neighours
        Returns:
            {int, set([int, ...])}: dictionary that maps
                a cell id to its `k` nearest neighbors
        """
        self.kn_graph = {}
        for t, nodes in self.time_nodes.items():
            use_k = k if k < len(nodes) else len(nodes)
            idx3d, nodes = self.get_idx3d(t)
            pos = [self.pos[c] for c in nodes]
            _, neighbs = idx3d.query(pos, use_k)
            out = dict(
                zip(nodes, [set(nodes[ni[1:]]) for ni in neighbs], strict=True)
            )
            self.kn_graph.update(out)
        return self.kn_graph

    def compute_spatial_edges(self, th: int = 50) -> dict[int, set[int]]:
        """Computes the neighbors at a distance `th`
        Writes the output in the attribute `th_edge`
        and returns it.

        Args:
            th (float): distance to consider neighbors
        Returns:
            {int, set([int, ...])}: dictionary that maps
                a cell id to its neighbors at a distance `th`
        """
        self.th_edges = {}
        for t, _ in self.time_nodes.items():
            idx3d, nodes = self.get_idx3d(t)
            neighbs = idx3d.query_ball_tree(idx3d, th)
            out = dict(
                zip(nodes, [set(nodes[ni]) for ni in neighbs], strict=True)
            )
            self.th_edges.update(
                {k: v.difference([k]) for k, v in out.items()}
            )
        return self.th_edges

    def main_axes(self, time: int = None) -> tuple[np.array, np.array]:
        """Finds the main axes for a timepoint.
        If none will select the timepoint with the highest amound of cells.

        Args:
            time (int, optional): The timepoint to find the main axes.
                                  If None will find the timepoint
                                  with the largest number of cells.

        Returns:
            list: A list that contains the array of eigenvalues and eigenvectors.
        """
        if time is None:
            time = max(self.time_nodes, key=lambda x: len(self.time_nodes[x]))
        pos = np.array([self.pos[node] for node in self.time_nodes[time]])
        pos = pos - np.mean(pos, axis=0)
        cov = np.cov(np.array(pos).T)
        eig_val, eig_vec = np.linalg.eig(cov)
        srt = np.argsort(eig_val)[::-1]
        self.eig_val, self.eig_vec = eig_val[srt], eig_vec[:, srt]
        return eig_val[srt], eig_vec[:, srt]

    def scale_embryo(self, scale=1000) -> float:
        """Scale the embryo using their eigenvalues.

        Args:
            scale (int, optional): The resulting scale you want to achieve. Defaults to 1000.

        Returns:
            float: The scale factor.
        """
        eig = self.main_axes()[0]
        return scale / (np.sqrt(eig[0]))

    @staticmethod
    def __rodrigues_rotation_matrix(vector1, vector2=(0, 1, 0)):
        """Calculates the rodrigues matrix of a dataset. It should use vectors from the find_main_axes(eigenvectors) function of LineagTree.
        Uses the Rodrigues rotation formula.

        Args:
            vector1 (list|np.array): The vector that should be rotated to be aligned to the second vector
            vector2 (list|np.array, optional): The second vector. Defaults to [1,0,0].

        Returns:
            np.array: The rotation matrix.
        """
        vector1 = vector1 / np.linalg.norm(vector1)
        vector2 = vector2 / np.linalg.norm(vector2)
        if vector1 @ vector2 == 1:
            return np.eye(3)
        angle = np.arccos(vector1 @ vector2)
        axis = np.cross(vector1, vector2)
        axis = axis / np.linalg.norm(axis)
        K = np.array(
            [
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ]
        )
        return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K

    def get_ancestor_at_t(self, n: int, time: int = None) -> int:
        """
        Find the id of the ancestor of a give node `n`
        at a given time `time`.

        If there is no ancestor, returns `-1`
        If time is None return the root of the sub tree that spawns
        the node n.

        Args:
            n (int): node for which to look the ancestor
            time (int): time at which the ancestor has to be found.
                If `None` the ancestor at the first time point
                will be found (default `None`)

        Returns:
            (int): the id of the ancestor at time `time`,
                `-1` if it does not exist
        """
        if n not in self.nodes:
            return
        if time is None:
            time = self.t_b
        ancestor = n
        while time < self.time.get(ancestor, self.t_b - 1):
            ancestor = self._predecessor.get(ancestor, [-1])[0]
        return ancestor

    def get_labelled_ancestor(self, node: int) -> int:
        """Finds the first labelled ancestor and returns its ID otherwise returns None

        Args:
            node (int): The id of the node

        Returns:
            None | int: Returns the first ancestor found that has a label otherwise
            None.
        """
        if node not in self.nodes:
            return None
        ancestor = node
        while (
            self.t_b <= self.time.get(ancestor, self.t_b - 1)
            and ancestor != -1
        ):
            if ancestor in self.labels:
                return ancestor
            ancestor = self._predecessor.get(ancestor, [-1])[0]
        return None

    def unordered_tree_edit_distances_at_time_t(
        self,
        t: int,
        end_time: int = None,
        style="simple",
        downsample: int = 2,
        normalize: bool = True,
        recompute: bool = False,
    ) -> dict[int, float]:
        """
        TODO: change docstring
        Compute all the pairwise unordered tree edit distances from Zhang 996 between the trees spawned at time `t`

        Args:
            t (int): time to look at
            delta (callable): comparison function (see edist doc for more information)
            norm (callable): norming function that takes the number of nodes
                of the tree spawned by `n1` and the number of nodes
                of the tree spawned by `n2` as arguments.
            recompute (bool): if True, forces to recompute the distances (default: False)
            end_time (int): The final time point the comparison algorithm will take into account. If None all nodes
                            will be taken into account.

        Returns:
            (dict) a dictionary that maps a pair of cell ids at time `t` to their unordered tree edit distance
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
                normalize=normalize,
            )
        return self.uted[t]

    def unordered_tree_edit_distance(
        self,
        n1: int,
        n2: int,
        end_time: int = None,
        norm: Literal["max", "sum"] | None = "max",
        style="simple",
        downsample: int = 2,
    ) -> float:
        """
        Compute the unordered tree edit distance from Zhang 1996 between the trees spawned
        by two nodes `n1` and `n2`. The topology of the trees are compared and the matching
        cost is given by the function delta (see edist doc for more information).
        The distance is normed by the function norm that takes the two list of nodes
        spawned by the trees `n1` and `n2`.

        Args:
            n1 (int): id of the first node to compare
            n2 (int): id of the second node to compare
            tree_style ("mini","simple","fragmented","full"): Which tree approximation is going to be used for the comparisons.
                                                              Defaults to "fragmented".

        Returns:
            (float) The normed unordered tree edit distance
        """

        tree = tree_style[style].value
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
            return 0
        delta_tmp = partial(
            delta,
            corres1=corres1,
            corres2=corres2,
            times1=times1,
            times2=times2,
        )
        norm1 = tree1.get_norm()
        norm2 = tree2.get_norm()
        norm_dict = {"max": max, "sum": sum, "None": lambda x: 1}
        if norm is None:
            norm = "None"
        if norm not in norm_dict:
            raise Warning(
                "Select a viable normalization method (max, sum, None)"
            )
        return uted.uted(
            nodes1, adj1, nodes2, adj2, delta=delta_tmp
        ) / norm_dict[norm]([norm1, norm2])

    @staticmethod
    def __plot_nodes(
        hier: dict,
        selected_nodes: set,
        color: str,
        size: int,
        ax: plt.Axes,
        default_color: str = "black",
        **kwargs,
    ) -> None:
        """
        Private method that plots the nodes of the tree.
        """
        hier_unselected = np.array(
            [v for k, v in hier.items() if k not in selected_nodes]
        )
        if hier_unselected.any():
            ax.scatter(
                *hier_unselected.T,
                s=size,
                zorder=10,
                color=default_color,
                **kwargs,
            )
        if selected_nodes.intersection(hier.keys()):
            hier_selected = np.array(
                [v for k, v in hier.items() if k in selected_nodes]
            )
            ax.scatter(
                *hier_selected.T, s=size, zorder=10, color=color, **kwargs
            )

    @staticmethod
    def __plot_edges(
        hier: dict,
        lnks_tms: dict,
        selected_edges: set,
        color: str,
        ax: plt.Axes,
        default_color: str = "black",
        **kwargs,
    ) -> None:
        """
        Private method that plots the edges of the tree.
        """
        x, y = [], []
        for pred, succs in lnks_tms["links"].items():
            for succ in succs:
                if pred not in selected_edges or succ not in selected_edges:
                    x.extend((hier[succ][0], hier[pred][0], None))
                    y.extend((hier[succ][1], hier[pred][1], None))
        ax.plot(x, y, linewidth=0.3, zorder=0.1, c=default_color, **kwargs)
        x, y = [], []
        for pred, succs in lnks_tms["links"].items():
            for succ in succs:
                if pred in selected_edges and succ in selected_edges:
                    x.extend((hier[succ][0], hier[pred][0], None))
                    y.extend((hier[succ][1], hier[pred][1], None))
        ax.plot(x, y, linewidth=0.3, zorder=0.2, c=color, **kwargs)

    def draw_tree_graph(
        self,
        hier: dict[int : tuple[int, int]],
        lnks_tms: dict,
        selected_nodes: list | set = None,
        selected_edges: list | set = None,
        color_of_nodes: str = "magenta",
        color_of_edges: str = None,
        size: int = 10,
        ax: plt.Axes = None,
        default_color: str = "black",
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes]:
        """Function to plot the tree graph.

        Args:
            hier (dict): Dictinary that contains the positions of all nodes.
            lnks_tms (dict): 2 dictionaries: 1 contains all links from start of life cycle to end of life cycle and
                                             the succesors of each cell.
                                             1 contains the length of each life cycle.
            selected_nodes (list|set, optional): Which cells are to be selected (Painted with a different color). Defaults to None.
            selected_edges (list|set, optional): Which edges are to be selected (Painted with a different color). Defaults to None.
            color_of_nodes (str, optional): Color of selected nodes. Defaults to "magenta".
            color_of_edges (_type_, optional): Color of selected edges. Defaults to None.
            size (int, optional): Size of the nodes. Defaults to 10.
            ax (plt.Axes, optional): Plot the graph on existing ax. Defaults to None.
            default_color (str, optional): Default color of nodes. Defaults to "black".

        Returns:
            figure, ax: The matplotlib figure and ax object.
        """
        if selected_nodes is None:
            selected_nodes = []
        if selected_edges is None:
            selected_edges = []
        if ax is None:
            figure, ax = plt.subplots()
        else:
            ax.clear()
        if not isinstance(selected_nodes, set):
            selected_nodes = set(selected_nodes)
        if not isinstance(selected_edges, set):
            selected_edges = set(selected_edges)
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
            ax,
            default_color=default_color,
            **kwargs,
        )
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        return ax.get_figure(), ax

    def to_simple_graph(
        self, node: int = None, start_time: int = None
    ) -> dict[int, dict]:
        """Generates a dictionary of graphs where the keys are the index of the graph and
        the values are the graphs themselves which are produced by create_links_and _cycles

        Args:
            node (int, optional): The id of the node/nodes to produce the simple graphs. Defaults to None.
            start_time (int, optional): Important only if there are no nodes it will produce the graph of every
            root that starts before or at start time. Defaults to None.

        Returns:
            (dict): The keys are just index values  0-n and the values are the graphs produced.
        """
        if start_time is None:
            start_time = self.t_b
        if node is None:
            mothers = [
                root for root in self.roots if self.time[root] <= start_time
            ]
        else:
            mothers = node if isinstance(node, list | set) else [node]
        return {
            i: create_links_and_cycles(self, mother)
            for i, mother in enumerate(mothers)
        }

    def plot_all_lineages(
        self,
        nodes: list = None,
        last_time_point_to_consider: int = None,
        nrows: int = 2,
        figsize: tuple[int, int] = (10, 15),
        dpi: int = 100,
        fontsize: int = 15,
        axes: plt.Axes = None,
        vert_gap: int = 1,
        **kwargs,
    ) -> tuple[plt.Figure, plt.Axes, dict]:
        """Plots all lineages.

        Args:
            last_time_point_to_consider (int, optional): Which timepoints and upwards are the graphs to be plotted.
                                                        For example if start_time is 10, then all trees that begin
                                                        on tp 10 or before are calculated. Defaults to None, where
                                                        it will plot all the roots that exist on self.t_b.
            nrows (int):  How many rows of plots should be printed.
            kwargs: args accepted by matplotlib
        """

        nrows = int(nrows)
        if last_time_point_to_consider is None:
            last_time_point_to_consider = self.t_b
        if nrows < 1 or not nrows:
            nrows = 1
            raise Warning("Number of rows has to be at least 1")
        if nodes:
            graphs = {
                i: self.to_simple_graph(node) for i, node in enumerate(nodes)
            }
        else:
            graphs = self.to_simple_graph(
                start_time=last_time_point_to_consider
            )
        pos = {
            i: hierarchical_pos(
                g,
                g["root"],
                ycenter=-int(self.time[g["root"]]),
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

    def plot_node(
        self,
        node: int,
        figsize: tuple[int, int] = (4, 7),
        dpi: int = 150,
        vert_gap: int = 2,
        ax: plt.Axes = None,
        **kwargs,
    ):
        """Plots the subtree spawn by a node.

        Args:
            node (int): The id of the node that is going to be plotted.
            kwargs: args accepted by matplotlib
        """
        graph = self.to_simple_graph(node)
        if len(graph) > 1:
            raise Warning("Please enter only one node")
        graph = graph[0]
        if not ax:
            _, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=dpi)
        self.draw_tree_graph(
            hier=hierarchical_pos(
                graph,
                graph["root"],
                vert_gap=vert_gap,
                ycenter=-int(self.time[node]),
            ),
            lnks_tms=graph,
            ax=ax,
        )
        return ax.get_figure(), ax

    def nodes_at_t(
        self, *, r: [int, Iterable[int]] = None, t: int = None
    ) -> list:
        """
        Returns the list of cells at time `t` that are spawn by the node(s) `r`.

            Args:
                r (int | list): id or list of ids of the spawning node
                t (int): target time, if None goes as far as possible
                        (default None)

            Returns:
                (list) list of nodes at time `t` spawned by `r`
        """
        if not r and r != 0:
            r = self.roots
        if isinstance(r, int):
            r = [r]
        if t is None:
            t = self.t_e
        to_do = list(r)
        final_nodes = []
        while len(to_do) > 0:
            curr = to_do.pop()
            for _next in self._successor[curr]:
                if self.time[_next] < t:
                    to_do.append(_next)
                elif self.time[_next] == t:
                    final_nodes.append(_next)
        if not final_nodes:
            return list(r)
        return final_nodes

    @staticmethod
    def __calculate_diag_line(dist_mat: np.ndarray) -> tuple[float, float]:
        """
        Calculate the line that centers the band w.

            Args:
                dist_mat (matrix): distance matrix obtained by the function calculate_dtw

            Returns:
                (float) Slope
                (float) intercept of the line
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
    ) -> tuple[tuple[int], np.ndarray, float]:
        """
        Find DTW minimum cost between two series using dynamic programming.

            Args:
                dist_mat (matrix): distance matrix obtained by the function calculate_dtw
                start_d (int): start delay
                back_d (int): end delay
                w (int): window constrain
                slope (float): to calculate window - givem by the function __calculate_diag_line
                intercept (flost): to calculate window - givem by the function __calculate_diag_line
                use_absolute (boolean): if the window constraing is calculate by the absolute difference between points (uncentered)

            Returns:
                (tuple of tuples) Aligment path
                (matrix) Cost matrix
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
        self, track1: list, track2: list, threshold: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Interpolate two series that have different lengths

            Args:
                track1 (list): list of nodes of the first cell cycle to compare
                track2 (list): list of nodes of the second cell cycle to compare
                threshold (int): set a maximum number of points a track can have

            Returns:
                (list of list) x, y, z postions for track1
                (list of list) x, y, z postions for track2
        """
        inter1_pos = []
        inter2_pos = []

        track1_pos = np.array([self.pos[c_id] for c_id in track1])
        track2_pos = np.array([self.pos[c_id] for c_id in track2])

        # Both tracks have the same length and size below the threshold - nothing is done
        if len(track1) == len(track2) and (
            len(track1) <= threshold or len(track2) <= threshold
        ):
            return track1_pos, track2_pos
        # Both tracks have the same length but one or more sizes are above the threshold
        elif len(track1) > threshold or len(track2) > threshold:
            sampling = threshold
        # Tracks have different lengths and the sizes are below the threshold
        else:
            sampling = max(len(track1), len(track2))

        for pos in range(3):
            track1_interp = InterpolatedUnivariateSpline(
                np.linspace(0, 1, len(track1_pos[:, pos])),
                track1_pos[:, pos],
                k=1,
            )
            inter1_pos.append(track1_interp(np.linspace(0, 1, sampling)))

            track2_interp = InterpolatedUnivariateSpline(
                np.linspace(0, 1, len(track2_pos[:, pos])),
                track2_pos[:, pos],
                k=1,
            )
            inter2_pos.append(track2_interp(np.linspace(0, 1, sampling)))

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
    ) -> tuple[float, tuple, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate DTW distance between two cell cycles

            Args:
                nodes1 (int): node to compare distance
                nodes2 (int): node to compare distance
                threshold: set a maximum number of points a track can have
                regist (boolean): Rotate and translate trajectories
                start_d (int): start delay
                back_d (int): end delay
                fast (boolean): True if the user wants to run the fast algorithm with window restrains
                w (int): window size
                centered_band (boolean): if running the fast algorithm, True if the windown is centered
                cost_mat_p (boolean): True if print the not normalized cost matrix

            Returns:
                (float) DTW distance
                (tuple of tuples) Aligment path
                (matrix) Cost matrix
                (list of lists) pos_cycle1: rotated and translated trajectories positions
                (list of lists) pos_cycle2: rotated and translated trajectories positions
        """
        nodes1_cycle = self.get_cycle(nodes1)
        nodes2_cycle = self.get_cycle(nodes2)

        interp_cycle1, interp_cycle2 = self.__interpolate(
            nodes1_cycle, nodes2_cycle, threshold
        )

        pos_cycle1 = np.array([self.pos[c_id] for c_id in nodes1_cycle])
        pos_cycle2 = np.array([self.pos[c_id] for c_id in nodes2_cycle])

        if regist:
            R, t = self.__rigid_transform_3D(
                np.transpose(interp_cycle1), np.transpose(interp_cycle2)
            )
            pos_cycle1 = np.transpose(np.dot(R, pos_cycle1.T) + t)

        dist_mat = distance.cdist(pos_cycle1, pos_cycle2, "euclidean")

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
            return cost, path, cost_mat, pos_cycle1, pos_cycle2
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
        Plot DTW cost matrix between two cell cycles in heatmap format

            Args:
                nodes1 (int): node to compare distance
                nodes2 (int): node to compare distance
                start_d (int): start delay
                back_d (int): end delay
                fast (boolean): True if the user wants to run the fast algorithm with window restrains
                w (int): window size
                centered_band (boolean): if running the fast algorithm, True if the windown is centered

            Returns:
                (float) DTW distance
                (figure) Heatmap of cost matrix with opitimal path
        """
        cost, path, cost_mat, pos_cycle1, pos_cycle2 = self.calculate_dtw(
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
        pos_cycle1: list[np.ndarray],
        pos_cycle2: list[np.ndarray],
        nodes1: list[int],
        nodes2: list[int],
        ax: plt.Axes,
        x_idx: list[int],
        y_idx: list[int],
        x_label: str,
        y_label: str,
    ) -> None:
        ax.plot(
            pos_cycle1[:, x_idx],
            pos_cycle1[:, y_idx],
            "-",
            label=f"root = {nodes1}",
        )
        ax.plot(
            pos_cycle2[:, x_idx],
            pos_cycle2[:, y_idx],
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
        projection: str = None,
        alig: bool = False,
    ) -> tuple[float, plt.Figure]:
        """
        Plots DTW trajectories aligment between two cell cycles in 2D or 3D

            Args:
                nodes1 (int): node to compare distance
                nodes2 (int): node to compare distance
                threshold (int): set a maximum number of points a track can have
                regist (boolean): Rotate and translate trajectories
                start_d (int): start delay
                back_d (int): end delay
                w (int): window size
                fast (boolean): True if the user wants to run the fast algorithm with window restrains
                centered_band (boolean): if running the fast algorithm, True if the windown is centered
                projection (string): specify which 2D to plot ->
                    '3d' : for the 3d visualization
                    'xy' or None (default) : 2D projection of axis x and y
                    'xz' : 2D projection of axis x and z
                    'yz' : 2D projection of axis y and z
                    'pca' : PCA projection
                alig (boolean): True to show alignment on plot

            Returns:
                (float) DTW distance
                (figue) Trajectories Plot
        """
        (
            distance,
            alignment,
            cost_mat,
            pos_cycle1,
            pos_cycle2,
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
                pos_cycle1[:, 0],
                pos_cycle1[:, 1],
                pos_cycle1[:, 2],
                "-",
                label=f"root = {nodes1}",
            )
            ax.plot(
                pos_cycle2[:, 0],
                pos_cycle2[:, 1],
                pos_cycle2[:, 2],
                "-",
                label=f"root = {nodes2}",
            )
            ax.set_ylabel("y position")
            ax.set_xlabel("x position")
            ax.set_zlabel("z position")
        else:
            if projection == "xy" or projection == "yx" or projection is None:
                self.__plot_2d(
                    pos_cycle1,
                    pos_cycle2,
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
                    pos_cycle1,
                    pos_cycle2,
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
                    pos_cycle1,
                    pos_cycle2,
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
                        "scikit-learn is not installed, the PCA orientation cannot be used. You can install scikit-learn with pip install"
                    )

                # Apply PCA
                pca = PCA(n_components=2)
                pca.fit(np.vstack([pos_cycle1, pos_cycle2]))
                pos_cycle1_2d = pca.transform(pos_cycle1)
                pos_cycle2_2d = pca.transform(pos_cycle2)

                ax.plot(
                    pos_cycle1_2d[:, 0],
                    pos_cycle1_2d[:, 1],
                    "-",
                    label=f"root = {nodes1}",
                )
                ax.plot(
                    pos_cycle2_2d[:, 0],
                    pos_cycle2_2d[:, 1],
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

        connections = [[pos_cycle1[i], pos_cycle2[j]] for i, j in alignment]

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

    def first_labelling(self):
        self.labels = {i: "Unlabeled" for i in self.roots}

    def __init__(
        self,
        file_format: str = None,
        tb: int = None,
        te: int = None,
        z_mult: float = 1.0,
        file_type: str = "",
        delim: str = ",",
        eigen: bool = False,
        shape: tuple = None,
        raw_size: tuple = None,
        reorder: bool = False,
        xml_attributes: tuple = None,
        name: str = None,
        time_resolution: int | None = None,
    ):
        """
        TODO: complete the doc
        Main library to build tree graph representation of lineage tree data
        It can read TGMM, ASTEC, SVF, MaMuT and TrackMate outputs.

        Args:
            file_format (str): either - path format to TGMM xmls
                                      - path to the MaMuT xml
                                      - path to the binary file
            tb (int, optional):first time point (necessary for TGMM xmls only)
            te (int, optional): last time point (necessary for TGMM xmls only)
            z_mult (float, optional):z aspect ratio if necessary (usually only for TGMM xmls)
            file_type (str, optional):type of input file. Accepts:
                'TGMM, 'ASTEC', MaMuT', 'TrackMate', 'csv', 'celegans', 'binary'
                default is 'binary'
            delim (str, optional): _description_. Defaults to ",".
            eigen (bool, optional): _description_. Defaults to False.
            shape (tuple, optional): _description_. Defaults to None.
            raw_size (tuple, optional): _description_. Defaults to None.
            reorder (bool, optional): _description_. Defaults to False.
            xml_attributes (tuple, optional): _description_. Defaults to None.
            name (str, optional): The name of the dataset. Defaults to None.
            time_resolution (int | None, optional): Time resolution in mins (If time resolution is smaller than one minute input the time in ms). Defaults to None.
        """

        self.name = name
        self.time_edges = {}
        self.max_id = -1
        self.next_id = []
        self._successor = {}
        self._time = {}
        self._predecessor = {}
        self.pos = {}
        self.time_id = {}
        if time_resolution is not None:
            self._time_resolution = time_resolution
        self.kdtrees = {}
        self.spatial_density = {}
        if file_type and file_format:
            if xml_attributes is None:
                self.xml_attributes = []
            else:
                self.xml_attributes = xml_attributes
            file_type = file_type.lower()
            if file_type == "tgmm":
                self.read_tgmm_xml(file_format, tb, te, z_mult)
            elif file_type == "mamut" or file_type == "trackmate":
                self.read_from_mamut_xml(file_format)
            elif file_type == "celegans":
                self.read_from_txt_for_celegans(file_format)
            elif file_type == "celegans_cao":
                self.read_from_txt_for_celegans_CAO(
                    file_format,
                    reorder=reorder,
                    shape=shape,
                    raw_size=raw_size,
                )
            elif file_type == "mastodon":
                if isinstance(file_format, list) and len(file_format) == 2:
                    self.read_from_mastodon_csv(file_format)
                else:
                    if isinstance(file_format, list):
                        file_format = file_format[0]
                    self.read_from_mastodon(file_format, name)
            elif file_type == "astec":
                self.read_from_ASTEC(file_format, eigen)
            elif file_type == "csv":
                self.read_from_csv(file_format, z_mult, link=1, delim=delim)
            elif file_type == "bao":
                self.read_C_elegans_bao(file_format)
            elif file_format and file_format.endswith(".lT"):
                with open(file_format, "br") as f:
                    tmp = pkl.load(f)
                    f.close()
                self.__dict__.update(tmp.__dict__)
            elif file_format is not None:
                self.read_from_binary(file_format)
            if self.name is None:
                try:
                    self.name = Path(file_format).stem
                except TypeError:
                    self.name = Path(file_format[0]).stem
        for k, v in self._successor.items():
            self._successor[k] = tuple(v)
        for k, v in self._predecessor.items():
            self._predecessor[k] = tuple(v)
        for node in set(self._successor).difference(self._predecessor):
            self._predecessor[node] = ()
        for node in set(self._predecessor).difference(self._successor):
            self._successor[node] = ()


class lineageTreeDicts(lineageTree):
    """Placeholder class to give a proof of concept of what the lineageTree init method would look like."""

    @classmethod
    def load(clf, fname: str):
        """
        Loading a lineage tree from a ".lT" file.

        Args:
            fname (str): path to and name of the file to read

        Returns:
            (lineageTree): loaded file
        """
        with open(fname, "br") as f:
            lT = pkl.load(f)
            f.close()
        if not hasattr(lT, "__version__") or Version(lT.__version__) < Version(
            "2.0.0"
        ):
            properties = {
                prop_name: prop
                for prop_name, prop in lT.__dict__.items()
                if isinstance(prop, dict)
                and prop_name
                not in ["successor", "predecessor", "time", "pos"]
                and set(prop).symmetric_difference(lT.nodes) == set()
            }
            lT = lineageTreeDicts(
                successor=lT._successor,
                time=lT._time,
                pos=lT.pos,
                name=lT.name if hasattr(lT, "name") else None,
                **properties,
            )
        if not hasattr(lT, "time_resolution"):
            lT.time_resolution = None

        return lT

    def __init__(
        self,
        *,
        successor: dict[int, tuple] = None,
        predecessor: dict[int, tuple] = None,
        time: dict[int, tuple] = None,
        starting_time: float = 0,
        pos: dict[int, Iterable] = None,
        name: str = None,
        **kwargs,
    ):
        """Creates a lineageTree object from minimal information, without reading from a file.
        Either `successor` or `predecessor` should be specified.

        Args:
            successor (dict[int, tuple]): Dictionary assigning nodes to their successors.
            predecessor (dict[int, tuple]): Dictionary assigning nodes to their predecessors.
            time (dict[int, float], optional): Dictionary assigning nodes to the time point they were recorded to.  Defaults to None, in which case all times are set to `starting_time`.
            starting_time (float, optional): Starting time of the lineage tree. Defaults to 0.
            pos (dict[int, Iterable], optional): Dictionary assigning nodes to their positions. Defaults to None.
            name (str, optional): Name of the lineage tree. Defaults to None.
            **kwargs: Supported keyword arguments are dictionaries assigning nodes to any custom property. The property must be specified for every node, and named differently from lineageTree's own attributes.
        """
        self.__version__ = importlib.metadata.version("LineageTree")

        self.name = name
        if successor is not None and predecessor is not None:
            raise ValueError(
                "You cannot have both successors and predecessors."
            )
        self._changed_roots = True
        self._changed_leaves = True
        if successor is not None:
            self._successor = successor
            self._predecessor = {}
            for pred, succ in successor.items():
                for s in succ:
                    if s in self._predecessor:
                        raise ValueError(
                            "Node can have at most one predecessor."
                        )
                    self._predecessor[s] = (pred,)
        elif predecessor is not None:
            self._successor = {}
            self._predecessor = predecessor
            for succ, pred in predecessor.items():
                if isinstance(pred, Iterable):
                    if 1 < len(pred):
                        raise ValueError(
                            "Node can have at most one predecessor."
                        )
                    pred = pred[0]
                successor.setdefault(pred, ())
                successor[pred] += (succ,)
        else:
            warnings.warn(
                "Both successor and predecessor attributes are empty.",
                stacklevel=2,
            )
        for root in set(self.successor).difference(self.predecessor):
            self._predecessor[root] = ()
        for leaf in set(self.predecessor).difference(self.successor):
            self._successor[leaf] = ()

        if pos is None:
            self.pos = {}
        else:
            if self.nodes.difference(pos) != set():
                raise ValueError("Please provide the position of all nodes.")
            self.pos = pos

        if time is None:
            self._time = {node: starting_time for node in self.roots}
            queue = list(self.roots)
            for node in queue:
                for succ in self._successor[node]:
                    self._time[succ] = self.time[node] + 1
                    queue.append(succ)
        else:
            self._time = time
            if self.nodes.difference(self.time) != set():
                raise ValueError("Please provide the time of all nodes.")
            if not all(
                self.time[node] < self.time[s]
                for node, succ in self._successor.items()
                for s in succ
            ):
                raise ValueError(
                    "Provided times are not strictly increasing. Setting times to default."
                )
        # custom properties
        for name, d in kwargs.items():
            if name in self.__dict__:
                warnings.warn(
                    f"Attribute name {name} is reserved.", stacklevel=2
                )
                continue
            if set(d) != self.nodes:
                warnings.warn(
                    f"Please specify {name} for all nodes.", stacklevel=2
                )
                continue
            setattr(self, name, d)
