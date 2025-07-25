#!python
# This file is subject to the terms and conditions defined in
# file 'LICENCE', which is part of this source code package.
# Author: Leo Guignard (leo.guignard...@AT@...gmail.com)

from __future__ import annotations

import importlib.metadata
import os
import pickle as pkl
import struct
import warnings
from collections.abc import Callable, Iterable, Sequence
from functools import partial, wraps
from itertools import combinations
from numbers import Number
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import svgwrite
from edist import uted
from matplotlib import colormaps
from matplotlib.collections import LineCollection
from packaging.version import Version
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.sparse import dok_array
from scipy.spatial import Delaunay, KDTree, distance

from . import __version__
from .tree_approximation import TreeApproximationTemplate, tree_style
from .utils import (
    convert_style_to_number,
    create_links_and_chains,
    hierarchical_pos,
)

if TYPE_CHECKING:
    from edist.alignment import Alignment


class dynamic_property(property):
    def __init__(
        self, fget=None, fset=None, fdel=None, doc=None, protected_name=None
    ):
        super().__init__(fget, fset, fdel, doc)
        self.protected_name = protected_name

    def __set_name__(self, owner, name):
        self.name = name
        if self.protected_name is None:
            self.protected_name = f"_{name}"
        if not hasattr(owner, "_protected_dynamic_properties"):
            owner._protected_dynamic_properties = []
        owner._protected_dynamic_properties.append(self.protected_name)
        if not hasattr(owner, "_dynamic_properties"):
            owner._dynamic_properties = []
        owner._dynamic_properties += [name, self.protected_name]
        setattr(owner, self.protected_name, None)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        instance._has_been_reset = False
        if getattr(instance, self.protected_name) is None:
            value = super().__get__(instance, owner)
            setattr(instance, self.protected_name, value)
            return value
        else:
            return getattr(instance, self.protected_name)


class lineageTree:
    norm_dict = {"max": max, "sum": sum, None: lambda x: 1}

    def modifier(wrapped_func):
        @wraps(wrapped_func)
        def raising_flag(self, *args, **kwargs):
            should_reset = (
                not hasattr(self, "_has_been_reset")
                or not self._has_been_reset
            )
            out_func = wrapped_func(self, *args, **kwargs)
            if should_reset:
                for prop in self._protected_dynamic_properties:
                    self.__dict__[prop] = None
                self._has_been_reset = True
            return out_func

        return raising_flag

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
        if isinstance(other, lineageTree):
            return (
                other._successor == self._successor
                and other._predecessor == self._predecessor
                and other._time == self._time
            )
        else:
            return False

    def get_next_id(self) -> int:
        """Computes the next authorized id and assign it.

        Returns
        -------
        int
            next authorized id
        """
        if not hasattr(self, "max_id") or (self.max_id == -1 and self.nodes):
            self.max_id = max(self.nodes) if len(self.nodes) else 0
        if not hasattr(self, "next_id") or self.next_id == []:
            self.max_id += 1
            return self.max_id
        else:
            return self.next_id.pop()

    ###TODO pos can be callable and stay motionless (copy the position of the succ node, use something like optical flow)
    @modifier
    def add_chain(
        self,
        node: int,
        length: int,
        downstream: bool,
        pos: Callable | None = None,
    ) -> int:
        """Adds a chain of specific length to a node either as a successor or as a predecessor.
        If it is placed on top of a tree all the nodes will move timepoints #length down.

        Parameters
        ----------
        node : int
            Id of the successor (predecessor if `downstream==False`)
        length : int
            The length of the new chain.
        downstream : bool, default=True
            If `True` will create a chain that goes forwards in time otherwise backwards.
        pos : np.ndarray, optional
            The new position of the chain. Defaults to None.

        Returns
        -------
        int
            Id of the first node of the sublineage.
        """
        if length == 0:
            return node
        if length < 1:
            raise ValueError("Length cannot be <1")
        if downstream:
            for _ in range(int(length)):
                old_node = node
                node = self._add_node(pred=[old_node])
                self._time[node] = self._time[old_node] + 1
        else:
            if self._predecessor[node]:
                raise Warning("The node already has a predecessor.")
            if self._time[node] - length < self.t_b:
                raise Warning(
                    "A node cannot created outside the lower bound of the dataset. (It is possible to change it by lT.t_b = int(...))"
                )
            for _ in range(int(length)):
                old_node = node
                node = self._add_node(succ=[old_node])
                self._time[node] = self._time[old_node] - 1
        return node

    @modifier
    def add_root(self, t: int, pos: list | None = None) -> int:
        """Adds a root to a specific timepoint.

        Parameters
        ----------
        t :int
            The timepoint the node is going to be added.
        pos : list
            The position of the new node.
        Returns
        -------
        int
            The id of the new root.
        """
        C_next = self.get_next_id()
        self._successor[C_next] = ()
        self._predecessor[C_next] = ()
        self._time[C_next] = t
        self.pos[C_next] = pos if isinstance(pos, list) else []
        self._changed_roots = True
        return C_next

    def _add_node(
        self,
        succ: list | None = None,
        pred: list | None = None,
        pos: np.ndarray | None = None,
        nid: int | None = None,
    ) -> int:
        """Adds a node to the LineageTree object that is either a successor or a predecessor of another node.
        Does not handle time! You cannot enter both a successor and a predecessor.

        Parameters
        ----------
        succ : list
            list of ids of the nodes the new node is a successor to
        pred : list
            list of ids of the nodes the new node is a predecessor to
        pos : np.ndarray, optional
            position of the new node
        nid : int, optional
            id value of the new node, to be used carefully,
            if None is provided the new id is automatically computed.

        Returns
        -------
        int
            id of the new node.
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

    @modifier
    def remove_nodes(self, group: int | set | list) -> None:
        """Removes a group of nodes from the LineageTree

        Parameters
        ----------
        group : set of int or list of int or int
            One or more nodes that are to be removed.
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

    @property
    def successor(self) -> MappingProxyType[int, tuple[int]]:
        """The successor of the tree."""
        if not hasattr(self, "_protected_successor"):
            self._protected_successor = MappingProxyType(self._successor)
        return self._protected_successor

    @property
    def predecessor(self) -> MappingProxyType[int, tuple[int]]:
        """The predecessor of the tree."""
        if not hasattr(self, "_protected_predecessor"):
            self._protected_predecessor = MappingProxyType(self._predecessor)
        return self._protected_predecessor

    @property
    def time(self) -> MappingProxyType[int, int]:
        """The time of the tree."""
        if not hasattr(self, "_protected_time"):
            self._protected_time = MappingProxyType(self._time)
        return self._protected_time

    @dynamic_property
    def t_b(self) -> int:
        """The first timepoint of the tree."""
        return min(self._time.values())

    @dynamic_property
    def t_e(self) -> int:
        """The last timepoint of the tree."""
        return max(self._time.values())

    @dynamic_property
    def nodes(self) -> frozenset[int]:
        """Nodes of the tree"""
        return frozenset(self._successor.keys())

    @dynamic_property
    def number_of_nodes(self) -> int:
        return len(self.nodes)

    @dynamic_property
    def depth(self) -> dict[int, int]:
        """The depth of each node in the tree."""
        _depth = {}
        for leaf in self.leaves:
            _depth[leaf] = 1
            while leaf in self._predecessor and self._predecessor[leaf]:
                parent = self._predecessor[leaf][0]
                current_depth = _depth.get(parent, 0)
                _depth[parent] = max(_depth[leaf] + 1, current_depth)
                leaf = parent
        for root in self.roots - set(_depth):
            _depth[root] = 1
        return _depth

    @dynamic_property
    def roots(self) -> frozenset[int]:
        """Set of roots of the tree"""
        return frozenset({s for s, p in self._predecessor.items() if p == ()})

    @dynamic_property
    def leaves(self) -> frozenset[int]:
        """Set of leaves"""
        return frozenset({p for p, s in self._successor.items() if s == ()})

    @dynamic_property
    def edges(self) -> tuple[tuple[int, int]]:
        """Set of edges"""
        return tuple((p, si) for p, s in self._successor.items() for si in s)

    @property
    def labels(self) -> dict[int, str]:
        """The labels of the nodes."""
        if not hasattr(self, "_labels"):
            if hasattr(self, "node_name"):
                self._labels = {
                    i: self.node_name.get(i, "Unlabeled") for i in self.roots
                }
            else:
                self._labels = {
                    root: "Unlabeled"
                    for root in self.roots
                    for leaf in self.find_leaves(root)
                    if abs(self._time[leaf] - self._time[root])
                    >= abs(self.t_e - self.t_b) / 4
                }
        return self._labels

    @property
    def time_resolution(self) -> float:
        if not hasattr(self, "_time_resolution"):
            self._time_resolution = 0
        return self._time_resolution / 10

    @time_resolution.setter
    def time_resolution(self, time_resolution) -> None:
        if time_resolution is not None and time_resolution > 0:
            self._time_resolution = int(time_resolution * 10)
        else:
            warnings.warn("Time resolution set to default 0", stacklevel=2)
            self._time_resolution = 0

    def __setstate__(self, state):
        if "_successor" not in state:
            state["_successor"] = state["successor"]
        if "_predecessor" not in state:
            state["_predecessor"] = state["predecessor"]
        if "_time" not in state:
            state["_time"] = state["time"]
        self.__dict__.update(state)

    def _get_height(self, c: int, done: dict) -> float:
        """Recursively computes the height of a node within a tree times a space factor.
        This function is specific to the function write_to_svg.

        Parameters
        ----------
        c : int
            id of a node in a lineage tree from which the height will be computed from
        done : dict mapping int to list of two int
            a dictionary that maps a node id to its vertical and horizontal position

        Returns
        -------
        float
            the height of the node `c`
        """
        if c in done:
            return done[c][0]
        else:
            P = np.mean(
                [self._get_height(di, done) for di in self._successor[c]]
            )
            done[c] = [P, self.vert_space_factor * self._time[c]]
            return P

    def write_to_svg(
        self,
        file_name: str,
        roots: list | None = None,
        draw_nodes: bool = True,
        draw_edges: bool = True,
        order_key: Callable | None = None,
        vert_space_factor: float = 0.5,
        horizontal_space: float = 1,
        node_size: Callable | str | None = None,
        stroke_width: Callable | None = None,
        factor: float = 1.0,
        node_color: Callable | str | None = None,
        stroke_color: Callable | None = None,
        positions: dict | None = None,
        node_color_map: Callable | str | None = None,
    ) -> None:
        """Writes the lineage tree to an SVG file.
        Node and edges coloring and size can be provided.

        Parameters
        ----------
        file_name : str
            filesystem filename valid for `open()`
        roots : list of int, defaults to `self.roots`
            list of node ids to be drawn. If `None` or not provided all the nodes will be drawn. Default `None`
        draw_nodes : bool, default True
            wether to print the nodes or not
        draw_edges : bool, default True
            wether to print the edges or not
        order_key : Callable, optional
            function that would work for the attribute `key=` for the `sort`/`sorted` function
        vert_space_factor : float, default=0.5
            the vertical position of a node is its time. `vert_space_factor` is a
            multiplier to space more or less nodes in time
        horizontal_space : float, default=1
            space between two consecutive nodes
        node_size : Callable or str, optional
            a function that maps a node id to a `float` value that will determine the
            radius of the node. The default function return the constant value `vertical_space_factor/2.1`
            If a string is given instead and it is a property of the tree,
            the the size will be mapped according to the property
        stroke_width : Callable, optional
            a function that maps a node id to a `float` value that will determine the
            width of the daughter edge.  The default function return the constant value `vertical_space_factor/2.1`
        factor : float, default=1.0
            scaling factor for nodes positions, default 1
        node_color : Callable or str, optional
            a function that maps a node id to a triplet between 0 and 255.
            The triplet will determine the color of the node. If a string is given instead and it is a property
            of the tree, the the color will be mapped according to the property
        node_color_map : Callable or str, optional
            the name of the colormap to use to color the nodes, or a colormap function
        stroke_color : Callable, optional
            a function that maps a node id to a triplet between 0 and 255.
            The triplet will determine the color of the stroke of the inward edge.
        positions : dict mapping int to list of two float, optional
            dictionary that maps a node id to a 2D position.
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
                roots = [node for node in roots if self.image_label[node] != 1]

        if node_size is None:

            def node_size(x):
                return vert_space_factor / 2.1

        else:
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
            from matplotlib import colormaps

            if node_color_map in colormaps:
                cmap = colormaps[node_color_map]
            else:
                cmap = colormaps["viridis"]
            values = np.array(
                [self._successor[node_color][c] for c in self.nodes]
            )
            normed_vals = normalize_values(values, self.nodes, 1, 0, 1)

            def node_color(x):
                return [k * 255 for k in cmap(normed_vals(x))[:-1]]

        coloring_edges = stroke_color is not None
        if not coloring_edges:

            def stroke_color(x):
                return 0, 0, 0

        elif isinstance(stroke_color, str) and stroke_color in self.__dict__:
            from matplotlib import colormaps

            if node_color_map in colormaps:
                cmap = colormaps[node_color_map]
            else:
                cmap = colormaps["viridis"]
            values = np.array(
                [self._successor[stroke_color][c] for c in self.nodes]
            )
            normed_vals = normalize_values(values, self.nodes, 1, 0, 1)

            def stroke_color(x):
                return [k * 255 for k in cmap(normed_vals(x))[:-1]]

        prev_x = 0
        self.vert_space_factor = vert_space_factor
        if order_key is not None:
            roots.sort(key=order_key)
        treated_nodes = []

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
                treated_nodes += [curr]
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
                    self.vert_space_factor * self._time[leave],
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
            to_do = set(treated_nodes)
            while len(to_do) > 0:
                curr = to_do.pop()
                c_chain = self.get_chain_of_node(curr)
                x1, y1 = positions[c_chain[0]]
                x2, y2 = positions[c_chain[-1]]
                dwg.add(
                    dwg.line(
                        (factor * x1, factor * y1),
                        (factor * x2, factor * y2),
                        stroke=svgwrite.rgb(0, 0, 0),
                    )
                )
                for si in self._successor[c_chain[-1]]:
                    x3, y3 = positions[si]
                    dwg.add(
                        dwg.line(
                            (factor * x2, factor * y2),
                            (factor * x3, factor * y3),
                            stroke=svgwrite.rgb(0, 0, 0),
                        )
                    )
                to_do.difference_update(c_chain)
        else:
            for c in treated_nodes:
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
            for c in treated_nodes:
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
        nodes_to_use: list[int] | None = None,
        temporal: bool = True,
        spatial: str | None = None,
        write_layout: bool = True,
        node_properties: dict | None = None,
        Names: bool = False,
    ) -> None:
        """Write a lineage tree into an understable tulip file.

        Parameters
        ----------
        fname : str
            path to the tulip file to create
        t_min : int, default=-1
            minimum time to consider
        t_max : int, default=np.inf
            maximum time to consider
        nodes_to_use : list of int, optional
            list of nodes to show in the graph,
            if `None` then self.nodes is used
            (taking into account `t_min` and `t_max`)
        temporal : bool, default=True
            True if the temporal links should be printed
        spatial : str, optional
            Build spatial edges from a spatial neighbourhood graph.
            The graph has to be computed before running this function
            'ball': neighbours at a given distance,
            'kn': k-nearest neighbours,
            'GG': gabriel graph,
            None: no spatial edges are writen.
            Default None
        write_layout : bool, default=True
            write the spatial position as layout if True
            do not write spatial position otherwise
        node_properties : dict mapping str to list of dict of properties and its default value, optional
            a dictionary of properties to write
            To a key representing the name of the property is
            paired a dictionary that maps a node id to a property
            and a default value for this property
        Names : bool, default=True
            Only works with ASTEC outputs, True to sort the nodes by their names
        """

        def format_names(names_which_matter):
            """Return an ensured formated node names"""
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
                        n for n in self.nodes if t_min < self._time[n] <= t_max
                    ]
                    edges_to_use = []
                    if temporal:
                        edges_to_use += [
                            e
                            for e in self.edges
                            if t_min < self._time[e[0]] < t_max
                        ]
                    if spatial:
                        edges_to_use += [
                            e
                            for e in s_edges
                            if t_min < self._time[e[0]] < t_max
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
                        e for e in s_edges if t_min < self._time[e[0]] < t_max
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
                        or self._time[k] == t_min + 1
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
                    "\t(node " + str(n) + ' "' + str(self._time[n]) + '")\n'
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

    def to_binary(
        self, fname: str, starting_points: list[int] | None = None
    ) -> None:
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

        Parameters
        ----------
        fname : str
            name of the binary file
        starting_points : list of int, optional
            list of the roots to be written.
            If `None`, all roots are written, default value, None
        """
        if starting_points is None:
            starting_points = list(self.roots)
        number_sequence = [-1]
        pos_sequence = []
        time_sequence = []
        for c in starting_points:
            time_sequence.append(self._time.get(c, 0))
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
            time_sequence.append(self._time.get(c, 0))
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
        """Write a lineage tree on disk as an .lT file.

        Parameters
        ----------
        fname : str
            path to and name of the file to save
        """
        if os.path.splitext(fname)[-1].upper() != ".LT":
            fname = os.path.extsep.join((fname, "lT"))
        if hasattr(self, "_protected_predecessor"):
            del self._protected_predecessor
        if hasattr(self, "_protected_successor"):
            del self._protected_successor
        if hasattr(self, "_protected_time"):
            del self._protected_time
        with open(fname, "bw") as f:
            pkl.dump(self, f)
            f.close()

    @classmethod
    def load(clf, fname: str):
        """Loading a lineage tree from a '.lT' file.

        Parameters
        ----------
        fname : str
            path to and name of the file to read

        Returns
        -------
        LineageTree
            loaded file
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
                if (isinstance(prop, dict) or prop_name == "_time_resolution")
                and prop_name
                not in [
                    "successor",
                    "predecessor",
                    "time",
                    "_successor",
                    "_predecessor",
                    "_time",
                    "pos",
                    "labels",
                ]
                + lineageTree._dynamic_properties
                + lineageTree._protected_dynamic_properties
            }
            lT = lineageTree(
                successor=lT._successor,
                time=lT._time,
                pos=lT.pos,
                name=lT.name if hasattr(lT, "name") else None,
                **properties,
            )
        if not hasattr(lT, "time_resolution"):
            lT.time_resolution = 1

        return lT

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

    @dynamic_property
    def all_chains(self) -> list[list[int]]:
        """List of all chains in the tree, ordered in depth-first search."""
        return self._compute_all_chains()

    @dynamic_property
    def time_nodes(self):
        _time_nodes = {}
        for c, t in self._time.items():
            _time_nodes.setdefault(t, set()).add(c)
        return _time_nodes

    def m(self, i, j):
        if (i, j) not in self._tmp_parenting:
            if i == j:  # the distance to the node itself is 0
                self._tmp_parenting[(i, j)] = 0
                self._parenting[i, j] = self._tmp_parenting[(i, j)]
            elif not self._predecessor[
                j
            ]:  # j and i are note connected so the distance if inf
                self._tmp_parenting[(i, j)] = np.inf
            else:  # the distance between i and j is the distance between i and pred(j) + 1
                self._tmp_parenting[(i, j)] = (
                    self.m(i, self._predecessor[j][0]) + 1
                )
                self._parenting[i, j] = self._tmp_parenting[(i, j)]
                self._parenting[j, i] = -self._tmp_parenting[(i, j)]
        return self._tmp_parenting[(i, j)]

    @property
    def parenting(self):
        if not hasattr(self, "_parenting"):
            self._parenting = dok_array((max(self.nodes) + 1,) * 2)
            self._tmp_parenting = {}
            for i, j in combinations(self.nodes, 2):
                if self._time[j] < self.time[i]:
                    i, j = j, i
                self._tmp_parenting[(i, j)] = self.m(i, j)
            del self._tmp_parenting
        return self._parenting

    def get_idx3d(self, t: int) -> tuple[KDTree, np.ndarray]:
        """Get a 3d kdtree for the dataset at time `t`.
        The  kdtree is stored in `self.kdtrees[t]` and returned.
        The correspondancy list is also returned.

        Parameters
        ----------
        t : int
            time

        Returns
        -------
        KDTree
            The KDTree corresponding to the lineage tree at time `t`
        np.ndarray
            The correspondancy list in the KDTree.
            If the query in the kdtree gives you the value `i`,
            then it corresponds to the id in the tree `to_check_self[i]`
        """
        to_check_self = list(self.nodes_at_t(t=t))

        if not hasattr(self, "kdtrees"):
            self.kdtrees = {}

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
        """Build the Gabriel graph of the given graph for time point `t`.
        The Garbiel graph is then stored in `self.Gabriel_graph` and returned.

        .. warning:: the graph is not recomputed if already computed, even if the point cloud has changed

        Parameters
        ----------
        t : int
            time

        Returns
        -------
        dict of int to set of int
            A dictionary that maps a node to the set of its neighbors
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

    def compute_spatial_density(
        self, t_b: int | None = None, t_e: int | None = None, th: float = 50
    ) -> dict[int, float]:
        """Computes the spatial density of nodes between `t_b` and `t_e`.
        The results is stored in `self.spatial_density` and returned.

        Parameters
        ----------
        t_b : int, optional
            starting time to look at, default first time point
        t_e : int, optional
            ending time to look at, default last time point
        th : float, default=50
            size of the neighbourhood

        Returns
        -------
        dict mapping int to float
            dictionary that maps a node id to its spatial density
        """
        if not hasattr(self, "spatial_density"):
            self.spatial_density = {}
        s_vol = 4 / 3.0 * np.pi * th**3
        if t_b is None:
            t_b = self.t_b
        if t_e is None:
            t_e = self.t_e
        time_range = set(range(t_b, t_e)).intersection(self._time.values())
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

        Parameters
        ----------
        k : float
            number of nearest neighours

        Returns
        -------
        dict mapping int to set of int
            dictionary that maps
            a node id to its `k` nearest neighbors
        """
        self.kn_graph = {}
        for t in set(self._time.values()):
            nodes = self.nodes_at_t(t)
            if 1 < len(nodes):
                use_k = k if k < len(nodes) else len(nodes)
                idx3d, nodes = self.get_idx3d(t)
                pos = [self.pos[c] for c in nodes]
                _, neighbs = idx3d.query(pos, use_k)
                out = dict(
                    zip(
                        nodes,
                        map(set, nodes[neighbs]),
                        strict=True,
                    )
                )
                self.kn_graph.update(out)
            else:
                n = nodes.pop
                self.kn_graph.update({n: {n}})
        return self.kn_graph

    def compute_spatial_edges(self, th: int = 50) -> dict[int, set[int]]:
        """Computes the neighbors at a distance `th`
        Writes the output in the attribute `th_edge`
        and returns it.

        Parameters
        ----------
        th : float, default=50
            distance to consider neighbors

        Returns
        -------
        dict mapping int to set of int
            dictionary that maps a node id to its neighbors at a distance `th`
        """
        self.th_edges = {}
        for t in set(self._time.values()):
            nodes = self.nodes_at_t(t)
            idx3d, nodes = self.get_idx3d(t)
            neighbs = idx3d.query_ball_tree(idx3d, th)
            out = dict(
                zip(nodes, [set(nodes[ni]) for ni in neighbs], strict=True)
            )
            self.th_edges.update(
                {k: v.difference([k]) for k, v in out.items()}
            )
        return self.th_edges

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
        Alignment
        | tuple[TreeApproximationTemplate, TreeApproximationTemplate],
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
            raise Warning(
                "Select a viable normalization method (max, sum, None)"
            )
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

    def get_subtree(self, node_list: set[int]) -> lineageTree:
        new_successors = {
            n: tuple(vi for vi in self.successor[n] if vi in node_list)
            for n in node_list
        }
        return lineageTree(
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
        """Create a lineageTree object from minimal information, without reading from a file.
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
            The property must be specified for every node, and named differently from lineageTree's own attributes.
        """
        self.__version__ = __version__
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
