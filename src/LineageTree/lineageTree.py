#!python
# This file is subject to the terms and conditions defined in
# file 'LICENCE', which is part of this source code package.
# Author: Leo Guignard (leo.guignard...@AT@...gmail.com)

from turtle import color
from scipy.spatial import cKDTree as KDTree
import os
import xml.etree.ElementTree as ET
from copy import copy
import numpy as np
from multiprocessing import Pool
from scipy.spatial import Delaunay
from itertools import combinations
from numbers import Number
import struct
from scipy.spatial.distance import cdist


class lineageTree(object):
    def get_next_id(self):
        """Computes the next authorized id.

        Returns:
            int: next authorized id
        """
        if self.next_id == []:
            self.max_id += 1
            return self.max_id
        else:
            return self.next_id.pop()

    def add_node(self, t=None, succ=None, pos=None, id=None, reverse=False):
        """Adds a node to the lineageTree and update it accordingly.

        Args:
            t (int): int, time to which to add the node
            succ (int): id of the node the new node is a successor to
            pos ([float, ]): list of three floats representing the 3D spatial position of the node
            id (int): id value of the new node, to be used carefully,
                if None is provided the new id is automatically computed.
            reverse (bool): True if in this lineageTree the predecessors are the successors and reciprocally.
                This is there for bacward compatibility, should be left at False.
        Returns:
            int: id of the new node.
        """
        if id is None:
            C_next = self.get_next_id()
        else:
            C_next = id
        self.time_nodes.setdefault(t, []).append(C_next)
        if not succ is None and not reverse:
            self.successor.setdefault(succ, []).append(C_next)
            self.predecessor.setdefault(C_next, []).append(succ)
            self.edges.add((succ, C_next))
        elif not succ is None:
            self.predecessor.setdefault(succ, []).append(C_next)
            self.successor.setdefault(C_next, []).append(succ)
            self.edges.add((C_next, succ))
        self.nodes.add(C_next)
        self.pos[C_next] = pos
        self.progeny[C_next] = 0
        self.time[C_next] = t
        return C_next

    def remove_node(self, c):
        """Removes a node and update the lineageTree accordingly

        Args:
            c (int): id of the node to remove
        """
        self.nodes.remove(c)
        self.time_nodes[self.time[c]].remove(c)
        # self.time_nodes.pop(c, 0)
        pos = self.pos.pop(c, 0)
        e_to_remove = [e for e in self.edges if c in e]
        for e in e_to_remove:
            self.edges.remove(e)
        if c in self.roots:
            self.roots.remove(c)
        succ = self.successor.pop(c, [])
        s_to_remove = [s for s, ci in self.successor.items() if c in ci]
        for s in s_to_remove:
            self.successor[s].remove(c)

        pred = self.predecessor.pop(c, [])
        p_to_remove = [s for s, ci in self.predecessor.items() if ci == c]
        for s in p_to_remove:
            self.predecessor[s].remove(c)

        self.time.pop(c, 0)
        self.spatial_density.pop(c, 0)

        self.next_id.append(c)
        return e_to_remove, succ, s_to_remove, pred, p_to_remove, pos

    def fuse_nodes(self, c1, c2):
        """Fuses together two nodes that belong to the same time point
        and update the lineageTree accordingly.

        Args:
            c1 (int): id of the first node to fuse
            c2 (int): id of the second node to fuse
        """
        (
            e_to_remove,
            succ,
            s_to_remove,
            pred,
            p_to_remove,
            c2_pos,
        ) = self.remove_node(c2)
        for e in e_to_remove:
            new_e = [c1] + [other_c for other_c in e if e != c2]
            self.edges.add(new_e)

        self.successor.setdefault(c1, []).extend(succ)
        self.predecessor.setdefault(c1, []).extend(pred)

        for s in s_to_remove:
            self.successor[s].append(c1)

        for p in p_to_remove:
            self.predecessor[p].append(c1)

        self.pos[c1] = np.mean([self.pos[c1], c2_pos], axis=0)
        self.progeny[c1] += 1

    def _write_header_am(self, f, nb_points, length):
        """Header for Amira .am files"""
        f.write("# AmiraMesh 3D ASCII 2.0\n")
        f.write("define VERTEX %d\n" % (nb_points * 2))
        f.write("define EDGE %d\n" % nb_points)
        f.write("define POINT %d\n" % ((length) * nb_points))
        f.write("Parameters {\n")
        f.write('\tContentType "HxSpatialGraph"\n')
        f.write("}\n")

        f.write("VERTEX { float[3] VertexCoordinates } @1\n")
        f.write("EDGE { int[2] EdgeConnectivity } @2\n")
        f.write("EDGE { int NumEdgePoints } @3\n")
        f.write("POINT { float[3] EdgePointCoordinates } @4\n")
        f.write("VERTEX { float Vcolor } @5\n")
        f.write("VERTEX { int Vbool } @6\n")
        f.write("EDGE { float Ecolor } @7\n")
        f.write("VERTEX { int Vbool2 } @8\n")

    def write_to_am(
        self,
        path_format,
        t_b=None,
        t_e=None,
        length=5,
        manual_labels=None,
        default_label=5,
        new_pos=None,
    ):
        """Writes a lineageTree into an Amira readable data (.am format).

        Args:
            path_format (str): path to the output. It should contain 1 %03d where the time step will be entered
            t_b (int): first time point to write (if None, min(LT.to_take_time) is taken)
            t_e (int): last time point to write (if None, max(LT.to_take_time) is taken)
                note, if there is no 'to_take_time' attribute, self.time_nodes
                is considered instead (historical)
            length (int): length of the track to print (how many time before).
            manual_labels ({id: label, }): dictionary that maps cell ids to
            default_label (int): default value for the manual label
            new_pos ({id: [x, y, z]}): dictionary that maps a 3D position to a cell ID.
                if new_pos == None (default) then self.pos is considered.
        """
        if not hasattr(self, "to_take_time"):
            self.to_take_time = self.time_nodes
        if t_b is None:
            t_b = min(self.to_take_time.keys())
        if t_e is None:
            t_e = max(self.to_take_time.keys())
        if new_pos is None:
            new_pos = self.pos

        if manual_labels is None:
            manual_labels = {}
        for t in range(t_b, t_e + 1):
            f = open(path_format % t, "w")
            nb_points = len(self.to_take_time[t])
            self._write_header_am(f, nb_points, length)
            points_v = {}
            for C in self.to_take_time[t]:
                C_tmp = C
                positions = []
                for i in range(length):
                    C_tmp = self.predecessor.get(C_tmp, [C_tmp])[0]
                    positions.append(new_pos[C_tmp])
                points_v[C] = positions

            f.write("@1\n")
            for i, C in enumerate(self.to_take_time[t]):
                f.write("%f %f %f\n" % tuple(points_v[C][0]))
                f.write("%f %f %f\n" % tuple(points_v[C][-1]))

            f.write("@2\n")
            for i, C in enumerate(self.to_take_time[t]):
                f.write("%d %d\n" % (2 * i, 2 * i + 1))

            f.write("@3\n")
            for i, C in enumerate(self.to_take_time[t]):
                f.write("%d\n" % (length))

            f.write("@4\n")
            tmp_velocity = {}
            for i, C in enumerate(self.to_take_time[t]):
                for p in points_v[C]:
                    f.write("%f %f %f\n" % tuple(p))

            f.write("@5\n")
            for i, C in enumerate(self.to_take_time[t]):
                f.write("%f\n" % (manual_labels.get(C, default_label)))
                f.write("%f\n" % (0))

            f.write("@6\n")
            for i, C in enumerate(self.to_take_time[t]):
                f.write(
                    "%d\n"
                    % (
                        int(
                            manual_labels.get(C, default_label)
                            != default_label
                        )
                    )
                )
                f.write("%d\n" % (0))

            f.write("@7\n")
            for i, C in enumerate(self.to_take_time[t]):
                f.write(
                    "%f\n" % (np.linalg.norm(points_v[C][0] - points_v[C][-1]))
                )

            f.write("@8\n")
            for i, C in enumerate(self.to_take_time[t]):
                f.write("%d\n" % (1))
                f.write("%d\n" % (0))
            f.close()

    def _get_height(self, c, done):
        """Recursively computes the height of a cell within a tree * a space factor.
        This function is specific to the function write_to_svg.

        Args:
            c (int): id of a cell in a lineage tree from which the height will be computed from
            done ({int: [int, int]}): a dictionary that maps a cell id to its vertical and horizontal position
        Returns:
            float:
        """
        if c in done:
            return done[c][0]
        else:
            P = np.mean(
                [self._get_height(di, done) for di in self.successor[c]]
            )
            done[c] = [P, self.vert_space_factor * self.time[c]]
            return P

    def write_to_svg(
        self,
        file_name,
        roots=None,
        draw_nodes=True,
        draw_edges=True,
        order_key=None,
        vert_space_factor=0.5,
        horizontal_space=1,
        node_size=None,
        stroke_width=None,
        factor=1.0,
        node_color=None,
        stroke_color=None,
        positions=None,
        node_color_map=None,
    ):
        """Writes the lineage tree to an SVG file.
        Node and edges coloring and size can be provided.

        Args:
            file_name: str, filesystem filename valid for `open()`
            roots: [int, ...], list of node ids to be drawn. If `None` all the nodes will be drawn. Default `None`
            draw_nodes: bool, wether to print the nodes or not, default `True`
            draw_edges: bool, wether to print the edges or not, default `True`
            order_key: function that would work for the attribute `key=` for the `sort`/`sorted` function
            vert_space_factor: float, the vertical position of a node is its time. `vert_space_factor` is a
                               multiplier to space more or less nodes in time
            horizontal_space: float, space between two consecutive nodes
            node_size: func | str, a function that maps a node id to a `float` value that will determine the
                       radius of the node. The default function return the constant value `vertical_space_factor/2.1`
                       If a string is given instead and it is a property of the tree,
                       the the size will be mapped according to the property
            stroke_width: func, a function that maps a node id to a `float` value that will determine the
                          width of the daughter edge.  The default function return the constant value `vertical_space_factor/2.1`
            factor: float, scaling factor for nodes positions, default 1
            node_color: func | str, a function that maps a node id to a triplet between 0 and 255.
                        The triplet will determine the color of the node. If a string is given instead and it is a property
                        of the tree, the the color will be mapped according to the property
            node_color_map: str, the name of the colormap to use to color the nodes
            stroke_color: func, a function that maps a node id to a triplet between 0 and 255.
                          The triplet will determine the color of the stroke of the inward edge.
            positions: {int: [float, float], ...}, dictionary that maps a node id to a 2D position.
                       Default `None`. If provided it will be used to position the nodes.
        """
        import svgwrite

        def normalize_values(v, nodes, range, shift, mult):
            min_ = np.percentile(v, 1)
            max_ = np.percentile(v, 99)
            values = range * ((v - min_) / (max_ - min_)) + shift
            values_dict_nodes = dict(zip(nodes, values))
            return lambda x: values_dict_nodes[x] * mult

        if roots is None:
            roots = list(set(self.successor).difference(self.predecessor))

        if node_size is None:
            node_size = lambda x: vert_space_factor / 2.1
        elif isinstance(node_size, str) and node_size in self.__dict__:
            values = np.array([self[node_size][c] for c in self.nodes])
            node_size = normalize_values(
                values, self.nodes, 0.5, 0.5, vert_space_factor / 2.1
            )
        if stroke_width is None:
            stroke_width = lambda x: vert_space_factor / 2.2
        if node_color is None:
            node_color = lambda x: (0, 0, 0)
        elif isinstance(node_color, str) and node_color in self.__dict__:
            if isinstance(node_color_map, str):
                from matplotlib import colormaps

                if node_color_map in colormaps:
                    node_color_map = colormaps[node_color_map]
                else:
                    node_color_map = colormaps["viridis"]
            values = np.array([self[node_color][c] for c in self.nodes])
            normed_vals = normalize_values(values, self.nodes, 1, 0, 1)
            node_color = lambda x: [
                k * 255 for k in node_color_map(normed_vals(x))[:-1]
            ]
        coloring_edges = not stroke_color is None
        if not coloring_edges:
            stroke_color = lambda x: (0, 0, 0)
        elif isinstance(stroke_color, str) and stroke_color in self.__dict__:
            if isinstance(node_color_map, str):
                from matplotlib import colormaps

                if node_color_map in colormaps:
                    node_color_map = colormaps[node_color_map]
                else:
                    node_color_map = colormaps["viridis"]
            values = np.array([self[stroke_color][c] for c in self.nodes])
            normed_vals = normalize_values(values, self.nodes, 1, 0, 1)
            stroke_color = lambda x: [
                k * 255 for k in node_color_map(normed_vals(x))[:-1]
            ]
        prev_x = 0
        self.vert_space_factor = vert_space_factor
        if order_key is not None:
            roots.sort(key=order_key)
        treated_cells = []

        pos_given = not positions is None
        if not pos_given:
            positions = dict(
                zip(
                    self.nodes,
                    [
                        [0.0, 0.0],
                    ]
                    * len(self.nodes),
                )
            )
        for i, r in enumerate(roots):
            r_leaves = []
            to_do = [r]
            while len(to_do) != 0:
                curr = to_do.pop(0)
                treated_cells += [curr]
                if curr in self.successor:
                    if order_key is not None:
                        to_do += sorted(self.successor[curr], key=order_key)
                    else:
                        to_do += self.successor[curr]
                else:
                    r_leaves += [curr]
            r_pos = {
                l: [
                    prev_x + horizontal_space * (1 + j),
                    self.vert_space_factor * self.time[l],
                ]
                for j, l in enumerate(r_leaves)
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
            while 0 < len(to_do):
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
                for si in self.successor.get(c_cycle[-1], []):
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
                for si in self.successor.get(c, []):
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

    def to_treex(self):
        """Convert the lineage tree into a treex file."""
        from treex.tree import Tree

        # id_to_tree = {id: Tree() for id in self.nodes}

        # for id_m, ids_d in self.successor.items():
        #     for id_d in ids_d:
        #         id_to_tree[id_m].add_subtree(id_to_tree[id_d])
        # roots = [id_to_tree[id] for id in set(self.successor).difference(self.predecessor)]

        roots = set(self.successor).difference(self.predecessor)
        id_to_tree = {}
        root_trees = []
        for r in roots:
            to_do = [r]
            while 0 < len(to_do):
                curr = to_do.pop()
                cycle = self.get_cycle(curr)
                lifetime = len(cycle)
                cell = Tree()
                cell.add_attribute_to_id("len", lifetime)
                if cycle[0] in self.predecessor:
                    id_to_tree[self.predecessor[cycle[0]][0]].add_subtree(cell)
                else:
                    root_trees.append(cell)
                id_to_tree[cycle[-1]] = cell
                to_do.extend(self.successor.get(cycle[-1], []))

        return roots

    def to_tlp(
        self,
        fname,
        t_min=-1,
        t_max=np.inf,
        nodes_to_use=None,
        temporal=True,
        spatial=None,
        write_layout=True,
        node_properties=None,
        Names=False,
    ):
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
                try:
                    val = v.split(".")[1][:-1]
                    tmp[k] = (
                        v.split(".")[0][0]
                        + "%02d" % int(v.split(".")[0][1:])
                        + "."
                        + "%04d" % int(v.split(".")[1][:-1])
                        + v.split(".")[1][-1]
                    )
                except Exception as e:
                    print(v)
                    exit()
            return tmp

        def spatial_adjlist_to_set(s_g):
            s_edges = set()
            for t, gg in s_g.items():
                for c, N in gg.items():
                    s_edges.update([tuple(sorted([c, ni])) for ni in N])
            return s_edges

        f = open(fname, "w")
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
            if t_max != np.inf or -1 < t_min:
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
                        (e, ei)
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
            if temporal:
                edges_to_use += [
                    e
                    for e in self.edges
                    if e[0] in nodes_to_use and e[1] in nodes_to_use
                ]
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
                        self.successor.get(
                            self.predecessor.get(k, [-1])[0], []
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
                "(edge " + str(i) + " " + str(e[0]) + " " + str(e[1]) + ")\n"
            )

        f.write('(property 0 int "time"\n')
        f.write('\t(default "0" "0")\n')
        for n in nodes_to_use:
            f.write(
                "\t(node " + str(n) + str(' "') + str(self.time[n]) + '")\n'
            )
        f.write(")\n")

        if write_layout:
            f.write('(property 0 layout "viewLayout"\n')
            f.write('\t(default "(0, 0, 0)" "()")\n')
            for n in nodes_to_use:
                f.write(
                    "\t(node "
                    + str(n)
                    + str(' "')
                    + str(tuple(self.pos[n]))
                    + '")\n'
                )
            f.write(")\n")
            f.write('(property 0 double "distance"\n')
            f.write('\t(default "0" "0")\n')
            for i, e in enumerate(edges_to_use):
                d_tmp = np.linalg.norm(self.pos[e[0]] - self.pos[e[1]])
                f.write("\t(edge " + str(i) + str(' "') + str(d_tmp) + '")\n')
                f.write(
                    "\t(node " + str(e[0]) + str(' "') + str(d_tmp) + '")\n'
                )
            f.write(")\n")

        if node_properties:
            for p_name, (p_dict, default) in node_properties.items():
                if type(list(p_dict.values())[0]) == str:
                    f.write('(property 0 string "%s"\n' % p_name)
                    f.write("\t(default %s %s)\n" % (default, default))
                elif isinstance(list(p_dict.values())[0], Number):
                    f.write('(property 0 double "%s"\n' % p_name)
                    f.write('\t(default "0" "0")\n')
                for n in nodes_to_use:
                    f.write(
                        "\t(node "
                        + str(n)
                        + str(' "')
                        + str(p_dict.get(n, default))
                        + '")\n'
                    )
                f.write(")\n")

        f.write(")")
        f.close()

    def read_from_csv(self, file_path, z_mult, link=1, delim=","):
        def convert_for_csv(v):
            if v.isdigit():
                return int(v)
            else:
                return float(v)

        with open(file_path) as f:
            lines = f.readlines()
            f.close()
        self.time_nodes = {}
        self.time_edges = {}
        unique_id = 0
        self.nodes = set()
        self.edges = set()
        self.successor = {}
        self.predecessor = {}
        self.pos = {}
        self.time_id = {}
        self.time = {}
        self.lin = {}
        self.C_lin = {}
        if not link:
            self.displacement = {}
        lines_to_int = []
        corres = {}
        for l in lines:
            lines_to_int += [
                [convert_for_csv(v.strip()) for v in l.split(delim)]
            ]
        lines_to_int = np.array(lines_to_int)
        if link == 2:
            lines_to_int = lines_to_int[np.argsort(lines_to_int[:, 0])]
        else:
            lines_to_int = lines_to_int[np.argsort(lines_to_int[:, 1])]
        for l in lines_to_int:
            if link == 1:
                id_, t, z, y, x, pred, lin_id = l
            elif link == 2:
                t, z, y, x, id_, pred, lin_id = l
            else:
                id_, t, z, y, x, dz, dy, dx = l
                pred = None
                lin_id = None
            t = int(t)
            pos = np.array([x, y, z])
            C = unique_id
            corres[id_] = C
            pos[-1] = pos[-1] * z_mult
            if pred in corres:
                M = corres[pred]
                self.predecessor[C] = [M]
                self.successor.setdefault(M, []).append(C)
                self.edges.add((M, C))
                self.time_edges.setdefault(t, set()).add((M, C))
                self.lin.setdefault(lin_id, []).append(C)
                self.C_lin[C] = lin_id
            self.pos[C] = pos
            self.nodes.add(C)
            self.time_nodes.setdefault(t, set()).add(C)
            # self.time_id[(t, cell_id)] = C
            self.time[C] = t
            if not link:
                self.displacement[C] = np.array([dx, dy, dz * z_mult])
            unique_id += 1
        self.max_id = unique_id - 1
        self.t_b = min(self.time_nodes)
        self.t_e = max(self.time_nodes)

    def read_from_ASTEC(self, file_path, eigen=False):
        """Read an `xml` or `pkl` file produced by the ASTEC algorithm.

        Args:
            file_path (str): path to an output generated by ASTEC
            eigen (bool): whether or not to read the eigen values, default False
        """
        self._astec_keydictionary = {
            "lineage": {
                "output_key": "cell_lineage",
                "input_keys": [
                    "lineage_tree",
                    "lin_tree",
                    "Lineage tree",
                    "cell_lineage",
                ],
            },
            "h_min": {
                "output_key": "cell_h_min",
                "input_keys": ["cell_h_min", "h_mins_information"],
            },
            "volume": {
                "output_key": "cell_volume",
                "input_keys": [
                    "cell_volume",
                    "volumes_information",
                    "volumes information",
                    "vol",
                ],
            },
            "surface": {
                "output_key": "cell_surface",
                "input_keys": ["cell_surface", "cell surface"],
            },
            "compactness": {
                "output_key": "cell_compactness",
                "input_keys": [
                    "cell_compactness",
                    "Cell Compactness",
                    "compacity",
                    "cell_sphericity",
                ],
            },
            "sigma": {
                "output_key": "cell_sigma",
                "input_keys": ["cell_sigma", "sigmas_information", "sigmas"],
            },
            "label_in_time": {
                "output_key": "cell_labels_in_time",
                "input_keys": [
                    "cell_labels_in_time",
                    "Cells labels in time",
                    "time_labels",
                ],
            },
            "barycenter": {
                "output_key": "cell_barycenter",
                "input_keys": [
                    "cell_barycenter",
                    "Barycenters",
                    "barycenters",
                ],
            },
            "fate": {
                "output_key": "cell_fate",
                "input_keys": ["cell_fate", "Fate"],
            },
            "fate2": {
                "output_key": "cell_fate_2",
                "input_keys": ["cell_fate_2", "Fate2"],
            },
            "fate3": {
                "output_key": "cell_fate_3",
                "input_keys": ["cell_fate_3", "Fate3"],
            },
            "fate4": {
                "output_key": "cell_fate_4",
                "input_keys": ["cell_fate_4", "Fate4"],
            },
            "all-cells": {
                "output_key": "all_cells",
                "input_keys": [
                    "all_cells",
                    "All Cells",
                    "All_Cells",
                    "all cells",
                    "tot_cells",
                ],
            },
            "principal-value": {
                "output_key": "cell_principal_values",
                "input_keys": ["cell_principal_values", "Principal values"],
            },
            "name": {
                "output_key": "cell_name",
                "input_keys": ["cell_name", "Names", "names", "cell_names"],
            },
            "contact": {
                "output_key": "cell_contact_surface",
                "input_keys": [
                    "cell_contact_surface",
                    "cell_cell_contact_information",
                ],
            },
            "history": {
                "output_key": "cell_history",
                "input_keys": [
                    "cell_history",
                    "Cells history",
                    "cell_life",
                    "life",
                ],
            },
            "principal-vector": {
                "output_key": "cell_principal_vectors",
                "input_keys": ["cell_principal_vectors", "Principal vectors"],
            },
            "name-score": {
                "output_key": "cell_naming_score",
                "input_keys": ["cell_naming_score", "Scores", "scores"],
            },
            "problems": {
                "output_key": "problematic_cells",
                "input_keys": ["problematic_cells"],
            },
            "unknown": {
                "output_key": "unknown_key",
                "input_keys": ["unknown_key"],
            },
        }
        if os.path.splitext(file_path)[-1] == ".xml":
            tmp_data = self._read_from_ASTEC_xml(file_path)
        else:
            tmp_data = self._read_from_ASTEC_pkl(file_path, eigen)

        self.name = {}
        self.volume = {}
        self.lT2pkl = {}
        self.pkl2lT = {}
        self.contact = {}
        self.prob_cells = set()
        if "cell_lineage" in tmp_data:
            lt = tmp_data["cell_lineage"]
        elif "lin_tree" in tmp_data:
            lt = tmp_data["lin_tree"]
        else:
            lt = tmp_data["Lineage tree"]
        if "cell_name" in tmp_data:
            names = tmp_data["cell_name"]
        elif "Names" in tmp_data:
            names = tmp_data["Names"]
        else:
            names = {}
        if "cell_fate" in tmp_data:
            self.fates = {}
            fates = tmp_data["cell_fate"]
            do_fates = True
        else:
            do_fates = False
        if "cell_volume" in tmp_data:
            do_volumes = True
            volumes = tmp_data["cell_volume"]
        elif "volume_information" in tmp_data:
            do_volumes = True
            volumes = tmp_data["volume_information"]
        else:
            do_volumes = False
        if "cell_contact_surface" in tmp_data:
            do_surf = True
            surfaces = tmp_data["cell_contact_surface"]
        else:
            do_surf = False
        if "problematic_cells" in tmp_data:
            prob_cells = set(tmp_data["problematic_cells"])
        else:
            prob_cells = set()
        if eigen:
            self.eigen_vectors = {}
            self.eigen_values = {}
            eig_val = tmp_data["cell_principal_values"]
            eig_vec = tmp_data["cell_principal_vectors"]

        inv = {vi: [c] for c, v in lt.items() for vi in v}
        nodes = set(lt).union(inv)
        if "cell_barycenter" in tmp_data:
            pos = tmp_data["cell_barycenter"]
        elif "Barycenters" in tmp_data:
            pos = tmp_data["Barycenters"]
        else:
            pos = dict(
                list(
                    zip(
                        nodes,
                        [
                            [0.0, 0.0, 0.0],
                        ]
                        * len(nodes),
                    )
                )
            )

        unique_id = 0
        id_corres = {}
        for n in nodes:
            if n in prob_cells:
                self.prob_cells.add(unique_id)
            # if n in pos and n in names:
            t = n // 10**4
            self.lT2pkl[unique_id] = n
            self.pkl2lT[n] = unique_id
            self.name[unique_id] = names.get(n, "")
            position = np.array(pos.get(n, [0, 0, 0]), dtype=float)
            self.time_nodes.setdefault(t, set()).add(unique_id)
            self.nodes.add(unique_id)
            self.pos[unique_id] = position
            self.time[unique_id] = t
            id_corres[n] = unique_id
            if do_volumes:
                self.volume[unique_id] = volumes.get(n, 0.0)
            if eigen:
                self.eigen_vectors[unique_id] = eig_vec.get(n)
                self.eigen_values[unique_id] = eig_val.get(n)
            if do_fates:
                self.fates[unique_id] = fates.get(n, "")

            unique_id += 1
        # self.contact = {self.pkl2lT[c]: v for c, v in surfaces.iteritems() if c in self.pkl2lT}
        if do_surf:
            for c in nodes:
                if c in surfaces and c in self.pkl2lT:
                    self.contact[self.pkl2lT[c]] = {
                        self.pkl2lT.get(n, -1): s
                        for n, s in surfaces[c].items()
                        if n % 10**4 == 1 or n in self.pkl2lT
                    }

        for n, new_id in id_corres.items():
            # new_id = id_corres[n]
            if n in inv:
                self.predecessor[new_id] = [id_corres[ni] for ni in inv[n]]
            if n in lt:
                self.successor[new_id] = [
                    id_corres[ni] for ni in lt[n] if ni in id_corres
                ]
                self.edges.update(
                    [(new_id, ni) for ni in self.successor[new_id]]
                )
                for ni in self.successor[new_id]:
                    # self.edges += [(new_id, ni)]
                    self.time_edges.setdefault(t - 1, set()).add((new_id, ni))

        self.t_b = min(self.time_nodes)
        self.t_e = max(self.time_nodes)
        self.max_id = unique_id

    def _read_from_ASTEC_xml(self, file_path):
        def _set_dictionary_value(root):
            if len(root) == 0:
                if root.text is None:
                    return None
                else:
                    return eval(root.text)
            else:
                dictionary = {}
                for child in root:
                    key = child.tag
                    if child.tag == "cell":
                        key = int(child.attrib["cell-id"])
                    dictionary[key] = _set_dictionary_value(child)
            return dictionary

        tree = ET.parse(file_path)
        root = tree.getroot()
        dictionary = {}

        for k, v in self._astec_keydictionary.items():
            if root.tag == v["output_key"]:
                dictionary[str(root.tag)] = _set_dictionary_value(root)
                break
        else:
            for child in root:
                value = _set_dictionary_value(child)
                if value is not None:
                    dictionary[str(child.tag)] = value
        return dictionary

    def _read_from_ASTEC_pkl(self, file_path, eigen=False):
        import pickle as pkl

        with open(file_path, "rb") as f:
            tmp_data = pkl.load(f, encoding="latin1")
            f.close()
        new_ref = {}
        for k, v in self._astec_keydictionary.items():
            for key in v["input_keys"]:
                new_ref[key] = v["output_key"]
        new_dict = {}

        for k, v in tmp_data.items():
            if k in new_ref:
                new_dict[new_ref[k]] = v
        return new_dict

    def read_from_txt_for_celegans(self, file):
        implicit_l_t = {
            "AB": "P0",
            "P1": "P0",
            "EMS": "P1",
            "P2": "P1",
            "MS": "EMS",
            "E": "EMS",
            "C": "P2",
            "P3": "P2",
            "D": "P3",
            "P4": "P3",
            "Z2": "P4",
            "Z3": "P4",
        }
        with open(file, "r") as f:
            raw = f.readlines()[1:]
            f.close()
        self.name = {}

        unique_id = 0
        for l in raw:
            t = int(l.split("\t")[0])
            self.name[unique_id] = l.split("\t")[1]
            position = np.array(l.split("\t")[2:5], dtype=float)
            self.time_nodes.setdefault(t, set()).add(unique_id)
            self.nodes.add(unique_id)
            self.pos[unique_id] = position
            self.time[unique_id] = t
            unique_id += 1

        self.t_b = min(self.time_nodes)
        self.t_e = max(self.time_nodes)

        for t, cells in self.time_nodes.items():
            if t != self.t_b:
                prev_cells = self.time_nodes[t - 1]
                name_to_id = {self.name[c]: c for c in prev_cells}
                for c in cells:
                    if self.name[c] in name_to_id:
                        p = name_to_id[self.name[c]]
                    elif self.name[c][:-1] in name_to_id:
                        p = name_to_id[self.name[c][:-1]]
                    elif implicit_l_t.get(self.name[c]) in name_to_id:
                        p = name_to_id[implicit_l_t.get(self.name[c])]
                    else:
                        print(
                            "error, cell %s has no predecessors" % self.name[c]
                        )
                        p = None
                    self.predecessor.setdefault(c, []).append(p)
                    self.successor.setdefault(p, []).append(c)
                    self.edges.add((p, c))
                    self.time_edges.setdefault(t - 1, set()).add((p, c))
            self.max_id = unique_id

    def read_from_txt_for_celegans_CAO(
        self, file, reorder=False, raw_size=None, shape=None
    ):
        implicit_l_t = {
            "AB": "P0",
            "P1": "P0",
            "EMS": "P1",
            "P2": "P1",
            "MS": "EMS",
            "E": "EMS",
            "C": "P2",
            "P3": "P2",
            "D": "P3",
            "P4": "P3",
            "Z2": "P4",
            "Z3": "P4",
        }
        split_line = lambda line: (
            line.split()[0],
            eval(line.split()[1]),
            eval(line.split()[2]),
            eval(line.split()[3]),
            eval(line.split()[4]),
        )
        with open(file, "r") as f:
            raw = f.readlines()[1:]
            f.close()
        self.name = {}

        unique_id = 0
        for name, t, z, x, y in map(split_line, raw):
            self.name[unique_id] = name
            position = np.array([x, y, z], dtype=np.float)
            self.time_nodes.setdefault(t, set()).add(unique_id)
            self.nodes.add(unique_id)
            if reorder:
                flip = lambda x: np.array([x[0], x[1], raw_size[2] - x[2]])
                adjust = lambda x: (shape / raw_size * flip(x))[[1, 0, 2]]
                self.pos[unique_id] = adjust(position)
            else:
                self.pos[unique_id] = position
            self.time[unique_id] = t
            unique_id += 1

        self.t_b = min(self.time_nodes)
        self.t_e = max(self.time_nodes)

        for t, cells in self.time_nodes.items():
            if t != self.t_b:
                prev_cells = self.time_nodes[t - 1]
                name_to_id = {self.name[c]: c for c in prev_cells}
                for c in cells:
                    if self.name[c] in name_to_id:
                        p = name_to_id[self.name[c]]
                    elif self.name[c][:-1] in name_to_id:
                        p = name_to_id[self.name[c][:-1]]
                    elif implicit_l_t.get(self.name[c]) in name_to_id:
                        p = name_to_id[implicit_l_t.get(self.name[c])]
                    else:
                        print(
                            "error, cell %s has no predecessors" % self.name[c]
                        )
                        p = None
                    self.predecessor.setdefault(c, []).append(p)
                    self.successor.setdefault(p, []).append(c)
                    self.edges.add((p, c))
                    self.time_edges.setdefault(t - 1, set()).add((p, c))
            self.max_id = unique_id

    def read_tgmm_xml(self, file_format, tb, te, z_mult=1.0):
        """Reads a lineage tree from TGMM xml output.

        Args:
            file_format (str): path to the xmls location.
                    it should be written as follow:
                    path/to/xml/standard_name_t{t:06d}.xml where (as an example)
                    {t:06d} means a series of 6 digits representing the time and
                    if the time values is smaller that 6 digits, the missing
                    digits are filed with 0s
            tb (int): first time point to read
            te (int): last time point to read
            z_mult (float): aspect ratio
        """
        self.time_nodes = {}
        self.time_edges = {}
        unique_id = 0
        self.nodes = set()
        self.edges = set()
        self.successor = {}
        self.predecessor = {}
        self.pos = {}
        self.time_id = {}
        self.time = {}
        self.mother_not_found = []
        self.ind_cells = {}
        self.svIdx = {}
        self.lin = {}
        self.C_lin = {}
        self.coeffs = {}
        self.intensity = {}
        self.W = {}
        for t in range(tb, te + 1):
            print(t, end=" ")
            if t % 10 == 0:
                print()
            tree = ET.parse(file_format.format(t=t))
            root = tree.getroot()
            self.time_nodes[t] = set()
            self.time_edges[t] = set()
            for it in root:
                if (
                    not "-1.#IND" in it.attrib["m"]
                    and not "nan" in it.attrib["m"]
                ):
                    M_id, pos, cell_id, svIdx, lin_id = (
                        int(it.attrib["parent"]),
                        [
                            float(v)
                            for v in it.attrib["m"].split(" ")
                            if v != ""
                        ],
                        int(it.attrib["id"]),
                        [
                            int(v)
                            for v in it.attrib["svIdx"].split(" ")
                            if v != ""
                        ],
                        int(it.attrib["lineage"]),
                    )
                    try:
                        alpha, W, nu, alphaPrior = (
                            float(it.attrib["alpha"]),
                            [
                                float(v)
                                for v in it.attrib["W"].split(" ")
                                if v != ""
                            ],
                            float(it.attrib["nu"]),
                            float(it.attrib["alphaPrior"]),
                        )
                        pos = np.array(pos)
                        pos_tmp = np.round(pos).astype(np.uint16)
                        C = unique_id
                        pos[-1] = pos[-1] * z_mult
                        if (t - 1, M_id) in self.time_id:
                            M = self.time_id[(t - 1, M_id)]
                            self.successor.setdefault(M, []).append(C)
                            self.predecessor.setdefault(C, []).append(M)
                            self.edges.add((M, C))
                            self.time_edges[t].add((M, C))
                        else:
                            if M_id != -1:
                                self.mother_not_found.append(C)
                        self.pos[C] = pos
                        self.nodes.add(C)
                        self.time_nodes[t].add(C)
                        self.time_id[(t, cell_id)] = C
                        self.time[C] = t
                        self.svIdx[C] = svIdx
                        self.lin.setdefault(lin_id, []).append(C)
                        self.C_lin[C] = lin_id
                        self.intensity[C] = max(alpha - alphaPrior, 0)
                        tmp = list(np.array(W) * nu)
                        self.W[C] = np.array(W).reshape(3, 3)
                        self.coeffs[C] = (
                            tmp[:3] + tmp[4:6] + tmp[8:9] + list(pos)
                        )
                        unique_id += 1
                    except Exception as e:
                        pass
                else:
                    if t in self.ind_cells:
                        self.ind_cells[t] += 1
                    else:
                        self.ind_cells[t] = 1
        self.max_id = unique_id - 1

    def read_from_mamut_xml(self, path):
        """Read a lineage tree from a MaMuT xml.

        Args:
            path (str): path to the MaMut xml
        """
        tree = ET.parse(path)
        for elem in tree.getroot():
            if elem.tag == "Model":
                Model = elem
        FeatureDeclarations, AllSpots, AllTracks, FilteredTracks = list(Model)

        for attr in self.xml_attributes:
            self.__dict__[attr] = {}
        self.time_nodes = {}
        self.time_edges = {}
        self.nodes = set()
        self.pos = {}
        self.time = {}
        self.node_name = {}
        for frame in AllSpots:
            t = int(frame.attrib["frame"])
            self.time_nodes[t] = set()
            for cell in frame:
                cell_id, n, x, y, z = (
                    int(cell.attrib["ID"]),
                    cell.attrib["name"],
                    float(cell.attrib["POSITION_X"]),
                    float(cell.attrib["POSITION_Y"]),
                    float(cell.attrib["POSITION_Z"]),
                )
                self.time_nodes[t].add(cell_id)
                self.nodes.add(cell_id)
                self.pos[cell_id] = np.array([x, y, z])
                self.time[cell_id] = t
                self.node_name[cell_id] = n
                if "TISSUE_NAME" in cell.attrib:
                    if not hasattr(self, "fate"):
                        self.fate = {}
                    self.fate[cell_id] = cell.attrib["TISSUE_NAME"]
                if "TISSUE_TYPE" in cell.attrib:
                    if not hasattr(self, "fate_nb"):
                        self.fate_nb = {}
                    self.fate_nb[cell_id] = eval(cell.attrib["TISSUE_TYPE"])
                for attr in cell.attrib:
                    if attr in self.xml_attributes:
                        self.__dict__[attr][cell_id] = eval(cell.attrib[attr])

        self.edges = set()
        self.roots = []
        tracks = {}
        self.successor = {}
        self.predecessor = {}
        self.track_name = {}
        for track in AllTracks:
            t_id, l = int(track.attrib["TRACK_ID"]), float(
                track.attrib["TRACK_DURATION"]
            )
            t_name = track.attrib["name"]
            tracks[t_id] = []
            for edge in track:
                s, t = int(edge.attrib["SPOT_SOURCE_ID"]), int(
                    edge.attrib["SPOT_TARGET_ID"]
                )
                if s in self.nodes and t in self.nodes:
                    if self.time[s] > self.time[t]:
                        s, t = t, s
                    self.successor.setdefault(s, []).append(t)
                    self.predecessor.setdefault(t, []).append(s)
                    self.track_name[s] = t_name
                    self.track_name[t] = t_name
                    tracks[t_id].append((s, t))
                    self.edges.add((s, t))
        self.t_b = min(self.time_nodes.keys())
        self.t_e = max(self.time_nodes.keys())

    def to_binary(self, fname, starting_points=None):
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
            starting_points = [
                c
                for c in self.successor.keys()
                if self.predecessor.get(c, []) == []
            ]
        number_sequence = [-1]
        pos_sequence = []
        time_sequence = []
        default_lin = -1
        for c in starting_points:
            time_sequence.append(self.time.get(c, 0))
            to_treat = [c]
            while to_treat != []:
                curr_c = to_treat.pop()
                number_sequence.append(curr_c)
                pos_sequence += list(self.pos[curr_c])
                if self.successor.get(curr_c, []) == []:
                    number_sequence.append(-1)
                elif len(self.successor[curr_c]) == 1:
                    to_treat += self.successor[curr_c]
                else:
                    number_sequence.append(-2)
                    to_treat += self.successor[curr_c]
        remaining_nodes = set(self.nodes) - set(number_sequence)

        for c in remaining_nodes:
            time_sequence.append(self.time.get(c, 0))
            number_sequence.append(c)
            pos_sequence += list(self.pos[c])
            number_sequence.append(-1)

        f = open(fname, "wb")
        f.write(struct.pack("q", len(number_sequence)))
        f.write(struct.pack("q", len(time_sequence)))
        f.write(struct.pack("q", len(pos_sequence)))
        f.write(struct.pack("q" * len(number_sequence), *number_sequence))
        f.write(struct.pack("H" * len(time_sequence), *time_sequence))
        f.write(struct.pack("d" * len(pos_sequence), *pos_sequence))

        f.close()

    def read_from_binary(self, fname, reverse_time=False):
        """Reads a binary lineageTree file name.
        Format description: see self.to_binary

        Args:
            fname: string, path to the binary file
            reverse_time: bool, not used
        """
        q_size = struct.calcsize("q")
        H_size = struct.calcsize("H")
        d_size = struct.calcsize("d")

        f = open(fname, "rb")
        len_tree = struct.unpack("q", f.read(q_size))[0]
        len_time = struct.unpack("q", f.read(q_size))[0]
        len_pos = struct.unpack("q", f.read(q_size))[0]
        number_sequence = list(
            struct.unpack("q" * len_tree, f.read(q_size * len_tree))
        )
        time_sequence = list(
            struct.unpack("H" * len_time, f.read(H_size * len_time))
        )
        pos_sequence = np.array(
            struct.unpack("d" * len_pos, f.read(d_size * len_pos))
        )

        f.close()

        successor = {}
        predecessor = {}
        time = {}
        time_nodes = {}
        time_edges = {}
        pos = {}
        is_root = {}
        nodes = []
        edges = []
        waiting_list = []
        print(number_sequence[0])
        i = 0
        done = False
        if max(number_sequence[::2]) == -1:
            tmp = number_sequence[1::2]
            if len(tmp) * 3 == len(pos_sequence) == len(time_sequence) * 3:
                time = dict(list(zip(tmp, time_sequence)))
                for c, t in time.items():
                    time_nodes.setdefault(t, set()).add(c)
                pos = dict(
                    list(zip(tmp, np.reshape(pos_sequence, (len_time, 3))))
                )
                is_root = {c: True for c in tmp}
                nodes = tmp
                done = True
        shown = set()
        while (
            i < len(number_sequence) and not done
        ):  # , c in enumerate(number_sequence[:-1]):
            c = number_sequence[i]
            if c == -1:
                if waiting_list != []:
                    prev_mother = waiting_list.pop()
                    successor[prev_mother].insert(0, number_sequence[i + 1])
                    edges.append((prev_mother, number_sequence[i + 1]))
                    time_edges.setdefault(t, set()).add(
                        (prev_mother, number_sequence[i + 1])
                    )
                    is_root[number_sequence[i + 1]] = False
                    t = time[prev_mother] + 1
                else:
                    t = time_sequence.pop(0)
                    is_root[number_sequence[i + 1]] = True

            elif c == -2:
                successor[waiting_list[-1]] = [number_sequence[i + 1]]
                edges.append((waiting_list[-1], number_sequence[i + 1]))
                time_edges.setdefault(t, set()).add(
                    (waiting_list[-1], number_sequence[i + 1])
                )
                is_root[number_sequence[i + 1]] = False
                pos[waiting_list[-1]] = pos_sequence[:3]
                pos_sequence = pos_sequence[3:]
                nodes.append(waiting_list[-1])
                time[waiting_list[-1]] = t
                time_nodes.setdefault(t, set()).add(waiting_list[-1])
                t += 1

            elif number_sequence[i + 1] >= 0:
                successor[c] = [number_sequence[i + 1]]
                edges.append((c, number_sequence[i + 1]))
                time_edges.setdefault(t, set()).add(
                    (c, number_sequence[i + 1])
                )
                is_root[number_sequence[i + 1]] = False
                pos[c] = pos_sequence[:3]
                pos_sequence = pos_sequence[3:]
                nodes.append(c)
                time[c] = t
                time_nodes.setdefault(t, set()).add(c)
                t += 1

            elif number_sequence[i + 1] == -2:
                waiting_list += [c]

            elif number_sequence[i + 1] == -1:
                pos[c] = pos_sequence[:3]
                pos_sequence = pos_sequence[3:]
                nodes.append(c)
                time[c] = t
                time_nodes.setdefault(t, set()).add(c)
                t += 1
                i += 1
                if waiting_list != []:
                    prev_mother = waiting_list.pop()
                    successor[prev_mother].insert(0, number_sequence[i + 1])
                    edges.append((prev_mother, number_sequence[i + 1]))
                    time_edges.setdefault(t, set()).add(
                        (prev_mother, number_sequence[i + 1])
                    )
                    if i + 1 < len(number_sequence):
                        is_root[number_sequence[i + 1]] = False
                    t = time[prev_mother] + 1
                else:
                    if 0 < len(time_sequence):
                        t = time_sequence.pop(0)
                    if i + 1 < len(number_sequence):
                        is_root[number_sequence[i + 1]] = True
            i += 1

        predecessor = {vi: [k] for k, v in successor.items() for vi in v}

        self.successor = successor
        self.predecessor = predecessor
        self.time = time
        self.time_nodes = time_nodes
        self.time_edges = time_edges
        self.pos = pos
        self.nodes = set(nodes)
        self.edges = set(edges)
        self.t_b = min(time_nodes.keys())
        self.t_e = max(time_nodes.keys())
        self.is_root = is_root
        self.max_id = max(self.nodes)

    def get_idx3d(self, t):
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
        to_check_self = list(self.time_nodes[t])
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

    def get_gabriel_graph(self, t):
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
                    delaunay_graph.setdefault(e1, set([])).add(e2)
                    delaunay_graph.setdefault(e2, set([])).add(e1)

            Gabriel_graph = {}

            for e1, neighbs in delaunay_graph.items():
                for ni in neighbs:
                    if not any(
                        [
                            np.linalg.norm((data[ni] + data[e1]) / 2 - data[i])
                            < np.linalg.norm(data[ni] - data[e1]) / 2
                            for i in delaunay_graph[e1].intersection(
                                delaunay_graph[ni]
                            )
                        ]
                    ):
                        Gabriel_graph.setdefault(data_corres[e1], set()).add(
                            data_corres[ni]
                        )
                        Gabriel_graph.setdefault(data_corres[ni], set()).add(
                            data_corres[e1]
                        )

            self.Gabriel_graph[t] = Gabriel_graph

        return self.Gabriel_graph[t]

    def get_predecessors(self, x, depth=None):
        """Computes the predecessors of the node *x* up to
        *depth* predecessors. The ordered list of ids is returned.

        Args:
            x (int): id of the node to compute
            depth (int): maximum number of predecessors to return
        Returns:
            [int, ]: list of ids, the last id is *x*
        """
        cycle = [x]
        acc = 0
        while (
            len(
                self.successor.get(self.predecessor.get(cycle[0], [-1])[0], [])
            )
            == 1
            and acc != depth
        ):
            cycle.insert(0, self.predecessor[cycle[0]][0])
            acc += 1
        return cycle

    def get_successors(self, x, depth=None):
        """Computes the successors of the node *x* up to
        *depth* successors. The ordered list of ids is returned.

        Args:
            x (int): id of the node to compute
            depth (int): maximum number of predecessors to return
        Returns:
            [int, ]: list of ids, the first id is *x*
        """
        cycle = [x]
        acc = 0
        while len(self.successor.get(cycle[-1], [])) == 1 and acc != depth:
            cycle += self.successor[cycle[-1]]
            acc += 1
        return cycle

    def get_cycle(self, x, depth=None, depth_pred=None, depth_succ=None):
        """Computes the predecessors and successors of the node *x* up to
        *depth_pred* predecessors plus *depth_succ* successors.
        If the value *depth* is provided and not None,
        *depth_pred* and *depth_succ* are overwrited by *depth*.
        The ordered list of ids is returned.

        Args:
            x (int): id of the node to compute
            depth (int): maximum number of predecessors and successor to return
            depth_pred (int): maximum number of predecessors to return
            depth_succ (int): maximum number of successors to return
        Returns:
            [int, ]: list of ids
        """
        if depth is not None:
            depth_pred = depth_succ = depth
        return self.get_predecessors(x, depth_pred)[:-1] + self.get_successors(
            x, depth_succ
        )

    def get_all_tracks(self):
        """Computes all the tracks of a given lineage tree,
        stores it in *self.all_tracks* and returns it.

        Returns:
            ([[int, ...], ...]): list of lists containing track cell ids
        """
        if not hasattr(self, "all_tracks"):
            self.all_tracks = []
            to_do = set(self.nodes)
            while len(to_do) != 0:
                current = to_do.pop()
                track = self.get_cycle(current)
                self.all_tracks += [track]
                to_do -= set(track)
        return self.all_tracks

    def get_sub_tree(self, x, preorder=False):
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
        to_do = [x]
        sub_tree = []
        while 0 < len(to_do):
            curr = to_do.pop(0)
            succ = self.successor.get(curr, [])
            if preorder:
                to_do = succ + to_do
            else:
                to_do += succ
            sub_tree += [curr]
        return sub_tree

    def compute_spatial_density(self, t_b=None, t_e=None, th=50):
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
        time_range = set(range(t_b, t_e + 1)).intersection(self.time_nodes)
        for t in time_range:
            idx3d, nodes = self.get_idx3d(t)
            nb_ni = [
                (len(ni) - 1) / s_vol
                for ni in idx3d.query_ball_tree(idx3d, th)
            ]
            self.spatial_density.update(dict(zip(nodes, nb_ni)))
        return self.spatial_density

    def compute_k_nearest_neighbours(self, k=10):
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
        for t, Cs in self.time_nodes.items():
            use_k = k if k < len(nodes) else len(nodes)
            idx3d, nodes = self.get_idx3d(t)
            pos = [self.pos[c] for c in nodes]
            dist, neighbs = idx3d.query(pos, use_k)
            out = dict(zip(nodes, [set(nodes[ni[1:]]) for ni in neighbs]))
            self.kn_graph.update(out)
        return self.kn_graph

    def compute_spatial_edges(self, th=50):
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
        for t, Cs in self.time_nodes.items():
            idx3d, nodes = self.get_idx3d(t)
            neighbs = idx3d.query_ball_tree(idx3d, th)
            out = dict(zip(nodes, [set(nodes[ni]) for ni in neighbs]))
            self.th_edges.update(
                {k: v.difference([k]) for k, v in out.items()}
            )
        return self.th_edges

    # def DTW(self, t1, t2, max_w=None, start_delay=None, end_delay=None,
    #         metric='euclidian', **kwargs):
    #     """ Computes the dynamic time warping distance between the tracks t1 and t2

    #         Args:
    #             t1 ([int, ]): list of node ids for the first track
    #             t2 ([int, ]): list of node ids for the second track
    #             w (int): maximum wapring allowed (default infinite),
    #                      if w=1 then the DTW is the distance between t1 and t2
    #             start_delay (int): maximum number of time points that can be
    #                                skipped at the beginning of the track
    #             end_delay (int): minimum number of time points that can be
    #                              skipped at the beginning of the track
    #             metric (str): str or callable, optional The distance metric to use.
    #                           Default='euclidean'. Refer to the documentation for
    #                           scipy.spatial.distance.cdist. Some examples:
    #                           'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
    #                           'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
    #                           'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao',
    #                           'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
    #                           'wminkowski', 'yule'
    #             **kwargs (dict): Extra arguments to `metric`: refer to each metric
    #                              documentation in scipy.spatial.distance (optional)

    #         Returns:
    #             float: the dynamic time warping distance between the two tracks
    #     """
    #     from scipy.sparse import
    #     pos_t1 = [self.pos[ti] for ti in t1]
    #     pos_t2 = [self.pos[ti] for ti in t2]
    #     distance_matrix = np.zeros((len(t1), len(t2))) + np.inf

    #     c = distance.cdist(exp_data, num_data, metric=metric, **kwargs)

    #     d = np.zeros(c.shape)
    #     d[0, 0] = c[0, 0]
    #     n, m = c.shape
    #     for i in range(1, n):
    #         d[i, 0] = d[i-1, 0] + c[i, 0]
    #     for j in range(1, m):
    #         d[0, j] = d[0, j-1] + c[0, j]
    #     for i in range(1, n):
    #         for j in range(1, m):
    #             d[i, j] = c[i, j] + min((d[i-1, j], d[i, j-1], d[i-1, j-1]))
    #     return d[-1, -1], d

    def __getitem__(self, item):
        return self.__dict__[item]

    def __init__(
        self,
        file_format=None,
        tb=None,
        te=None,
        z_mult=1.0,
        file_type="",
        delim=",",
        eigen=False,
        shape=None,
        raw_size=None,
        reorder=False,
        xml_attributes=[],
    ):
        """Main library to build tree graph representation of lineage tree data
        It can read TGMM, ASTEC, SVF, MaMuT and TrackMate outputs.

        Args:
            file_format (strr): either - path format to TGMM xmls
                                       - path to the MaMuT xml
                                       - path to the binary file
            tb (int): first time point (necessary for TGMM xmls only)
            te (int): last time point (necessary for TGMM xmls only)
            z_mult (float): z aspect ratio if necessary (usually only for TGMM xmls)
            file_type (str): type of input file. Accepts:
                'TGMM, 'ASTEC', MaMuT', 'TrackMate', 'csv', 'celegans', 'binary'
                default is 'binary'
        """
        self.time_nodes = {}
        self.time_edges = {}
        self.max_id = -1
        self.next_id = []
        self.nodes = set()
        self.edges = set()
        self.roots = set()
        self.successor = {}
        self.predecessor = {}
        self.pos = {}
        self.time_id = {}
        self.time = {}
        self.kdtrees = {}
        self.spatial_density = {}
        self.progeny = {}
        self.xml_attributes = xml_attributes
        file_type = file_type.lower()
        if file_type == "tgmm":
            self.read_tgmm_xml(file_format, tb, te, z_mult)
            self.t_b = tb
            self.t_e = te
        elif file_type == "mamut" or file_type == "trackmate":
            self.read_from_mamut_xml(file_format)
        elif file_type == "celegans":
            self.read_from_txt_for_celegans(file_format)
        elif file_type == "celegans_cao":
            self.read_from_txt_for_celegans_CAO(
                file_format, reorder=reorder, shape=shape, raw_size=raw_size
            )
        elif file_type == "astec":
            self.read_from_ASTEC(file_format, eigen)
        elif file_type == "csv":
            self.read_from_csv(file_format, z_mult, link=1, delim=delim)
        elif file_format is not None:
            self.read_from_binary(file_format)
