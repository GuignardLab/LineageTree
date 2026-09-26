from __future__ import annotations

import os
import pickle as pkl
from collections.abc import Callable
from numbers import Number
from typing import TYPE_CHECKING

import numpy as np
import svgwrite

if TYPE_CHECKING:
    from ..lineage_tree import LineageTree


def _get_height(lT: LineageTree, c: int, done: dict) -> float:
    """Compute the horizontal position of a node and its descendants.

    The position of a node is the mean of the positions of its successors;
    leaves must already be in ``done``. This function is specific to
    ``write_to_svg``.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    c : int
        Id of the node from which the positions are computed.
    done : dict of {int: list of float}
        Maps a node id to its ``[x, y]`` position. Filled in place.

    Returns
    -------
    float
        The horizontal position of the node ``c``.
    """
    to_do = [c]
    while to_do:
        curr = to_do[-1]
        if curr in done:
            to_do.pop()
            continue
        missing = [di for di in lT._successor[curr] if di not in done]
        if missing:
            to_do.extend(missing)
            continue
        to_do.pop()
        x = np.mean([done[di][0] for di in lT._successor[curr]])
        done[curr] = [x, lT.vert_space_factor * lT._time[curr]]
    return done[c][0]


def write_to_svg(
    lT: LineageTree,
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
    """Write the lineage tree to an SVG file.

    Node and edge colouring and size can be provided.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    file_name : str
        Path of the SVG file to create.
    roots : list of int, optional
        The nodes whose subtrees are drawn. Defaults to all the roots
        (except, for ASTEC data, those whose ``image_label`` is 1).
    draw_nodes : bool, default=True
        Whether to draw the nodes.
    draw_edges : bool, default=True
        Whether to draw the edges.
    order_key : Callable, optional
        Function usable as the ``key`` argument of ``sort``/``sorted``, used
        to order the roots and the successors.
    vert_space_factor : float, default=0.5
        The vertical position of a node is its time; ``vert_space_factor`` is a
        multiplier to space nodes more or less in time.
    horizontal_space : float, default=1
        Space between two consecutive nodes.
    node_size : Callable or str, optional
        A function that maps a node id to a float value determining the radius
        of the node. The default returns the constant value
        ``vert_space_factor / 2.1``. If a string is given and is a property of
        the tree, the size is mapped according to that property.
    stroke_width : Callable, optional
        A function that maps a node id to a float value determining the width of
        the daughter edge. The default returns the constant value
        ``vert_space_factor / 2.2``.
    factor : float, default=1.0
        Scaling factor for node positions.
    node_color : Callable or str, optional
        A function that maps a node id to an RGB triplet between 0 and 255,
        determining the colour of the node. If a string is given and is a
        property of the tree, the colour is mapped according to that property.
    stroke_color : Callable, optional
        A function that maps a node id to an RGB triplet between 0 and 255,
        determining the colour of the stroke of the inward edge.
    positions : dict of {int: list of float}, optional
        Dictionary that maps a node id to a 2D position. If provided, it is used
        to position the nodes.
    node_color_map : str, optional
        Name of the matplotlib colormap used when ``node_color`` or
        ``stroke_color`` is a property name. Defaults to ``"viridis"``.
    """

    def normalize_values(v, nodes, _range, shift, mult):
        min_ = np.percentile(v, 1)
        max_ = np.percentile(v, 99)
        values = _range * ((v - min_) / (max_ - min_)) + shift
        values_dict_nodes = dict(zip(nodes, values, strict=True))
        return lambda x: values_dict_nodes[x] * mult

    if roots is None:
        roots = list(lT.roots)
        if hasattr(lT, "image_label"):
            roots = [node for node in roots if lT.image_label[node] != 1]
    else:
        roots = list(roots)

    if node_size is None:

        def node_size(x):
            return vert_space_factor / 2.1

    elif isinstance(node_size, str):
        values = np.array([getattr(lT, node_size)[c] for c in lT.nodes])
        node_size = normalize_values(
            values, lT.nodes, 0.5, 0.5, vert_space_factor / 2.1
        )
    if stroke_width is None:

        def stroke_width(x):
            return vert_space_factor / 2.2

    if node_color is None:

        def node_color(x):
            return 0, 0, 0

    elif isinstance(node_color, str) and node_color in lT.__dict__:
        from matplotlib import colormaps

        if node_color_map in colormaps:
            cmap = colormaps[node_color_map]
        else:
            cmap = colormaps["viridis"]
        values = np.array([getattr(lT, node_color)[c] for c in lT.nodes])
        normed_vals = normalize_values(values, lT.nodes, 1, 0, 1)

        def node_color(x):
            return [k * 255 for k in cmap(normed_vals(x))[:-1]]

    coloring_edges = stroke_color is not None
    if not coloring_edges:

        def stroke_color(x):
            return 0, 0, 0

    elif isinstance(stroke_color, str) and stroke_color in lT.__dict__:
        from matplotlib import colormaps

        if node_color_map in colormaps:
            cmap = colormaps[node_color_map]
        else:
            cmap = colormaps["viridis"]
        values = np.array([getattr(lT, stroke_color)[c] for c in lT.nodes])
        normed_vals = normalize_values(values, lT.nodes, 1, 0, 1)

        def stroke_color(x):
            return [k * 255 for k in cmap(normed_vals(x))[:-1]]

    prev_x = 0
    lT.vert_space_factor = vert_space_factor
    if order_key is not None:
        roots.sort(key=order_key)
    treated_nodes = []

    pos_given = positions is not None
    if not pos_given:
        positions = dict(
            zip(
                lT.nodes,
                [
                    [0.0, 0.0],
                ]
                * len(lT.nodes),
                strict=True,
            ),
        )
    for _i, r in enumerate(roots):
        r_leaves = []
        to_do = [r]
        while len(to_do) != 0:
            curr = to_do.pop(0)
            treated_nodes += [curr]
            if lT._successor[curr]:
                if order_key is not None:
                    to_do += sorted(lT._successor[curr], key=order_key)
                else:
                    to_do += lT._successor[curr]
            else:
                r_leaves += [curr]
        r_pos = {
            leave: [
                prev_x + horizontal_space * (1 + j),
                lT.vert_space_factor * lT._time[leave],
            ]
            for j, leave in enumerate(r_leaves)
        }
        lT._get_height(r, r_pos)
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
            c_chain = lT.get_chain_of_node(curr)
            x1, y1 = positions[c_chain[0]]
            x2, y2 = positions[c_chain[-1]]
            dwg.add(
                dwg.line(
                    (factor * x1, factor * y1),
                    (factor * x2, factor * y2),
                    stroke=svgwrite.rgb(0, 0, 0),
                )
            )
            for si in lT._successor[c_chain[-1]]:
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
            for si in lT._successor[c]:
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


def write_to_tlp(
    lT: LineageTree,
    fname: str,
    t_min: int = -1,
    t_max: int = np.inf,
    nodes_to_use: list[int] | None = None,
    temporal: bool = True,
    spatial: str | None = None,
    write_layout: bool = True,
    node_properties: dict | None = None,
    Names: str | bool = False,
) -> None:
    """Write a lineage tree into a Tulip (``.tlp``) file.

    Tulip is a graph visualisation tool (https://tulip.labri.fr).

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    fname : str
        Path to the Tulip file to create.
    t_min : int, default=-1
        Nodes at ``t_min`` or earlier are left out.
    t_max : int, default=np.inf
        Nodes after ``t_max`` are left out.
    nodes_to_use : list of int, optional
        The nodes to write. If None, all the nodes between ``t_min`` and
        ``t_max`` are written.
    temporal : bool, default=True
        Whether the temporal links should be written.
    spatial : {"ball", "kn", "GG"}, optional
        Also write spatial edges from a spatial neighbourhood graph, which
        has to be computed before running this function:

        - ``"ball"``: neighbours within a distance, from
          [`spatial_edges`][lineagetree.LineageTree.spatial_edges],
        - ``"kn"``: k nearest neighbours, from
          [`k_nearest_neighbours`][lineagetree.LineageTree.k_nearest_neighbours],
        - ``"GG"``: Gabriel graph, from
          [`gabriel_graph`][lineagetree.LineageTree.gabriel_graph].

        If None, no spatial edges are written.

    Raises
    ------
    ValueError
        If ``spatial`` is not one of the values above, or if the
        corresponding graph has not been computed.
    write_layout : bool, default=True
        Whether to write the spatial position as layout.
    node_properties : dict of {str: list}, optional
        Properties to write. Each key (the property name) maps to a pair made
        of a dictionary (node id to property value) and a default value.
    Names : str or bool, default=False
        For ASTEC outputs only: the key of ``node_properties`` that holds the
        cell names. The nodes are then written sorted by name. False to keep
        the default order.
    """

    def format_names(names_which_matter):
        """Return formatted node names."""
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
        for c, N in s_g.items():
            s_edges.update(tuple(sorted([int(c), int(ni)])) for ni in N)
        return s_edges

    if spatial:
        spatial_graphs = {
            "gg": ("Gabriel_graph", "gabriel_graph"),
            "kn": ("kn_graph", "k_nearest_neighbours"),
            "ball": ("th_edges", "spatial_edges"),
        }
        if spatial.lower() not in spatial_graphs:
            raise ValueError(
                f"spatial should be one of 'ball', 'kn' or 'GG', not {spatial!r}."
            )
        attr, method = spatial_graphs[spatial.lower()]
        if not hasattr(lT, attr):
            raise ValueError(
                f"Compute the spatial graph with lT.{method}() before "
                f"writing it with spatial={spatial!r}."
            )
        s_edges = spatial_adjlist_to_set(getattr(lT, attr))

    with open(fname, "w") as f:
        f.write('(tlp "2.0"\n')
        f.write("(nodes ")

        if not nodes_to_use:
            if t_max != np.inf or t_min > -1:
                nodes_to_use = [
                    n for n in lT.nodes if t_min < lT._time[n] <= t_max
                ]
                edges_to_use = []
                if temporal:
                    edges_to_use += [
                        e for e in lT.edges if t_min < lT._time[e[0]] < t_max
                    ]
                if spatial:
                    edges_to_use += [
                        e for e in s_edges if t_min < lT._time[e[0]] < t_max
                    ]
            else:
                nodes_to_use = list(lT.nodes)
                edges_to_use = []
                if temporal:
                    edges_to_use += list(lT.edges)
                if spatial:
                    edges_to_use += list(s_edges)
        else:
            edges_to_use = []
            nodes_to_use = set(nodes_to_use)
            if temporal:
                for n in nodes_to_use:
                    for d in lT._successor[n]:
                        if d in nodes_to_use:
                            edges_to_use.append((n, d))
            if spatial:
                edges_to_use += [
                    e
                    for e in s_edges
                    if e[0] in nodes_to_use and e[1] in nodes_to_use
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
                    len(lT._successor.get(lT._predecessor.get(k, [-1])[0], ()))
                    != 1
                    or lT._time[k] == t_min + 1
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
            f.write("\t(node " + str(n) + ' "' + str(lT._time[n]) + '")\n')
        f.write(")\n")

        if write_layout:
            f.write('(property 0 layout "viewLayout"\n')
            f.write('\t(default "(0, 0, 0)" "()")\n')
            for n in nodes_to_use:
                f.write(
                    "\t(node " + str(n) + ' "' + str(tuple(lT.pos[n])) + '")\n'
                )
            f.write(")\n")
            f.write('(property 0 double "distance"\n')
            f.write('\t(default "0" "0")\n')
            for i, e in enumerate(edges_to_use):
                d_tmp = np.linalg.norm(lT.pos[e[0]] - lT.pos[e[1]])
                f.write("\t(edge " + str(i) + ' "' + str(d_tmp) + '")\n')
                f.write("\t(node " + str(e[0]) + ' "' + str(d_tmp) + '")\n')
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


def write(lT: LineageTree, fname: str) -> None:
    """Save the lineage tree to an ``.lT`` file.

    The file can be read back with
    [`LineageTree.load`][lineagetree.LineageTree.load].

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    fname : str
        Path to and name of the file to save. The ``.lT`` extension is added
        if missing.
    """
    if os.path.splitext(fname)[-1].upper() != ".LT":
        fname = os.path.extsep.join((fname, "lT"))
    if hasattr(lT, "_protected_predecessor"):
        del lT._protected_predecessor
    if hasattr(lT, "_protected_successor"):
        del lT._protected_successor
    if hasattr(lT, "_protected_time"):
        del lT._protected_time
    with open(fname, "bw") as f:
        pkl.dump(lT, f)
        f.close()
