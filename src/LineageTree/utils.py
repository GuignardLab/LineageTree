import csv
import random
import warnings

import networkx as nx

from LineageTree import lineageTree

try:
    import motile
except ImportError:
    warnings.warn(
        "No motile installed therefore you will not be able to produce links with motile."
    )


def hierarchy_pos(
    G,
    a,
    root=None,
    width=2000.0,
    vert_gap=0.5,
    vert_loc=0,
    xcenter=0,
):
    """
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.


    #The graph represents the lifetimes of cells, so there is no new point for each timepoint.
    #Each lifetime is represented by length.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.


    width: horizontal space allocated for this branch - avoids overlap with other branches


    vert_gap: gap between levels of hierarchy


    vert_loc: vertical location of root


    xcenter: horizontal location of root
    """
    if not nx.is_tree(G):
        raise TypeError(
            "cannot use hierarchy_pos on a graph that is not a tree"
        )

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(
                iter(nx.topological_sort(G))
            )  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def lengths(cell):
        succ = a.successor.get(cell, [])
        if len(succ) < 2:
            if list(G.neighbors(cell)) == []:
                return 0
            if list(G.neighbors(cell))[0] in a.get_cycle(cell):
                return (
                    len(a.get_successors(cell))
                    - len(a.get_successors(list(G.neighbors(cell))[0]))
                    - 1
                )
            return len(a.get_successors(cell))
        else:
            return 0.7

    def _hierarchy_pos(
        G,
        root,
        width=2.0,
        a=a,
        vert_gap=0.5,
        vert_loc=0,
        xcenter=0.5,
        pos=None,
        parent=None,
    ):
        """
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        """
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        elif not a.predecessor.get(a.get_predecessors(root)[0]):
            vert_loc = vert_loc - len(a.get_predecessors(root))
            pos[root] = (xcenter, vert_loc)
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))

        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(
                    G,
                    child,
                    width=dx,
                    vert_gap=lengths(child),
                    vert_loc=vert_loc - vert_gap,
                    xcenter=nextx,
                    pos=pos,
                    a=a,
                    parent=root,
                )
        return pos

    return _hierarchy_pos(G, root, width, a, vert_gap, vert_loc, xcenter)


def to_motile(
    lT: lineageTree, crop: int = None, max_dist=200, max_skip_frames=1
):
    fmt = nx.DiGraph()
    if not crop:
        crop = lT.t_e
    # time_nodes = [
    for time in range(crop):
        #     time_nodes += lT.time_nodes[time]
        # print(time_nodes)
        for time_node in lT.time_nodes[time]:
            fmt.add_node(
                time_node,
                t=lT.time[time_node],
                pos=lT.pos[time_node],
                score=1,
            )
            # for suc in lT.successor:
            #     fmt.add_edge(time_node, suc, **{"score":0})

    motile.add_cand_edges(fmt, max_dist, max_skip_frames=max_skip_frames)

    return fmt


def write_csv_from_lT_to_lineaja(
    lT, path_to, start: int = 0, finish: int = 300
):
    csv_dict = {}
    for time in range(start, finish):
        for node in lT.time_nodes[time]:
            csv_dict[node] = {"pos": lT.pos[node], "t": time}
    with open(path_to, "w", newline="\n") as file:
        fieldnames = [
            "time",
            "positions_x",
            "positions_y",
            "positions_z",
            "id",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for node in csv_dict:
            writer.writerow(
                {
                    "time": csv_dict[node]["t"],
                    "positions_z": csv_dict[node]["pos"][0],
                    "positions_y": csv_dict[node]["pos"][1],
                    "positions_x": csv_dict[node]["pos"][2],
                    "id": node,
                }
            )


def postions_of_nx(lt, graphs):
    """Calculates the positions of the Lineagetree to be plotted.

    Args:
        graphs (nx.Digraph): Graphs produced by export_nx_simple_graph

    Returns:
        pos (list): The positions of the nodes of the graphs for plotting
    """
    pos = {}
    for i in range(len(graphs)):
        pos[i] = hierarchy_pos(
            graphs[i],
            lt,
            root=[n for n, d in graphs[i].in_degree() if d == 0][0],
        )
    return pos
