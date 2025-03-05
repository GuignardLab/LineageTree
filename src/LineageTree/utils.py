
from LineageTree import lineageTree


def create_links_and_cycles(lT: lineageTree, roots=None) -> dict[str, dict]:
    """Generates a dictionary containing all the edges (from start of lifetime to end not the intermediate timepoints)
      of a subtree spawned by node/s and their duration


    Parameters
    ----------
    lT : lineageTree
        The lineagetree that the user is working on
    roots : _type_, optional
        The root/s from which the tree/s will be generated, by default None

    Returns
    -------
    dict[str,dict]
        Returns a dictionary that contains 3 dictionaries the "links" ( contains all the edges) the "times" (contains all lifetime durations)
        and "roots" (contains the roots.).
    """
    if roots is None:
        to_do = set(lT.roots)
    elif isinstance(roots, list):
        to_do = set(roots)
    else:
        to_do = {int(roots)}
    times = {}
    links = {}
    while to_do:
        curr = to_do.pop()
        cyc = lT.get_successors(curr)
        last = cyc[-1]
        times[curr] = len(cyc)
        if last != curr:
            links[curr] = [last]
        else:
            links[curr] = []
        succ = lT._successor.get(last)
        if succ:
            times[cyc[-1]] = 0
            to_do.update(succ)
            links[last] = succ
    return {"links": links, "times": times, "root": roots}


def hierarchical_pos(
    lnks_tms: dict, root, width=1000, vert_gap=2, xcenter=0, ycenter=0
) -> dict[int, list[int]]:
    """Calculates the position of each node on the tree graph.

    Parameters
    ----------
    lnks_tms : dict
         a dictionary created by create_links_and_cycles.
    root : _type_
        The id of the node, usually it exists inside lnks_tms dictionary, however you may use your own root.
    width : int, optional
        Max width, will not change the graph but interacting with the graph takes this distance into account, by default 1000
    vert_gap : int, optional
        How far downwards each timepoint will go, by default 2
    xcenter : int, optional
        Where the root will be placed on the x axis, by default 0
    ycenter : int, optional
        Where the root will be placed on the y axis, by default 0

    Returns
    -------
    dict[int, list[int]]
        Provides a dictionary that contains the id of each node as keys and its 2-d position on the
                                tree graph as values.
    """
    to_do = [root]
    if root not in lnks_tms["times"]:
        return None
    pos_node = {root: [xcenter, ycenter]}
    prev_width = {root: width / 2}
    while to_do:
        curr = to_do.pop()
        succ = lnks_tms["links"].get(curr, [])
        if len(succ) == 0:
            continue
        elif len(succ) == 1:
            pos_node[succ[0]] = [
                pos_node[curr][0],
                pos_node[curr][1]
                - lnks_tms["times"].get(curr, 0)
                + min(vert_gap, lnks_tms["times"].get(curr, 0)),
            ]
            to_do.extend(succ)
            prev_width[succ[0]] = prev_width[curr]
        elif len(succ) == 2:
            pos_node[succ[0]] = [
                pos_node[curr][0] - prev_width[curr] / 2,
                pos_node[curr][1] - vert_gap,
            ]
            pos_node[succ[1]] = [
                pos_node[curr][0] + prev_width[curr] / 2,
                pos_node[curr][1] - vert_gap,
            ]
            to_do.extend(succ)
            prev_width[succ[0]], prev_width[succ[1]] = (
                prev_width[curr] / 2,
                prev_width[curr] / 2,
            )
    return pos_node
