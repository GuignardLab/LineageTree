from __future__ import annotations

import pickle
from collections.abc import Iterable
from typing import TYPE_CHECKING

from ..tree_approximation import TreeApproximationTemplate

if TYPE_CHECKING:
    from ..lineage_tree import LineageTree


def create_links_and_chains(
    lT: LineageTree,
    roots: int | Iterable | None = None,
    end_time: int | None = None,
) -> dict[str, dict]:
    """Generates a dictionary containing all the edges (from start of lifetime to end not the intermediate timepoints)
      of a subtree spawned by node/s and their duration


    Parameters
    ----------
    lT : LineageTree
        The LineageTree that the user is working on
    roots : int or Iterable, optional
        The root/s from which the tree/s will be generated, if 'None' all the roots will be selected.
    end_time : int, optional
        The last timepoint to be considered, if 'None' the last timepoint of the dataset (t_e) is considered, by default None.

    Returns
    -------
    dict mapping str to set or dict mapping int to list or int
        A dictionary that contains:
            - "links": The dictionary that contains the hierarchy of the nodes (only start and end of each chain)
            - "times": The time distance between the start and the end of a chain
            - "roots": The roots used
    """
    if roots is None:
        to_do = set(lT.roots)
    elif isinstance(roots, Iterable):
        to_do = set(roots)
    else:
        to_do = {int(roots)}
    if end_time is None:
        end_time = lT.t_e
    times = {}
    links = {}
    while to_do:
        curr = to_do.pop()
        cyc = lT.get_successors(curr, end_time=end_time)
        if cyc[-1] != curr or lT.time[cyc[-1]] <= end_time:
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
) -> dict[int, list[float]] | None:
    """Calculates the position of each node on the tree graph.

    Parameters
    ----------
    lnks_tms : dict
         a dictionary created by create_links_and_chains.
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
    dict mapping int to list of float
        Provides a dictionary that contains the id of each node as keys and its 2-d position on the
        tree graph as values.
        If the root requested does not exists, None is then returned
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


def convert_style_to_number(
    style: str | TreeApproximationTemplate,
    downsample: int | None,
) -> int:
    """Converts tree_style and downsampling to a single number.

    Parameters
    ----------
    style : str
        the tree style
    downsample : int
        the downsampling factor

    Returns
    -------
    int
        A number which serves as ID if the tree style and downsampling used.
    """
    style_dict = {
        "full": 0,
        "simple": -1,
        "normalized_simple": -2,
        "mini": -1000,
    }
    if style == "downsampled" and downsample is not None:
        return downsample
    elif not isinstance(style, str) and issubclass(
        style, TreeApproximationTemplate
    ):
        return hash(style.__name__)
    else:
        return style_dict[style]


class CompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "LineageTree.lineageTree" and name == "lineageTree":
            from lineagetree import LineageTree

            return LineageTree
        return super().find_class(module, name)
