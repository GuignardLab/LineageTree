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
    """Build a compact version of the subtrees spawned by some nodes.

    Each chain is represented only by its first and last nodes (not the
    intermediate time points) together with its duration. This is the input
    of the plotting functions.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    roots : int or Iterable of int, optional
        The node(s) spawning the subtree(s). If None, all the roots are used.
    end_time : int, optional
        The last time point to be considered. If None, the last time point of
        the dataset (``t_e``) is used.

    Returns
    -------
    dict
        A dictionary with three keys:

        - ``"links"``: maps the first node of each chain to its last node,
          and the last node of each chain to the first nodes of the next
          chains (an empty list for leaves);
        - ``"times"``: maps the first node of each chain to its number of
          nodes, and the last node of a dividing chain to 0;
        - ``"root"``: the ``roots`` argument, unchanged.
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
            if succ and lT.time[succ[0]] < end_time:
                times[cyc[-1]] = 0
                to_do.update(succ)
                links[last] = succ
            else:
                links[last] = []
    return {"links": links, "times": times, "root": roots}


def _find_leaves_and_depths_iterative(
    lnks_tms: dict, root: int
) -> tuple[list[int], dict[int, int]]:
    """Find all leaves and calculate depths for all nodes using iterative approach.

    Parameters
    ----------
    lnks_tms : dict
        A dictionary created by ``create_links_and_chains``.
    root : int
        The id of the root node.

    Returns
    -------
    leaves : list of int
        List of leaf node ids.
    depths : dict mapping int to int
        Dictionary mapping all node ids to their depth in the tree.
    """
    leaves = []
    depths = {}

    times = lnks_tms["times"]
    links = lnks_tms["links"]

    # Stack for DFS: (node, parent_depth)
    stack = [(root, 0)]

    while stack:
        parent_node, parent_depth = stack.pop()
        depths[parent_node] = parent_depth
        succ = links.get(parent_node, [])

        if not succ:  # This is a leaf
            leaves.append(parent_node)
        else:
            if (
                len(succ) == 1
            ):  # in this case, times[parent_node] is equal to the length of the chain
                child_depth = parent_depth + times[parent_node] - 1
            else:  # in this case, times[parent_node] is 0
                child_depth = parent_depth + 1
            # Add children to stack (reverse order to maintain left-to-right traversal)
            for child in reversed(succ):
                stack.append((child, child_depth))

    return leaves, depths


def _calculate_leaf_positions(
    leaves: list[int], width: int, xcenter: int
) -> dict[int, float]:
    """Calculate uniformly spaced x-positions for leaf nodes.

    Leaves are distributed symmetrically around ``xcenter`` with a total
    spread of ``width``.

    Parameters
    ----------
    leaves : list of int
        Ordered list of leaf node ids.
    width : int
        Total horizontal span available for the leaves.
    xcenter : int
        Horizontal centre position.

    Returns
    -------
    dict mapping int to float
        Mapping from leaf node id to its x-coordinate.
    """
    num_leaves = len(leaves)
    if num_leaves == 1:
        return {leaves[0]: xcenter}

    leaf_spacing = width / (num_leaves - 1)
    return {
        leaf: xcenter - width / 2 + i * leaf_spacing
        for i, leaf in enumerate(leaves)
    }


def _assign_positions_iterative(
    lnks_tms: dict,
    root: int,
    depths: dict[int, int],
    leaf_x_positions: dict[int, float],
    vert_gap: int,
    ycenter: int,
) -> dict[int, list[float]]:
    """Assign 2D positions to all nodes using an iterative post-order traversal.

    Leaves are placed at pre-computed x-positions (``leaf_x_positions``).
    Interior nodes with a single child are placed directly above that child.
    Interior nodes with multiple children are centred over their children.

    Parameters
    ----------
    lnks_tms : dict
        Dictionary produced by ``create_links_and_chains``, containing
        ``'links'`` and ``'times'`` sub-dicts.
    root : int
        Id of the root node to start placement from.
    depths : dict mapping int to int
        Pre-computed depth (in display units) of each node, as returned by
        ``_find_leaves_and_depths_iterative``.
    leaf_x_positions : dict mapping int to float
        Pre-computed x-coordinates of each leaf node, as returned by
        ``_calculate_leaf_positions``.
    vert_gap : int
        Vertical distance (in display units) between successive depth levels.
    ycenter : int
        Y-coordinate of the root node.

    Returns
    -------
    dict mapping int to list of float
        Mapping from node id to ``[x, y]`` position.
    """
    pos_node = {}

    # First pass: build parent-child relationships and find processing order
    children_map = lnks_tms["links"]

    # Reverse-order traversal using two stacks
    stack1 = [root]
    stack2 = []

    # This while loop stores nodes in stack2 so that children are processed before parents
    while stack1:
        node = stack1.pop()
        stack2.append(node)
        stack1.extend(children_map.get(node, []))

    # Process nodes in reverse-order (children before parents)
    while stack2:
        node = stack2.pop()
        succ = children_map.get(node, [])

        if not succ:  # This is a leaf
            pos_node[node] = [
                leaf_x_positions[node],
                ycenter - depths[node] * vert_gap,
            ]
        elif len(succ) == 1:
            # Single child: place directly above
            pos_node[node] = [
                pos_node[succ[0]][0],
                ycenter - depths[node] * vert_gap,
            ]
        else:
            # Multiple children: place at center of children
            child_x_positions = [pos_node[child][0] for child in succ]
            center_x = sum(child_x_positions) / len(child_x_positions)
            pos_node[node] = [center_x, ycenter - depths[node] * vert_gap]

    return pos_node


def hierarchical_pos(
    lnks_tms: dict, root, width=1000, vert_gap=2, xcenter=0, ycenter=0
) -> dict[int, list[float]] | None:
    """Compute the 2D position of each node of a tree graph.

    Leaves are spread uniformly along the x-axis, and each time point moves
    the nodes ``vert_gap`` further down.

    Parameters
    ----------
    lnks_tms : dict
        A dictionary created by ``create_links_and_chains``.
    root : int
        The node to start from, usually a key of ``lnks_tms["times"]``.
    width : int, default=1000
        Horizontal span of the leaves. It does not change the shape of the
        graph, only its coordinates.
    vert_gap : int, default=2
        Vertical distance between two consecutive time points.
    xcenter : int, default=0
        Position of the root along the x-axis.
    ycenter : int, default=0
        Position of the root along the y-axis.

    Returns
    -------
    dict of {int: list of float} or None
        Maps each node id to its ``[x, y]`` position on the tree graph, or
        None if ``root`` is not in ``lnks_tms["times"]``.
    """
    if root not in lnks_tms["times"]:
        return None

    # Find all leaves and calculate depths
    leaves, depths = _find_leaves_and_depths_iterative(lnks_tms, root)

    # Calculate uniform x-positions for leaves
    leaf_x_positions = _calculate_leaf_positions(leaves, width, xcenter)

    # Assign positions using iterative approach
    pos_node = _assign_positions_iterative(
        lnks_tms, root, depths, leaf_x_positions, vert_gap, ycenter
    )

    return pos_node


def convert_style_to_number(
    style: str | TreeApproximationTemplate,
    downsample: int | None,
) -> int:
    """Convert a tree style and downsampling factor to a single number.

    Parameters
    ----------
    style : str or TreeApproximationTemplate
        The tree style.
    downsample : int or None
        The downsampling factor.

    Returns
    -------
    int
        A number that serves as an identifier for the tree style and
        downsampling used.
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
    """A pickle unpickler that handles legacy module paths.

    When a ``LineageTree`` object was pickled under the old module path
    ``LineageTree.lineageTree`` (pre-v2.0), normal unpickling would fail with
    a ``ModuleNotFoundError``. This subclass intercepts the import and
    redirects it to the current path.
    """

    def find_class(self, module, name):
        """Resolve a pickled class reference, remapping legacy module paths.

        Parameters
        ----------
        module : str
            Module path stored in the pickle stream.
        name : str
            Class name stored in the pickle stream.

        Returns
        -------
        type
            The resolved class object.
        """
        if module == "LineageTree.lineageTree" and name == "lineageTree":
            from lineagetree import LineageTree

            return LineageTree
        return super().find_class(module, name)
