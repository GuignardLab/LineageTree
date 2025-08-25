from collections.abc import Iterable

from LineageTree import lineageTree, tree_approximation


def create_links_and_chains(
    lT: lineageTree,
    roots: int | Iterable | None = None,
    end_time: int | None = None,
) -> dict[str, dict]:
    """Generates a dictionary containing all the edges (from start of lifetime to end not the intermediate timepoints)
      of a subtree spawned by node/s and their duration


    Parameters
    ----------
    lT : lineageTree
        The lineagetree that the user is working on
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
    """Calculates the position of each node on the tree graph with uniform leaf spacing.

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
        tree graph as values. Leaves are uniformly spaced on the x-axis.
        If the root requested does not exists, None is then returned
    """
    if root not in lnks_tms["times"]:
        return None
    
    # First pass: find all leaves and calculate y-positions
    def find_leaves_and_depths(node, current_depth=0):
        """Find all leaves and calculate depths for all nodes."""
        succ = lnks_tms["links"].get(node, [])
        node_depth = current_depth + lnks_tms["times"].get(node, 0)
        
        if not succ:  # This is a leaf
            return [node], {node: node_depth}
        
        all_leaves = []
        all_depths = {node: current_depth}
        
        for child in succ:
            child_leaves, child_depths = find_leaves_and_depths(child, node_depth)
            all_leaves.extend(child_leaves)
            all_depths.update(child_depths)
        
        return all_leaves, all_depths
    
    leaves, depths = find_leaves_and_depths(root)
    
    # Calculate uniform x-positions for leaves
    num_leaves = len(leaves)
    if num_leaves == 1:
        leaf_spacing = 0
        leaf_x_positions = {leaves[0]: xcenter}
    else:
        leaf_spacing = width / (num_leaves - 1)
        leaf_x_positions = {
            leaf: xcenter - width/2 + i * leaf_spacing 
            for i, leaf in enumerate(leaves)
        }
    
    # Second pass: assign positions bottom-up
    pos_node = {}
    
    def assign_positions(node):
        """Assign positions working from leaves up to root."""
        succ = lnks_tms["links"].get(node, [])
        
        if not succ:  # This is a leaf
            pos_node[node] = [
                leaf_x_positions[node], 
                ycenter - depths[node] * vert_gap
            ]
            return
        
        # First assign positions to all children
        for child in succ:
            assign_positions(child)
        
        # Position this node based on its children
        if len(succ) == 1:
            # Single child: place directly above
            pos_node[node] = [
                pos_node[succ[0]][0],
                ycenter - depths[node] * vert_gap
            ]
        else:
            # Multiple children: place at the center of children
            child_x_positions = [pos_node[child][0] for child in succ]
            center_x = sum(child_x_positions) / len(child_x_positions)
            pos_node[node] = [
                center_x,
                ycenter - depths[node] * vert_gap
            ]
    
    assign_positions(root)
    return pos_node


def convert_style_to_number(
    style: str | tree_approximation.TreeApproximationTemplate,
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
        style, tree_approximation.TreeApproximationTemplate
    ):
        return hash(style.__name__)
    else:
        return style_dict[style]
