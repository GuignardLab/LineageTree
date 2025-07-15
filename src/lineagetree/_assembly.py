from ._core import LineageTree
from ._dynamic_time_warping import (
    __calculate_diag_line,
    __dp,
    __interpolate,
    __rigid_transform_3D,
    calculate_dtw,
)
from ._loaders import (
    load,
)
from ._modifier import (
    _add_node,
    add_chain,
    add_root,
    get_next_id,
    modifier,
    remove_nodes,
)
from ._navigation import (
    find_leaves,
    get_all_chains_of_subtree,
    get_ancestor_at_t,
    get_ancestor_with_attribute,
    get_chain_of_node,
    get_labelled_ancestor,
    get_predecessors,
    get_subtree,
    get_subtree_nodes,
    get_successors,
    nodes_at_t,
)
from ._plot import (
    __plot_2d,
    __plot_edges,
    __plot_nodes,
    _create_dict_of_plots,
    draw_tree_graph,
    plot_all_lineages,
    plot_dtw_heatmap,
    plot_dtw_trajectory,
    plot_subtree,
)
from ._spatial import (
    compute_k_nearest_neighbours,
    compute_spatial_density,
    compute_spatial_edges,
    get_gabriel_graph,
    get_idx3d,
)
from ._uted import (
    __calculate_distance_of_sub_tree,
    __unordereded_backtrace,
    clear_comparisons,
    labelled_mappings,
    norm_dict,
    plot_tree_distance_graphs,
    unordered_tree_edit_distance,
    unordered_tree_edit_distances_at_time_t,
)
from ._writers import (
    _get_height,
    write,
    write_to_binary,
    write_to_svg,
    write_to_tlp,
)

# Modifier functions
LineageTree._add_node = _add_node
LineageTree.add_chain = add_chain
LineageTree.add_root = add_root
LineageTree.get_next_id = get_next_id
LineageTree.modifier = modifier
LineageTree.remove_nodes = remove_nodes

# Writer functions
LineageTree._get_height = _get_height
LineageTree.write = write
LineageTree.write_to_binary = write_to_binary
LineageTree.write_to_svg = write_to_svg
LineageTree.write_to_tlp = write_to_tlp

# Loader function
LineageTree.load = load

# Spatial functions
LineageTree.get_idx3d = get_idx3d
LineageTree.get_gabriel_graph = get_gabriel_graph
LineageTree.compute_k_nearest_neighbours = compute_k_nearest_neighbours
LineageTree.compute_spatial_edges = compute_spatial_edges
LineageTree.compute_spatial_density = compute_spatial_density

# Uted functions
LineageTree.__calculate_distance_of_sub_tree = __calculate_distance_of_sub_tree
LineageTree.__unordereded_backtrace = __unordereded_backtrace
LineageTree.clear_comparisons = clear_comparisons
LineageTree.labelled_mappings = labelled_mappings
LineageTree.norm_dict = norm_dict
LineageTree.unordered_tree_edit_distances_at_time_t = (
    unordered_tree_edit_distances_at_time_t
)
LineageTree.unordered_tree_edit_distance = unordered_tree_edit_distance
LineageTree.plot_tree_distance_graphs = plot_tree_distance_graphs

# Plot functions
LineageTree.__plot_2d = __plot_2d
LineageTree.__plot_edges = __plot_edges
LineageTree.__plot_nodes = __plot_nodes
LineageTree._create_dict_of_plots = _create_dict_of_plots
LineageTree.draw_tree_graph = draw_tree_graph
LineageTree.plot_all_lineages = plot_all_lineages
LineageTree.plot_dtw_heatmap = plot_dtw_heatmap
LineageTree.plot_dtw_trajectory = plot_dtw_trajectory
LineageTree.plot_subtree = plot_subtree

# DTW functions
LineageTree.__calculate_diag_line = __calculate_diag_line
LineageTree.__dp = __dp
LineageTree.__interpolate = __interpolate
LineageTree.__rigid_transform_3D = __rigid_transform_3D
LineageTree.calculate_dtw = calculate_dtw

# Navigation functions
LineageTree.find_leaves = find_leaves
LineageTree.get_all_chains_of_subtree = get_all_chains_of_subtree
LineageTree.get_ancestor_at_t = get_ancestor_at_t
LineageTree.get_ancestor_with_attribute = get_ancestor_with_attribute
LineageTree.get_chain_of_node = get_chain_of_node
LineageTree.get_labelled_ancestor = get_labelled_ancestor
LineageTree.get_predecessors = get_predecessors
LineageTree.get_subtree = get_subtree
LineageTree.get_subtree_nodes = get_subtree_nodes
LineageTree.get_successors = get_successors
LineageTree.nodes_at_t = nodes_at_t
