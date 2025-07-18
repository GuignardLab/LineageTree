from .lineage_tree import LineageTree
from ._measure._dynamic_time_warping import (
    calculate_dtw,
)
from ._io._loaders import (
    load,
)
from ._basics._modifier import (
    _add_node,
    add_chain,
    add_root,
    get_next_id,
    modifier,
    remove_nodes,
)
from ._basics._navigation import (
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
from ._basics._plot import (
    _create_dict_of_plots,
    draw_tree_graph,
    plot_all_lineages,
    plot_dtw_heatmap,
    plot_dtw_trajectory,
    plot_subtree,
)
from ._measure._spatial import (
    compute_k_nearest_neighbours,
    compute_spatial_density,
    compute_spatial_edges,
    get_gabriel_graph,
    get_idx3d,
)
from ._measure._uted import (
    clear_comparisons,
    labelled_mappings,
    norm_dict,
    plot_tree_distance_graphs,
    unordered_tree_edit_distance,
    unordered_tree_edit_distances_at_time_t,
)
from ._io._writers import (
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
LineageTree.clear_comparisons = clear_comparisons
LineageTree.labelled_mappings = labelled_mappings
LineageTree.norm_dict = norm_dict
LineageTree.unordered_tree_edit_distances_at_time_t = (
    unordered_tree_edit_distances_at_time_t
)
LineageTree.unordered_tree_edit_distance = unordered_tree_edit_distance
LineageTree.plot_tree_distance_graphs = plot_tree_distance_graphs

# Plot functions
LineageTree._create_dict_of_plots = _create_dict_of_plots
LineageTree.draw_tree_graph = draw_tree_graph
LineageTree.plot_all_lineages = plot_all_lineages
LineageTree.plot_dtw_heatmap = plot_dtw_heatmap
LineageTree.plot_dtw_trajectory = plot_dtw_trajectory
LineageTree.plot_subtree = plot_subtree

# DTW functions
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
