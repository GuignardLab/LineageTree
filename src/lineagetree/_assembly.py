from ._core import LineageTree
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

LineageTree._add_node = _add_node
LineageTree.add_chain = add_chain
LineageTree.add_root = add_root
LineageTree.get_next_id = get_next_id
LineageTree.modifier = modifier
LineageTree.remove_nodes = remove_nodes

LineageTree._get_height = _get_height
LineageTree.write = write
LineageTree.write_to_binary = write_to_binary
LineageTree.write_to_svg = write_to_svg
LineageTree.write_to_tlp = write_to_tlp

LineageTree.load = load

LineageTree.get_idx3d = get_idx3d
LineageTree.get_gabriel_graph = get_gabriel_graph
LineageTree.compute_k_nearest_neighbours = compute_k_nearest_neighbours
LineageTree.compute_spatial_edges = compute_spatial_edges
LineageTree.compute_spatial_density = compute_spatial_density

LineageTree.__calculate_distance_of_sub_tree = __calculate_distance_of_sub_tree
LineageTree.__unordereded_backtrace = __unordereded_backtrace
LineageTree.clear_comparisons = clear_comparisons
LineageTree.unordered_tree_edit_distances_at_time_t = (
    unordered_tree_edit_distances_at_time_t
)
LineageTree.unordered_tree_edit_distance = unordered_tree_edit_distance
LineageTree.plot_tree_distance_graphs = plot_tree_distance_graphs
