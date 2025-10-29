from .._core._navigation import (
    change_labels,
    find_leaves,
    get_all_chains_of_subtree,
    get_ancestor_at_t,
    get_ancestor_with_attribute,
    get_available_labels,
    get_chain_of_node,
    get_labelled_ancestor,
    get_predecessors,
    get_subtree_nodes,
    get_successors,
    nodes_at_t,
)

from ._methodize import AutoMethodizeMeta


class NavigationMixin(metaclass=AutoMethodizeMeta):
    """Mixin for tree navigation operations."""

    get_available_labels = get_available_labels
    change_labels = change_labels
    find_leaves = find_leaves
    get_all_chains_of_subtree = get_all_chains_of_subtree
    get_ancestor_at_t = get_ancestor_at_t
    get_ancestor_with_attribute = get_ancestor_with_attribute
    get_chain_of_node = get_chain_of_node
    get_labelled_ancestor = get_labelled_ancestor
    get_predecessors = get_predecessors
    get_subtree_nodes = get_subtree_nodes
    get_successors = get_successors
    nodes_at_t = nodes_at_t
