## LineageTree properties

The core properties of a LineageTree. The user may access them by ```lT.property```.

## LineageTree properties

::: lineagetree._core._properties
    options:
        summary: true
        group_by_category: false
        show_signature_annotations: true
        members:
            - successor
            - predecessor
            - time
            - t_b
            - t_e
            - nodes
            - number_of_nodes
            - depth
            - roots
            - leaves
            - edges
            - labels
            - time_resolution
            - all_chains
            - time_nodes
            - parenting
        show_source: true


## LineageTree navigation functions

Functions to access different data that is available in LineageTree.

Every function below is a method of `LineageTree`, called as
`lT.get_successors(node)`. Follow a link for the full signature.

- [`lT.get_successors`][lineagetree.lineage_tree.LineageTree.get_successors]
- [`lT.get_predecessors`][lineagetree.lineage_tree.LineageTree.get_predecessors]
- [`lT.get_subtree_nodes`][lineagetree.lineage_tree.LineageTree.get_subtree_nodes]
- [`lT.get_all_chains_of_subtree`][lineagetree.lineage_tree.LineageTree.get_all_chains_of_subtree]
- [`lT.get_chain_of_node`][lineagetree.lineage_tree.LineageTree.get_chain_of_node]
- [`lT.nodes_at_t`][lineagetree.lineage_tree.LineageTree.nodes_at_t]
- [`lT.get_ancestor_at_t`][lineagetree.lineage_tree.LineageTree.get_ancestor_at_t]
- [`lT.get_ancestor_with_attribute`][lineagetree.lineage_tree.LineageTree.get_ancestor_with_attribute]
- [`lT.get_shortest_path_and_last_common_ancestor`][lineagetree.lineage_tree.LineageTree.get_shortest_path_and_last_common_ancestor]
- [`lT.find_leaves`][lineagetree.lineage_tree.LineageTree.find_leaves]
- [`lT.get_available_labels`][lineagetree.lineage_tree.LineageTree.get_available_labels]
- [`lT.get_labelled_ancestor`][lineagetree.lineage_tree.LineageTree.get_labelled_ancestor]
- [`lT.change_labels`][lineagetree.lineage_tree.LineageTree.change_labels]
