## LineageTree properties

The core properties of a LineageTree. The user may access them by ```lT.property```.

## LineageTree properties

::: lineagetree._core
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

#### ::: lineagetree._core
        options:
            summary: true
            filters:
                - "change_labels"
                - "find_leaves"
                - "get_all_chains_of_subtree"
                - "get_ancestor_at_t"
                - "get_ancestor_with_attribute"
                - "get_available_labels"
                - "get_chain_of_node"
                - "get_labelled_ancestor"
                - "get_predecessors"
                - "get_subtree_nodes"
                - "get_successors"
                - "nodes_at_t"
