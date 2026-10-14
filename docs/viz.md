## Visualising lineages

<p style="text-align: justify;">

Usually, tracked lineages contain hundreds of thousands of nodes, thus calculating each position for the nodes of the tree graph is time-consuming. To solve this problem, we decided to use a model that reduces the number of nodes to a minimum. In this model, we only use the start and end of each chain, while the length of their link would correspond to the time distance between the two nodes, as shown in the next image.
</p>
![viz](./images/2_trees.png)

This way, the whole lineage can be plotted efficiently, even if the second graph is more representative of the truth.

## API Reference

- [`lT.plot_all_lineages`][lineagetree.lineage_tree.LineageTree.plot_all_lineages]
- [`lT.plot_subtree`][lineagetree.lineage_tree.LineageTree.plot_subtree]
- [`lT.plot_chain_histogram`][lineagetree.lineage_tree.LineageTree.plot_chain_histogram]
- [`lT.draw_tree_graph`][lineagetree.lineage_tree.LineageTree.draw_tree_graph]
