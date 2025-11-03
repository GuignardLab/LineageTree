# Introduction to Unordered Tree Edit Distance (UTED)

While visual inspection allows for identifying similarities and differences between lineages, it is insufficient for scientific analysis. Therefore, mathematical tools are required to objectively analyze and compare such structures. This module focuses on finding such patterns across lineages using an unordered tree edit distance (UTED) algorithm developed by [citation]. UTED computes the minimum distance between two trees by transforming one into the other using the fewest possible operations: addition of a node, removal of a node, and matching two nodes (also referred to as substitution). 

UTED is label agnostic, meaning it can compare lineages without needing any prior knowledge of the naming of the cell, meaning that 2 daughter cells are born equal, however this strength comes with a significant drawback: the algorithm must map nodes from one tree to another which makes time consumed scale exponentially to the number of nodes, because there are many possible mappings. This pairing always respects the hierarchical structure of the trees; thus, nodes that exist at a specific depth will only be mapped to nodes of the same depth or greater. 

![uted_explanation](./images/uted_explanation.png)

## Different Tree approximations

To address this computational challenge, we developed 4 approaches that reduce the number of nodes, while preserving accuracy. 2 of these also alter the behaviour of UTED, so it is crucial to understand the mechanisms and implications of each approach:

1. **Normal Tree**: This algorithm is the simplest, as the dataset without changes is used to produce the distance. So the algorithm will either:

    - **Match** a node, where the cost will be 0
    - **Add** a node to either of the trees for a cost of 1
    - **Remove** a node from either of the trees for a cost of 1.

    Normal tree is the most accurate, but it does not scale well with time, meaning that computation time scales exponentially with the number of nodes.

2. **Downsampled Tree**: This algorithm reconstructs the tree but skips nodes every n time points, so the result will be similar to the first tree but with fewer nodes.

    - **Match** a node, where the cost will be 0
    - **Add** a node to either of the trees for a cost of 1 
    - **Remove** a node from either of the trees for a cost of 1.

    Downsampled Tree is slightly less accurate or equal to normal tree, but much faster

3. **Reduced Tree**: This algorithm will reconstruct the trees, but every chain will be replaced with a node of the same length. This approximation completely changes the behaviour of the algorithm, as it will match whole chains with other chains and not just nodes. Such an algorithm may be extremely useful for fast but not precise results, however some biological question may be better answered using this algorithm.

    - **Match** a node, the cost will be the absolute difference of the length of the 2 chains
    - **Add** a node, will add a node of size equal to the length of the existing chain, so the cost will be the value of the length of the chain.
    - **Remove** a node, will remove a node of size equal to the length of the existing chain, so the cost will be the value of the length of the chain.

    This algorithm is extremely lightweight and fast; however, the score is different than the normal tree. Some users may prefer this algorithm due to its capability of comparing chains instead of nodes, making it easily interpretable. Many other developers use the same or similar algorithms to this one [novel metrics, deeplineage citation] 

4. **Bound Reduced Tree**: This algorithm will use the same tree as the reduced tree, however the edit operations are modified so that big chains that have similar length will be easier to be matched together. Has the strengths of reduced tree, but the resulting distance is closer to the that of the normal tree.


    - **Match** a node, the cost will be the absolute difference of the length of the 2 chains divided by the sum of both of them, so the result will be between 0 and 1
    - **Add** a node, will add a node of size 1
    - **Remove** a node, will remove a node of size 1

    This algorithm has all the advantages of reduced tree, but tries to reduce its disadvantages but adding a new way of comparing chains.

![tree_styles](./images/tree_styles.png)

To inspect any style:

```python
from LineageTree import tree_styles
tree_styles.tree_style["simple"].value(parameters)
```

## The need for normalization

Uted produces distances in the range of zero to infinity, meaning that the results are not easily interpretable. For instance, 2 visually similar, large trees may have a difference of 1000 nodes, but 2 small, dissimilar trees may only have a difference of 10 nodes. This makes the results of this algorithm confusing and its application very limited. To enhance the use of this algorithm, we developed a way to normalize these distances so that the results are bound between 0 and 1 using the number of nodes of the trees. Each tree style employs its own way of normalization.

1. **Normal**: The distance of 2 trees is divided by the max or the sum of the number of nodes of the trees used.

2. **Downsampled**: The distance of 2 trees is divided by the max or the sum of the number of nodes of the downsapled tree.

3. **Reduced**: The distance of 2 trees is divided by the max or the sum of the number of nodes of the trees used before being converted to the reduced form ( Equal to the sum of all the node sizes after the conversion).

4.**Normalized Reduced**: The distance of 2 trees is divided by the max or the sum of the number of chains that exist in each tree.

Normalization in general

- Max: This way of normalization produces values that are well distributed along the [0,1] axis, however, trees that show extreme disimilarity may produce values slightly bigger than one (1.16 max). However, most of the time, the compared trees are not that different, so it is a rare occasion.

- Sum: This way of normalization bounds the values between [0,1];  however, most trees will be distributed along the [0,0.5] axis, meaning it's more difficult to distinguish the similar trees from the dissimilar. 

Template to create new styles:

```python
from LineageTree import lineageTree
from LineageTree.tree_styles import abstract_trees

class tree_style_template(abstract_trees):

   ### The user has to implement these methods

   def get_tree(self)-> tuple[dict[int, list], dict[int, int | float]]:
       out_dict = {}
       times_or_other_attribute = {}
       ... # If time_scale exists for cross embryo comparison use it here
       return (out_dict, # The hierarchy of the nodes should be a dict(unique_node:[successors])
               times_or_other_attribute) # An attribute that may be used for comparisons.
  
   def _edist_format(self, adj_dict): # This methods seldom needs changing, creates the adjacency matrix 
       return super()._edist_format(adj_dict)# and provides the corresponding nodes that between lt and the tree style
  
   def delta(self, x, y, corres1, corres2, times1, times2)-> int|float:
       if x is None and y is None:
           return 0
       if x is None: # Cost for removing a node
           return times2[corres2[y]]
       if y is None: # Cost for adding a node
           return times1[corres1[x]]
       len_x = times1[corres1[x]]
       len_y = times2[corres2[y]]
       return abs(len_x - len_y) # Cost for matching a node
  
   def get_norm(self)->int|float:
       ... # how will the normalization work, should return a function that
       ... # calculates the norm for every subtree of the starting tree
       return value
  
   @staticmethod
   def handle_resolutions(
       time_resolution1: float | int,
       time_resolution2: float | int,
       gcd, #the greatest common divisor of all the time resolutions of datasets in the manager.
       downsample: int,
   ) -> tuple[int | float, int | float]: #Needed for cross embryo analysis
       ...
       return (time_resolution_fix_for_dataset_1   # This list will be used as the time scale
               ,time_resolution_fix_for_dataset_2) # parameter when comparing.
```

## Testing the approximations

This section is focused on showcasing the advantages and disadvantages of the tree approximations shown in the previous section. 

![synthetic_trees](./images/synthetic_trees.png)

<!-- ### API reference

#### :::lineagetree.measure.unordered_tree_edit_distance
#### :::lineagetree.measure.unordered_tree_edit_distances_at_time_t
#### :::lineagetree.measure.clear_comparisons -->
## API reference for uted

### ::: lineagetree.measure
    options:
      filters:
        - "^unordered_tree_edit_distance$"
        - "^unordered_tree_edit_distances_at_time_t$"
        - "^clear_comparisons$"


## Using the Matching Component of UTED

UTED calculates distances between two tree graphs by matching their nodes. Extracting the matched nodes provides important information, particularly for labeled datasets. For example, when calculating the distance between two embryos, we can identify which clone or sublineage in one embryo corresponds to a clone or sublineage in the other. To support this, UTED includes functions specifically designed to find the matching between two lineages.

#### API

#### ::: lineagetree.measure
    options:
      filters:
        - "^labelled_mappings$"



### Tree distance Graphs

<p style="text-align: justify;">

The distance value is a useful metric for quantifying the similarity between two lineages, but it does not provide detailed information about which sublineages are similar or different. To address this, UTED generates tree distance graphs by leveraging the matched pairs produced during the mapping process.

In these graphs, each matched chain is colored according to the value of its subtree, providing a visual representation of mapping quality. Tree distance graphs reveal two important aspects:

- Variance: The color spectrum highlights the distances of mapped chains. Colors representing good mappings indicate low variance during development.

- Gain or Loss of Function: Unmapped regions indicate lineages that exist in one tree but not the other. This may reflect biological differences, such as the emergence or loss of a lineage, or it may point to data quality issues.
</p>
![add_an_example](./images/image39151.png)


#### API

#### ::: lineagetree.measure
    options:
      filters:
        - "^plot_tree_distance_graphs$"