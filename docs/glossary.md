
This tool may interest individuals with and without a background in computer science, so all users need to become familiar with the nomenclature used by LineageTree. To that end, this glossary defines the terms used throughout the project.


- ### Trees
 A tree is a hierarchical acyclic graph that contains nodes and edges, lineageTree works with at most 2 successors for one node. To create a demo tree the user can call 
```python
from lineagetree import LineageTree
lT = LineageTree(successors= {i:[i+1] for i in range(10)})
```
 ![Image for glossary](./images/glossary_image.png)
> The differences between a biological lineage and a LineageTree lineage 


- ### Nodes
 The smallest part of a tree, a point that may or may not connect to others. The set of all the nodes may be accessed by ``` lT.nodes```

- ### Edges
 The lTect that connects 2 nodes, in the case of the tree its also directed (goes one way). The edges may be accessed throug ```lT.edges```

- ### Successors
 The successor of a node n is a node connected with an edge to the node n that exists one level (timepoint) lower than n. The immutable dictionary (MappingProxy) of all the successors may be accessed by ```lT.successor```

- ### Predecessors
 The predecessor of a node n is a node connected to the node that exists one time point higher than n. The immutable dictionary (MappingProxy) of all the predecessors may be accessed by ```lT.predecessor```

- ### Roots
 The root of a tree is a node that has no predecessors. In the case of lineageTree the root has an empty tuple as predecessor. LineageTree allows for many roots to exist in the same file. The set of roots may be accessed by using ```lT.roots```

- ### Leaves
 The leaf of a tree is a node that has no successors. In the case of lineageTree the leaf has an empty tuple as a successor. The set of leaves may be accessed through ```lT.leaves```

- ### Divisions
 An event where one node has 2 successors instead of one. There is no direct way to access a division; however, the node that is before a division (or after) is always at the end or start of a chain (except for the cases that the chain contains a leaf or a root)

- ### Chains
 A tree segment, where all nodes are between a division or root and a division or a leaf. To access a chain that contains a specific node `n`, the user may use:``` lT.get_chain_of_node(n) ```
A continuous subset of a chain may be referred to as a subchain or a path, which is not a chain.

- ### Subtree
 Any segment of a tree that is also a tree. Using the function ```lT.get_sub_tree(node)``` will not provide a new lineageTree but a list of all the nodes contained inside the subtree. 

- ### Siblings
 Two nodes that have the same predecessor. Often used for chains where the first node of each chain has the same predecessor as the other one.

- ### Time
 The time point at which the nodes exist. This property concerns time points and is not real time.
 The immutable dictionary (MappingProxy) containing the time information for all the nodes may be accessed through ```lT.times```.

- ### Time resolution
 How long one time point lasts, relevant only for comparisons across lineages. This attribute can be a property or inspected by ```lT.time_resolution```.

