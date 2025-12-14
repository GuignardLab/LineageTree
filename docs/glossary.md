
This tool may interest individuals with and without a background in computer science, so all users need to become familiar with the nomenclature used by LineageTree. To that end, this glossary defines the terms used throughout the project.

---

## Tree Graphs
 A tree graph is a hierarchical acyclic graph that contains nodes and edges, **LineageTree is a tree graph** that has at most 2 successors for one node. To create a demo tree the user can call
 
```python
from lineagetree import LineageTree
lT = LineageTree(successors= {i:[i+1] for i in range(10)})
```

<div class="split-container">
  <div >

<ul>

  <li>
    <p><strong>Nodes:</strong> The smallest part of a LineageTree, representing a point (usually a cell at a timepoint). All nodes can be accessed with <code>lT.nodes</code>.</p>
  </li>

  <li>
    <p><strong>Edges:</strong> Objects that connect two nodes. In a lineage tree the edges are directed. All edges can be accessed with <code>lT.edges</code>.</p>
  </li>

  <li>
    <p><strong>Successors:</strong> The successor of a node n at time t is the node n' at time t+1 that connects from n. The mapping of all successors can be accessed with <code>lT.successor</code>.</p>
  </li>

  <li>
    <p><strong>Predecessors:</strong> The predecessor of a node n at time t is the node n' at time t−1 that connects into n. The mapping of all predecessors can be accessed with <code>lT.predecessor</code>.</p>
  </li>

  <li>
    <p><strong>Roots:</strong> Nodes with no predecessors. In LineageTree, by deafualt, a root has an empty tuple as predecessor. Multiple roots may exist. All roots can be accessed with <code>lT.roots</code>.</p>
  </li>

  <li>
    <p><strong>Leaves:</strong> Nodes with no successors. In LineageTree, by deafualt, a leaf has an empty tuple as successor. All leaves can be accessed with <code>lT.leaves</code>.</p>
  </li>

  <li>
    <p><strong>Chains:</strong> A tree segment where all nodes lie between a root or division and a division or leaf. The chain containing node <code>n</code> can be accessed with <code>lT.get_chain_of_node(n)</code>. A continuous part of a chain is a subchain or path.</p>
  </li>

  <li>
    <p><strong>Divisions:</strong> Events where one node has two successors instead of one. Divisions are inferred because the node before a division is at the end of a chain.</p>
  </li>

  <li>
    <p><strong>Subtree:</strong> Any tree segment that is itself a tree. The subtree rooted at a given node can be retrieved with <code>lT.get_sub_tree(node)</code>.</p>
  </li>

  <li>
    <p><strong>Siblings:</strong> Two nodes that share the same predecessor. Often refers to the first nodes of chains originating from the same division.</p>
  </li>

  <li>
    <p><strong>Time:</strong> The discrete timepoint at which nodes exist (not real time). All node → time information can be accessed with <code>lT.times</code>.</p>
  </li>

  <li>
    <p><strong>Time resolution:</strong> The duration represented by one timepoint. Useful for comparing different lineages. This value can be accessed via <code>lT.time_resolution</code>.</p>
  </li>

</ul>


  </div>
  <div class="fixed-right">
    <img src="../images/glossary_image.png" alt="My Image"> 
  </div>
</div>