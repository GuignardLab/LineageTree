# Initializing a LineageTree and accessing attributes

A LineageTree represents nodes and their lineage relationships over time.
At minimum, you must provide node IDs and the relationships between them.
## Minimum Required Input

You must specify the lineage relationships using **either**:

- a **successor dictionary**, {node: (node1,node2)} where node is the predecessor of both node1 and node2
  
- a **predecessor dictionary**, {node: (node1)} where node is the successor of node1 (For lineages the length of each value can be 1 or 0 

  The user may decide if they will used a successor dictionary or and predecessor dictionary. 
```python
lT = LineageTree(successor = successor_dict)
### OR ###
lT = LineageTree(predecessor = predecessor_dict)
```
Optional Inputs

In addition to the minimum input, you may provide extra node properties. These fall into two categories: common and custom (unexpected) attributes.

- Common: These properties are usually expected from a dataset and if they exist they will be taken into account for byt the object.
    1. **time**: A dictionary mapping each node to a timepoint.The timepoints must respect the lineage hierarchy, meaning if `n1` resides in `t` its suceessor `n2` resides in `t+1`. Alternatively the user may set a starting time using starting time= t
    -  **pos**: The spatial coordinates of the dataset's nodes.
    -  **labels**: A dictionary assigning labels to nodes, there is no need to label every node in a chain just the first one is enough. (Add docs about label sets) 
    -  **name**: The name of the dataset

```python
lT = LineageTree(successor = successor_dict,
labels = labels_dict,
pos=pos_dict,
time = time_dict)
```
         
- **Custom**: These properties are specific to a given experiment or dataset and are **not interpreted or processed internally** by the `LineageTree` object. This means that if the user performs operations that modify the dataset, any required updates to these custom properties must be handled manually. The user may define multiple custom attributes when constructing a `LineageTree`, such as dictionaries containing gene expression values per node, analysis results, or even non-dictionary attributes that should be stored within the object.

```python
properties = {dict_that_contains_multiple_attributes}
lT = LineageTree(successor = successor_dict,
labels = labels_dict,
pos=pos_dict,
time = time_dict,
**{properties})
```
To access any of the attributes the user may just:
```python
lT.time
lT.pos
lT.labels
lT.name
## OR ###
lT.name_of_custom_attribute 
```