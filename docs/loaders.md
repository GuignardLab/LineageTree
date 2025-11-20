
### Premade Loaders

Multiple well known formats are ready for import out of the box:

1. **csv** with the format: id, time, z, y, x, id, pred_id, lin_id

- **Astec** data
- **Multiple** C. Elegans formats (add citations here)
- **MaMut** xml files
- **Mastodon** files ( Either the .mastodon file or the 2 CSV files extracted using the User interface)
- **TGMM** data

The user may also decide to import their custom format. The bare minimum inormation needed to create a Lineage tree is the hierarchy of the nodes, which is a Python dictionary of the successors or the predecessors shown in the example:

 ```successor/predecessor : {unique_node_id (int) : [next_unique_node_ids (int)]}

    lT = Lineagetree(successor= successor) 
    ###or### 
    LineageTree(predecessor= predecessor)
 ```

Other attributes that can be used to initiate a lineageTree file apart from the hierarchy dictionary are:

- time: A dictionary that has this format {unique_node_id: time (int)}

- starting_time: If the time dictionary is not given, the algorithm sets the starting point of all roots to the value of this parameter. The default value is set to 0.

- pos: The positions of all the nodes in 3D space. The format is {unique_id: position}.

- **kwargs: Any other dictionary provided during imaging can be loaded into lineageTree. The format is {unique_id: value}

### Custom Loaders

```python

from pathlib import Path
from LineageTree import lineageTree

def template_load(path, name=None):
   """
   Load lineage data from a file and convert it into a LineageTree object.

   Parameters:
   - path (str or Path): Path to the data file.
   - name (str, optional): Name of the tree. Defaults to the filename stem.

   Returns:
   - lineageTree object
   """
   # Step 1: Load and parse data from the file
   data = extract_info_from_file(path)  # <-- Implement this function

   # Step 2: Build the relationship dictionary
   relations = {}
   for unique_id in data:
       related_ids = data.get_descendantsr(unique_id) # <- Implement this function or change the block according to format
       if related_ids:
           relations[unique_id] = related_ids

   # Step 3: Extract optional positional information
   pos = {
       uid: [data.x_of_id(uid), data.y_of_id(uid), data.z_of_id(uid)]
       for uid in data
   }
   time = {uid: data.time_of_id(uid) for uid in data}

   # Optional properties, like labels or others
   properties = {}
   labels = {uid: data.label_of_id(uid) for uid in data}
   if labels:
       properties["_label"] = labels

   for attr in data.get_additional_attributes():  # You can define how this works
       properties[attr] = {uid: data.info_of_id(uid, attr) for uid in data}

   # Step 4: Default name from filename
   if not name:
       name = Path(path).stem

   # If the relationship dictionary is successors
       return lineageTree(successor=relations, time=time, pos=pos, name=name, **properties
   )

   # If the relationship dictionary is predecessors
       return lineageTree(predecessor=relations, time=time, pos=pos, name=name, **properties
   )
```

### API reference for existing loaders

#### ::: lineagetree
            options:
                summary: true
                filters:
                    - "^read_"