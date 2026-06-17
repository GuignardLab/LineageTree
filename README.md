# LineageTree

[![PyPI version](https://badge.fury.io/py/lineagetree.svg)](https://badge.fury.io/py/lineagetree)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A Python library for importing, analyzing, and visualizing cell lineage trees.
Built for developmental biology workflows, it supports data from the most common cell-tracking algorithms and provides tools for spatial analysis, tree comparison, and visualization.

Full documentation: [guignardlab.github.io/LineageTree](https://guignardlab.github.io/LineageTree/)

---

## Features

- **Multi-format I/O** — read from TGMM, MaMuT/TrackMate, Mastodon, ASTEC, SVF, SWC, BMF, and CSV; export to pickle (`.lT`), SVG, or Tulip (`.tlp`)
- **Tree analysis** — unordered tree edit distance (UTED), dynamic time warping (DTW), chain extraction, depth computation
- **Spatial analysis** — KD-tree indexing, Gabriel graphs, k-nearest neighbours, spatial density
- **Trajectory manipulation** — smooth trajectories, stabilise positions across time points
- **Visualization** — lineage plots, subtree views, DTW heatmaps, chain histograms

---

## Installation

```shell
pip install lineagetree
```

For the development version:

```shell
pip install git+https://github.com/GuignardLab/LineageTree
```

Or from a local clone:

```shell
pip install .
```

---

## Quick start

### Loading a tree

```python
from lineagetree import LineageTree

# From a saved .lT file
lT = LineageTree.load("path/to/file.lT")

# Inspect basic properties
print(lT.nodes)    # frozenset of all node ids
print(lT.roots)    # frozenset of root nodes
print(lT.leaves)   # frozenset of leaf nodes
print(lT.t_b, lT.t_e)  # first and last time points
```

### Reading from tracking software

```python
from lineagetree import (
    read_from_ASTEC,
    read_from_mamut_xml,
    read_from_mastodon,
    read_from_mastodon_csv,
    read_from_tgmm_xml,
    read_from_swc,
)

# ASTEC (Guignard, Fiuza et al. 2020)
lT = read_from_ASTEC("path/to/ASTEC.pkl")

# MaMuT / TrackMate (Wolff et al. 2018 / Tinevez et al. 2017)
lT = read_from_mamut_xml("path/to/MaMuT.xml")

# Mastodon — binary format
lT = read_from_mastodon("path/to/file.mastodon")

# Mastodon — CSV export
lT = read_from_mastodon_csv(["path/to/nodes.csv", "path/to/links.csv"])

# TGMM (Amat et al. 2014) — one XML per time point
lT = read_from_tgmm_xml("path/to/single_time_file{t:04d}.xml", tb=0, te=500)

# SWC morphology files
lT = read_from_swc("path/to/morphology.swc")
```

### Building a tree programmatically

```python
from lineagetree import LineageTree

# From a successor dictionary
lT = LineageTree(
    successor={0: [1, 2], 1: [3], 2: [], 3: []},
    time={0: 0, 1: 1, 2: 1, 3: 2},
    pos={0: [0, 0, 0], 1: [1, 0, 0], 2: [-1, 0, 0], 3: [1, 1, 0]},
    name="example",
)
```

### Tree navigation

```python
# Traverse successors / predecessors
lT.get_successors(node)
lT.get_predecessors(node)

# All nodes in a subtree rooted at `node`
lT.get_subtree_nodes({node})

# Unbroken chains (segments between division events)
lT.all_chains

# Nodes at a specific time point
lT.nodes_at_t(t)
```

### Saving

```python
lT.write("output.lT")          # pickle
lT.write_to_svg("tree.svg")    # SVG visualization
lT.write_to_tlp("tree.tlp")    # Tulip graph format
```

### Tree comparison

```python
# Unordered tree edit distance between two subtrees
dist = lT.unordered_tree_edit_distance(node_a, node_b)

# Dynamic time warping on trajectories
score, path = lT.dtw(node_a, node_b)
```

### Spatial analysis

```python
# KD-tree index at time t
idx = lT.idx3d(t)

# k nearest neighbours of a node
neighbours = lT.k_nearest_neighbours(node, t, k=5)

# Gabriel graph at time t
g = lT.gabriel_graph(t)

# Local cell density
density = lT.spatial_density(t)
```

### Visualization

```python
lT.plot_subtree(root_node)
lT.plot_all_lineages()
lT.draw_tree_graph()

# DTW visualizations
lT.plot_dtw_trajectory(node_a, node_b)
lT.plot_dtw_heatmap(node_a, node_b)
```

### Multi-tree workflows

```python
from lineagetree import LineageTreeManager

manager = LineageTreeManager()
manager["embryo_1"] = lT_1
manager["embryo_2"] = lT_2

# Pairwise UTED across all stored trees
manager.compute_distances()
```

---

## Supported input formats

| Format | Algorithm / Tool | Reference |
|--------|-----------------|-----------|
| `.xml` (TGMM) | TGMM | [Amat et al. 2014](https://www.nature.com/articles/nmeth.3036) |
| `.xml` (MaMuT/TrackMate) | MaMuT / TrackMate | [Wolff et al. 2018](https://doi.org/10.7554/eLife.34410) / [Tinevez et al. 2017](https://doi.org/10.1016/j.ymeth.2016.09.016) |
| `.mastodon` | Mastodon | — |
| `.pkl` (ASTEC) | ASTEC | [Guignard, Fiuza et al. 2020](https://doi.org/10.1126/science.aar5663) |
| `.lT` | LineageTree native | — |
| `.swc` | SWC morphology | — |
| `.bmf` | Binary mesh format | — |
| `.csv` | Generic / Mastodon CSV | — |
| `.txt` | C. elegans specific | [Du et al. 2014](https://www.cell.com/fulltext/S0092-8674(13)01542-0) |

---

## Development

```shell
pip install -e ".[dev]"
pytest
```

To build the documentation:

```shell
pip install -e ".[doc]"
cd docs && make html
```

---

## Citation

If you use LineageTree in your research, please cite the relevant tracking algorithm paper for the data you loaded, and consider citing this library directly.

---

## License

MIT — see [LICENSE](LICENSE).
