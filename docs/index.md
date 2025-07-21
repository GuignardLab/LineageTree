# Welcome to LineageTree Documentation

LineageTree is a Python framework specialized in cell tracking data. It supports importing standard or custom formats and comprehensive processing and comparison analysis capabilities.

## Why should I use LineageTree?

How does a cluster of cells become organized? Do cells move and divide randomly until an organism starts to exist? People from multiple fields have worked tirelessly to answer such questions, resulting in better microscopy techniques to image embryos developing in real time and software that allows users to track nuclei or particles inside the organisms. Many scientists have developed algorithms analyzing the positions of all these particles, however, there is a lack of software that takes time and division patterns into account. LineageTree offers out-of-the-box algorithms to compare lineage data to extract results on the similarity of lineages. It also offers a strong enough backbone for the user to build their algorithms to analyze time and spatial data. One type of lineage data analysis is lineage comparison, which can provide useful information on how variable the development of an organism is, or extract information on fate gain or loss, or check for symmetric lineages in one organism. A napari plugin called …. accompanies this framework, where the user can explore a digital clone of the embryo on both spatial and image data (4D 3D+time), label different lineages, and perform systematic comparison analysis on imported lineages.
Using LineageTree the user can:

- Import any tracking format (link to loaders)
- Perform analysis on the data ( link to tree_styles) (maybe talk a bit about the spatial algorithms)?
- Perform unordered tree edit distance analysis using different heuristics (tree_Styles)
- Visualize lineages imported (plot_node, plot_lineages)

## Quick Installation

```python
pip install LineageTree
```

## Content

[Getting started](getting-started.md): Installation guide

[Glossary](glossary.md): Contains all the different nomenclature the project is using and shows the connection with biological lineages.
