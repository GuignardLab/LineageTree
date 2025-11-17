# Welcome to LineageTree Documentation

LineageTree is a Python framework specialized in analyzing cell tracking data. It supports standard or custom tracking data formats and allows temporal and spatial analysis.

---

## Why should I use LineageTree?

<p style="text-align: justify;">

How does a cluster of cells become organized? Do cells move and divide randomly until an organism starts to exist? People from multiple fields have worked tirelessly to answer such questions, resulting in better microscopy techniques to image embryos developing in real time and software that allows users to track nuclei or particles inside the organisms. Many scientists have developed algorithms analyzing the positions of all these particles, however, there is a lack of software that takes time and division patterns into account. LineageTree offers out-of-the-box algorithms to compare lineage data to extract results on the similarity of lineages. It also offers a strong enough backbone for the user to build their algorithms to analyze temporal and spatial data.
</p>
<!-- <p style="text-align: justify;"> -->
---
## Unordered Tree Edit Distance (UTED)

One type of lineage data analysis is distance calculation, ehich means to find how similar or disimilar 2 lineages may be. This amalysis can provide useful information on how variable the development of an organism is, or extract information on fate gain or loss, or check for symmetric lineages in one organism. A napari plugin called [ReLAX](https://guignardlab.github.io/napari-relax/). accompanies this framework, where the user can explore a digital clone of the embryo on both spatial and temporal data, label different lineages, and perform systematic comparison analysis on imported lineages.
<!-- </p> -->

Using LineageTree the user can:

- [Import any tracking format](./loaders.md)
- Perform spatial analysis on the data
- Perform [unordered tree edit distance analysis](./uted.md) using different heuristics 
- [Visualize lineages](./viz.md) imported

## Quick Installation

```python
pip install LineageTree
```

## Content

[Getting started](getting-started.md): Installation guide

[Glossary](glossary.md): Contains all the different nomenclature the project is using and shows the connection with biological lineages.
