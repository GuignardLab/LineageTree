# Welcome to LineageTree Documentation

LineageTree is a Python framework specialized in analyzing cell tracking data. It supports standard or custom tracking data formats and allows temporal and spatial analysis.

---

## Why should I use LineageTree?

<p style="text-align: justify;">

Understanding how a cluster of cells becomes a structured organism is a central question in developmental biology. Modern microscopy now allows researchers to image embryos in real time, and many tools exist for tracking nuclei or particles over time. However, most available software focuses primarily on spatial positions and lacks robust support for analyzing temporal structure and division patterns within lineages. LineageTree offers out-of-the-box algorithms to compare cell lineage data using both temporal and spatial information, while offers a strong enough backbone for the user to build their own algorithms.
</p>

Using LineageTree the user can:

- [Import any tracking format](./loaders.md)
- Perform [spatial](./spatial.md) analysis on the data
- Perform [unordered tree edit distance analysis](./uted.md) using different heuristics 
- [Visualize lineages](./viz.md) imported

---
## Unordered Tree Edit Distance (UTED)

One of the core analysis algorithms provided by this module is UTED, an approach designed to quantify how similar or dissimilar two lineages are. Calculating lineage distances is a powerful way to analyze developmental variability, identify fate gain or loss, and detect lineage symmetry within a single organism.

This framework is complemented by the napari plugin [ReLAX](https://guignardlab.github.io/napari-relax/). ReLAX allows users to explore a digital clone of the embryo in both space and time, annotate lineages, and perform systematic comparisons on imported lineage data. Together, these tools offer an integrated environment for visualizing, labeling, and analyzing lineage structures with temporal and spatial context.
<!-- </p> -->



## Quick Installation

```python
pip install LineageTree
```

## Content

[Getting started](getting-started.md): Installation guide

[Glossary](glossary.md): Contains all the different nomenclature the project is using and shows the connection with biological lineages.
