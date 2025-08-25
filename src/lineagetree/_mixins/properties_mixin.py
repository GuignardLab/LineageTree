from .._core._properties import (
    all_chains,
    depth,
    edges,
    labels,
    leaves,
    nodes,
    number_of_nodes,
    parenting,
    predecessor,
    roots,
    successor,
    t_b,
    t_e,
    time,
    time_nodes,
    time_resolution,
)

from ._methodize import AutoMethodizeMeta


class PropertiesMixin(metaclass=AutoMethodizeMeta):
    """Mixin for tree properties and basic structure."""

    successor = successor
    predecessor = predecessor
    time = time
    t_b = t_b
    t_e = t_e
    nodes = nodes
    number_of_nodes = number_of_nodes
    depth = depth
    roots = roots
    leaves = leaves
    edges = edges
    labels = labels
    time_resolution = time_resolution
    all_chains = all_chains
    time_nodes = time_nodes
    parenting = parenting
