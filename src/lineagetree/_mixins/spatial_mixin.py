from ..measure.spatial import (
    compute_k_nearest_neighbours,
    compute_spatial_density,
    compute_spatial_edges,
    get_gabriel_graph,
    get_idx3d,
)

from ._methodize import AutoMethodizeMeta


class SpatialMixin(metaclass=AutoMethodizeMeta):
    """Mixin for spatial analysis operations."""

    get_idx3d = get_idx3d
    get_gabriel_graph = get_gabriel_graph
    compute_k_nearest_neighbours = compute_k_nearest_neighbours
    compute_spatial_edges = compute_spatial_edges
    compute_spatial_density = compute_spatial_density
