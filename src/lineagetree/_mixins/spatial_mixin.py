from ..measure.spatial import (
    k_nearest_neighbours,
    spatial_density,
    spatial_edges,
    gabriel_graph,
    idx3d,
    neighbours_in_radius,
)

from ._methodize import AutoMethodizeMeta


class SpatialMixin(metaclass=AutoMethodizeMeta):
    """Mixin for spatial analysis operations."""

    idx3d = idx3d
    gabriel_graph = gabriel_graph
    k_nearest_neighbours = k_nearest_neighbours
    spatial_edges = spatial_edges
    spatial_density = spatial_density
    neighbours_in_radius = neighbours_in_radius
