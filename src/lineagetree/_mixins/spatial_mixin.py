from ..measure.spatial import (
    k_nearest_neighbours,
    spatial_density,
    spatial_edges,
    gabriel_graph,
    idx3d,
    neighbours_in_radius,
)

from .._core._deprecated import (
    compute_k_nearest_neighbours,
    compute_neighbours_in_radius,
    compute_spatial_density,
    compute_spatial_edges,
    get_gabriel_graph,
    get_idx3d,
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

    # 3.2 names, deprecated in 3.3
    get_idx3d = get_idx3d
    get_gabriel_graph = get_gabriel_graph
    compute_k_nearest_neighbours = compute_k_nearest_neighbours
    compute_spatial_edges = compute_spatial_edges
    compute_spatial_density = compute_spatial_density
    compute_neighbours_in_radius = compute_neighbours_in_radius
