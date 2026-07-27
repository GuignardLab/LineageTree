from ..measure.spatial import (
    k_nearest_neighbours,
    spatial_density,
    spatial_edges,
    gabriel_graph,
    idx3d,
    neighbours_in_radius,
    angles,
    asphericity,
    displacement,
    displacement_ratio,
    duration,
    max_displacement,
    mean_squared_displacement,
    outreach_ratio,
    overall_angle,
    speed,
    straightness,
    track_length,
    velocity,
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
    angles = angles

    asphericity = asphericity

    displacement = displacement

    displacement_ratio = displacement_ratio

    duration = duration

    max_displacement = max_displacement

    mean_squared_displacement = mean_squared_displacement

    outreach_ratio = outreach_ratio

    overall_angle = overall_angle

    speed = speed

    straightness = straightness

    track_length = track_length

    velocity = velocity
