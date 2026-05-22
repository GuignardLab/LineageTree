from ..measure.spatial import (
    k_nearest_neighbours,
    spatial_density,
    spatial_edges,
    gabriel_graph,
    idx3d,
    neighbours_in_radius,
    get_angles,
    get_asphericity,
    get_displacement,
    get_displacement_ratio,
    get_duration,
    get_max_displacement,
    get_mean_squared_displacement,
    get_outreach_ratio,
    get_overall_angle,
    get_speed,
    get_straightness,
    get_track_length,
    get_velocity,
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
    get_angles = get_angles

    get_asphericity = get_asphericity

    get_displacement = get_displacement

    get_displacement_ratio = get_displacement_ratio

    get_duration = get_duration

    get_max_displacement = get_max_displacement

    get_mean_squared_displacement = get_mean_squared_displacement

    get_outreach_ratio = get_outreach_ratio

    get_overall_angle = get_overall_angle

    get_speed = get_speed

    get_straightness = get_straightness

    get_track_length = get_track_length

    get_velocity = get_velocity
