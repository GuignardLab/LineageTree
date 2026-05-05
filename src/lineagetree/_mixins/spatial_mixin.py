from ..measure.spatial import (
    compute_k_nearest_neighbours,
    compute_spatial_density,
    compute_spatial_edges,
    get_gabriel_graph,
    get_idx3d,
    compute_neighbours_in_radius,
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

    get_idx3d = get_idx3d
    get_gabriel_graph = get_gabriel_graph
    compute_k_nearest_neighbours = compute_k_nearest_neighbours
    compute_spatial_edges = compute_spatial_edges
    compute_spatial_density = compute_spatial_density
    compute_neighbours_in_radius = compute_neighbours_in_radius
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
