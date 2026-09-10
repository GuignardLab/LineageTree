from ..measure.spatial import (
    compute_k_nearest_neighbours,
    compute_spatial_density,
    compute_spatial_edges,
    compute_gabriel_graph,
    compute_idx3d,
    compute_neighbours_in_radius,
    compute_angles,
    compute_asphericity,
    compute_displacement,
    compute_displacement_ratio,
    compute_duration,
    compute_max_displacement,
    compute_mean_squared_displacement,
    compute_outreach_ratio,
    compute_overall_angle,
    compute_speed,
    compute_straightness,
    compute_track_length,
    compute_velocity,
)

from ._methodize import AutoMethodizeMeta


class SpatialMixin(metaclass=AutoMethodizeMeta):
    """Mixin for spatial analysis operations."""

    compute_idx3d = compute_idx3d
    compute_gabriel_graph = compute_gabriel_graph
    compute_k_nearest_neighbours = compute_k_nearest_neighbours
    compute_spatial_edges = compute_spatial_edges
    compute_spatial_density = compute_spatial_density
    compute_neighbours_in_radius = compute_neighbours_in_radius
    compute_angles = compute_angles
    compute_asphericity = compute_asphericity
    compute_displacement = compute_displacement
    compute_displacement_ratio = compute_displacement_ratio
    compute_duration = compute_duration
    compute_max_displacement = compute_max_displacement
    compute_mean_squared_displacement = compute_mean_squared_displacement
    compute_outreach_ratio = compute_outreach_ratio
    compute_overall_angle = compute_overall_angle
    compute_speed = compute_speed
    compute_straightness = compute_straightness
    compute_track_length = compute_track_length
    compute_velocity = compute_velocity
