from ..measure.dynamic_time_warping import calculate_dtw
from ..measure.uted import (
    clear_comparisons,
    labelled_mappings,
    norm_dict,
    plot_tree_distance_graphs,
    unordered_tree_edit_distance,
    unordered_tree_edit_distances_at_time_t,
)

from ._methodize import AutoMethodizeMeta


class AnalysisMixin(metaclass=AutoMethodizeMeta):
    """Mixin for analysis operations (DTW, UTED)."""

    # DTW functions
    calculate_dtw = calculate_dtw

    # UTED functions
    clear_comparisons = clear_comparisons
    labelled_mappings = labelled_mappings
    norm_dict = norm_dict
    unordered_tree_edit_distances_at_time_t = (
        unordered_tree_edit_distances_at_time_t
    )
    unordered_tree_edit_distance = unordered_tree_edit_distance
    plot_tree_distance_graphs = plot_tree_distance_graphs
