from ..measure.dynamic_time_warping import dtw
from ..measure.uted import (
    clear_comparisons,
    labelled_mappings,
    norm_dict,
    plot_tree_distance_graphs,
    unordered_tree_edit_distance,
    unordered_tree_edit_distances_at_time_t,
)

from .._core._deprecated import calculate_dtw
from ._methodize import AutoMethodizeMeta


class AnalysisMixin(metaclass=AutoMethodizeMeta):
    """Mixin for analysis operations (DTW, UTED)."""

    # DTW functions
    dtw = dtw
    calculate_dtw = calculate_dtw  # 3.2 name, deprecated in 3.3

    # UTED functions
    clear_comparisons = clear_comparisons
    labelled_mappings = labelled_mappings
    norm_dict = norm_dict
    unordered_tree_edit_distances_at_time_t = (
        unordered_tree_edit_distances_at_time_t
    )
    unordered_tree_edit_distance = unordered_tree_edit_distance
    plot_tree_distance_graphs = plot_tree_distance_graphs
