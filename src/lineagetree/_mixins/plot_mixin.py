from ..plot import (
    _create_dict_of_plots,
    draw_tree_graph,
    plot_all_lineages,
    plot_dtw_heatmap,
    plot_dtw_trajectory,
    plot_subtree,
)

from ._methodize import AutoMethodizeMeta


class PlotMixin(metaclass=AutoMethodizeMeta):
    """Mixin for plotting functionality."""

    _create_dict_of_plots = _create_dict_of_plots
    draw_tree_graph = draw_tree_graph
    plot_all_lineages = plot_all_lineages
    plot_dtw_heatmap = plot_dtw_heatmap
    plot_dtw_trajectory = plot_dtw_trajectory
    plot_subtree = plot_subtree
