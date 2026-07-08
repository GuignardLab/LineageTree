from .._core._labelling import (
    list_all_labels,
    # get_label,
    # del_label,
)

from ._methodize import AutoMethodizeMeta


class LabellingMixin(metaclass=AutoMethodizeMeta):
    """Mixin for tree modification operations."""

    # add_label = add_label
    list_all_labels = list_all_labels
    # get_label = get_label
    # del_label = del_label
