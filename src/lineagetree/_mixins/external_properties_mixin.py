from .._core._external_properites import (
    add_property,
    list_all_properties,
    # get_property,
    # del_property,
)

from ._methodize import AutoMethodizeMeta


class ExternalPropertiesMixin(metaclass=AutoMethodizeMeta):
    """Mixin for tree modification operations."""

    add_property = add_property
    list_all_properties = list_all_properties
    # get_property = get_property
    # del_property = del_property
