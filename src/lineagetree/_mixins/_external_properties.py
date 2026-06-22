from ..measure.external_properites import add_property, list_all_properties

from ._methodize import AutoMethodizeMeta


class ExtgernalPropertiesMixin(metaclass=AutoMethodizeMeta):
    """Mixin for tree modification operations."""

    add_property = add_property
    list_all_properties = list_all_properties
