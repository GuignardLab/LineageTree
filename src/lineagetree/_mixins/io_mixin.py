from .._io._writers import (
    _get_height,
    write,
    write_to_svg,
    write_to_tlp,
)

from ._methodize import AutoMethodizeMeta


class IOMixin(metaclass=AutoMethodizeMeta):
    """Mixin for input/output operations."""

    _get_height = _get_height
    write = write
    write_to_svg = write_to_svg
    write_to_tlp = write_to_tlp
