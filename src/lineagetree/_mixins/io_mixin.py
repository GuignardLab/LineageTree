from .._io._writers import (
    _get_height,
    write,
    write_to_binary,
    write_to_svg,
    write_to_tlp,
)


class IOMixin:
    """Mixin for input/output operations."""

    _get_height = _get_height
    write = write
    write_to_binary = write_to_binary
    write_to_svg = write_to_svg
    write_to_tlp = write_to_tlp
