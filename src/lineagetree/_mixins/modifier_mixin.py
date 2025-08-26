from .._core._modifier import (
    _add_node,
    add_chain,
    add_root,
    get_next_id,
    modifier,
    remove_nodes,
)

from ._methodize import AutoMethodizeMeta


class ModifierMixin(metaclass=AutoMethodizeMeta):
    """Mixin for tree modification operations."""

    _add_node = _add_node
    add_chain = add_chain
    add_root = add_root
    get_next_id = get_next_id
    modifier = modifier
    remove_nodes = remove_nodes
