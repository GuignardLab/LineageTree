"""Checks on what the documentation renders from the code."""

import inspect
import re

from lineagetree import LineageTree


def test_methods_hide_the_lineage_tree_parameter():
    """Methods are documented as called, without the `lT` first parameter."""
    assert "lT" not in inspect.signature(LineageTree.get_successors).parameters
    assert "lT" not in inspect.signature(LineageTree.write).parameters
    assert "lT : LineageTree" not in (LineageTree.write.__doc__ or "")


def test_no_empty_parameters_section():
    """Removing `lT` must not leave an empty `Parameters` section behind."""
    empty = re.compile(r"Parameters\n\s*-+\n\s*(\n|\Z|Returns|Raises)")
    for name, member in inspect.getmembers(LineageTree, callable):
        if not name.startswith("_") and member.__doc__:
            assert not empty.search(member.__doc__), name


def test_internal_helpers_are_not_public():
    """Internal helpers should not show up as methods of the tree."""
    assert not hasattr(LineageTree, "modifier")
    assert not hasattr(LineageTree, "norm_dict")
