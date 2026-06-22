from __future__ import annotations
from typing import TYPE_CHECKING, Sequence, Any
import numpy as np

if TYPE_CHECKING:
    from ..lineage_tree import LineageTree


def add_property(
    lT: LineageTree, d: dict | Sequence[dict] | Any, name: str | Sequence[str]
):
    """Adds a property to the `LineageTree` object.

    Parameters
    ----------
    lT : LineageTree
        The `LineageTree` object.
    d : dict | Sequence[dict]
        The property dictionary/ dictionaries.
    name : str | Sequence[str]
        The name of the property.
    """
    if not isinstance(name, str) and isinstance(name, Sequence):
        if (
            isinstance(name, Sequence)
            and len(name) != len(d)
            and isinstance(d, Sequence)
        ):
            raise ValueError(
                "When `name` is a sequence, `d` must be a sequence of properties of the same length."
            )
        for d_i, n_i in zip(d, name):
            add_property(lT, d_i, n_i)
        return
    if isinstance(d, dict):
        for node in lT.nodes:
            d.setdefault(node, np.nan)

    lT._property_dict[name] = d


def get_property(lT: LineageTree, key):
    if key not in lT._property_dict:
        raise KeyError(f"No attribute `{key}` in the properties.")
    return lT._property_dict[key]


def del_property(lT: LineageTree, key):
    lT._property_dict.pop(key)


def list_all_properties(
    lT: LineageTree,
):
    _fix_external_properties(lT)
    return list(lT._property_dict.keys())


def _fix_external_properties(
    lT: LineageTree,
):

    external_properties = {
        prop_name: prop
        for prop_name, prop in lT.__dict__.items()
        if prop_name
        not in [
            "successor",
            "predecessor",
            "time",
            "_successor",
            "_predecessor",
            "_time",
            "pos",
            "labels",
            "name",
            "node_name",
            "spatial_resolution",
            "time_edges",
            "time_id",
            "progeny",
        ]
        + lT._dynamic_properties
        + lT._protected_dynamic_properties
        and prop_name[0] != "_"
    }
    for n, prop in external_properties.items():
        if prop:
            add_property(lT, prop, n)
        delattr(lT, n)
