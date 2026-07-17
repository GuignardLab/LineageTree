from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping
import numpy as np

from abc import ABC
from ..util_types import StaticTypedValueDict
import numbers

if TYPE_CHECKING:
    from ..lineage_tree import LineageTree


def DatasetProperty(var, time=False):
    """Automatically returns the correct External Property subclass."""
    if isinstance(var, dict) and time is False:
        return var
    elif isinstance(var, dict) and time:
        return var

    else:
        raise Warning(
            "Value couldnot be converted to dataset property only numericals, lists, sets, tuples and dictionaries can be converted to dataset properties."
        )


class Properties:
    """The properties object holds different properties that are not part of the core feature set of a
    `LineageTree`, this may be the dictionary that holds the principal components for each timepoint
    the gene expression for each gene and so on. All properties in this object are converted to Exteranl properties,
    and are checked so that all their info corresponds to a node or a timepoint of the dataset.

    """

    def __repr__(self) -> str:
        ret = self.list_properties()
        if not ret:
            return "Empty properties"
        return str(ret)

    def remove(self, key: str):
        """Removes a property from any of the property dicts."""
        for prop in self.all_props:
            if key in prop:
                prop.pop(key)
                return

    def list_properties(
        self, constraint: Literal["node", "time", "forest"] | None = None
    ):
        if constraint == "node":
            props = self.node_properties
        elif constraint == "time":
            props = self.time_properties
        if constraint == "forest":
            props = self.forest_properties
        else:
            props = self.all_props
        ret = [prop for prop_set in props for prop in prop_set]
        return ret

    def __getattr__(self, name: str) -> Any:
        if name in (
            "node_properties",
            "time_properties",
            "forest_properties",
            "all_props",
            "lT",
        ):
            raise AttributeError(name)

        for i in [
            self.node_properties,
            self.time_properties,
            self.forest_properties,
        ]:
            try:
                return i[name]
            except KeyError:
                pass

        raise AttributeError(f"Property {name} does not exist.")

    def __dir__(self) -> list[str]:
        return (
            list(self.node_properties.keys())
            + list(self.time_properties.keys())
            + list(self.forest_properties.keys())
            + list(super().__dir__())
        )

    def __init__(self, lT: LineageTree) -> None:
        self.lT = lT
        self.node_properties = {}
        self.time_properties = {}
        self.forest_properties = {}
        self.all_props = [
            self.node_properties,
            self.time_properties,
            self.forest_properties,
        ]

    def add_property(self, name: str, value: Any, time_property: bool):
        """Adds a property to the `lT.properties` object.

        Parameters
        ----------
        name : str
            The name of the new property.
        value : Any
            The value of the property which is gonna be converted to an ExternalProperty style object.
        time_property : bool
            Only important for dict like properties, for other data types has no effect.
            If False the dictionary will become a `NodeProperty`,
            meaning all of the keys are nodes. If True, the dicitonary will become a TimeProperty, meaning that the keys
            are the same as `lT.times_nodes` of the dataset.
        """
        if name in self.list_properties():
            raise ValueError(f"Property named {name} already exists.")
        if isinstance(value, Mapping):
            value = StaticTypedValueDict(value)
        if isinstance(value, StaticTypedValueDict):
            if time_property:
                if problematic := set(value.keys()).difference(
                    self.lT.time_nodes
                ):
                    raise Warning(
                        f"Not all times exist in the dataset.Problematic keys are {problematic}"
                    )
                self.time_properties[name] = value
            else:
                if problematic := set(value.keys()).difference(self.lT.nodes):
                    raise Warning(
                        f"Not all nodes exist in the dataset.Problematic keys are {problematic}"
                    )
                self.node_properties[name] = value
        else:
            self.forest_properties[name] = value


def add_property(lT: LineageTree, name: str, value: Any, time_property: bool):

    return lT.properties.add_property(name, value, time_property)


def get_property(lT: LineageTree, name: str, default=None):
    """Function for getting properties from `lT.properties`, identical to accessing the
    properties object and receiving the attribute you want.

    Parameters
    ----------
    lT : LineageTree
        The `LineageTree` object.
    name : str
        The name of the attribute
    default : _type_, optional
        The default return if the object does not exist, by default None

    Returns
    -------
    ExternalProperty-subclass objects
        All properties in the `lT.properties` object are converted to a specialized subclass of ExternalProperty.
    """
    return getattr(lT.properties, name, default)


def remove_property(lT: LineageTree, name: str):
    return lT.properties.remove(name)


def list_all_properties(
    lT: LineageTree,
    constraint: Literal["node", "time", "forest"] | None = None,
) -> list[str]:
    """Lists all objects that exist in `lT.properies`.

    Parameters
    ----------
    lT : LineageTree
        The `LineageTree` object.

    Returns
    -------
    list of str
        The properties in `lT.properties`.
    """

    return lT.properties.list_properties(constraint=constraint)
