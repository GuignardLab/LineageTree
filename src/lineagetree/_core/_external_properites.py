from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping

from ..util_types import StaticTypedValueDict
from warnings import warn
if TYPE_CHECKING:
    from ..lineage_tree import LineageTree



class Properties:
    """The properties object holds different properties that are not part of the core feature set of a
    `LineageTree`, this may be the dictionary that holds the principal components for each timepoint
    the gene expression for each gene and so on. All properties in this object are converted to Exteranl properties,
    and are checked so that all their info corresponds to a node or a timepoint of the dataset.
    This class contains 3 kin ds of properties:
    - node_properties: Mapping nodes to values. All of the properties can be handled as labels, however the str values are automatically handled as labels.
    - time_properties: Mapping times to values. Example: average_density of each timepoint.
    - dataset property: Properties that are dataset wide. Example: A Transformation matrrix to rotate and translate the dataset.

    """
    _default_label: str|None = None

    def __repr__(self) -> str:
        ret = self.list_properties()
        if not ret:
            return "Empty properties"
        return str(ret)

    def remove(self, key: str):
        """Removes a property from any of the property dicts."""
        for prop in self._all_props:
            if key in prop:
                prop.pop(key)
                return

    def list_properties(
    self, constraint: Literal["node", "time", "forest", "labels"] | None = None
    ) -> list[str]:

        if constraint == "node":
            return list(self.node_properties.keys())

        elif constraint == "time":
            return list(self.time_properties.keys())

        elif constraint == "forest":
            return list(self.forest_properties.keys())

        elif constraint == "labels":
            return [
                name
                for name, prop in self.node_properties.items()
                if prop.data_type == str
            ]

        else:
            return (
                list(self.node_properties.keys())
                + list(self.time_properties.keys())
                + list(self.forest_properties.keys())
            )

    def __setattr__(
        self, name: str, value: Any
    ) -> None:  # To discuss if we want something like that.
        """Does not allow attribute assignement after the object initialization."""
        if hasattr(self, "_freeze") and not name.startswith("_"):
            raise TypeError(
                "'Properties' object does not support attribute assignment. Please use 'lT.add_attribute(...)'  or 'lT.properties.add_attribute(...)'"
            )
        return super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        """Makes the dictionaries saved inside `node_properties`,... accessible through the Properties object."""
        if name in (
            "node_properties",
            "time_properties",
            "forest_properties",
            "_all_props",
            "_lT",
        ):
            raise AttributeError(name)

        for prop_dict in self._all_props:
            try:
                return prop_dict[name]
            except KeyError:
                pass

        raise AttributeError(f"Property {name} does not exist.")

    def __dir__(self) -> list[str]:
        return [
            *self.node_properties,
            *self.time_properties,
            *self.forest_properties,
            *super().__dir__(),
        ]
    
    @property
    def label(self) -> StaticTypedValueDict:
        if self._default_label not in self.node_properties:
            for name, prop in self.node_properties.items():
                if prop.data_type is str:
                    self._default_label = name
                    warn(f"Label set to `{name}`")
                    break
            else:
                raise RuntimeError("No valid string property exists. Consider setting the label manually by lT.properties.set_label(...)")

        assert self._default_label is not None
        if self.node_properties[self._default_label].data_type is not str:
            return StaticTypedValueDict({k:str(val) for k,val in self.node_properties[self._default_label].items()})
        return self.node_properties[self._default_label]

    def set_label(self, name):
        if name in self.node_properties: 
            self._default_label = name
        else:
            raise KeyError("No such object exists in node properties.")

    def __init__(self, lT: LineageTree) -> None:
        self._lT = lT
        self.node_properties: dict[str,StaticTypedValueDict] = {}
        self.time_properties: dict[str,StaticTypedValueDict] = {}
        self.forest_properties: dict[str, Any] = {}
        self._all_props = [
            self.node_properties,
            self.time_properties,
            self.forest_properties,
        ]
        self._freeze = True

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
        if name in self.list_properties() and isinstance(value, Mapping):
            raise ValueError(
                f"Property named {name} already exists."
            )
        if isinstance(value, Mapping):
            value = StaticTypedValueDict(value)
        if isinstance(value, StaticTypedValueDict):
            if time_property:
                if problematic := set(value.keys()).difference(
                    self._lT.time_nodes
                ):
                    raise Warning(
                        f"Not all times exist in the dataset.Problematic keys are {problematic}"
                    )
                self.time_properties[name] = value
            else:
                if problematic := set(value.keys()).difference(self._lT.nodes):
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
    """
    return getattr(lT.properties, name, default)


def remove_property(lT: LineageTree, name: str):
    """Removes a property from any of the dictionaries in lT.properties."""
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
    constraint : Literal[node, time, forest] | None, optional
        Returns the keys from one of the Properties lists. If None returns allthe keys, by default None

    Returns
    -------
    list of str
        The properties in `lT.properties`.
    """

    return lT.properties.list_properties(constraint=constraint)
