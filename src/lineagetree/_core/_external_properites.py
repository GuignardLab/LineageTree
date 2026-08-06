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
        self, constraint: Literal["node", "time", "forest"] | None = None
    ) -> list[str]:
        """Returns a list with the properties saved either in node_proeprties, time_properties or forest_properties,
        or all at once.

        Parameters
        ----------
        constraint : Literal[node, time, forest] | None, optional
            What kind of properties to return, if None all of the properties are gonna be returned, by default None

        Returns
        -------
        list of str
            A list with the names of all the properties.
        """
        if constraint == "node":
            props = self.node_properties
        elif constraint == "time":
            props = self.time_properties
        if constraint == "forest":
            props = self.forest_properties
        else:
            props = self._all_props
        ret = [prop for prop_set in props for prop in prop_set]
        return ret

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
                f"Property named {name} already exists. Only forest_properties may be reassigned (non-Mapping objects)."
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
