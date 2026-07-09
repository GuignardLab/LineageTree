from __future__ import annotations
from typing import TYPE_CHECKING, Any
import numpy as np
from dataclasses import dataclass

from abc import ABC
from ..util_types import StaticTypedValueDict
import numbers

if TYPE_CHECKING:
    from ..lineage_tree import LineageTree


import operator

OPS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "truediv": operator.truediv,
    "floordiv": operator.floordiv,
    "mod": operator.mod,
    "pow": operator.pow,
    "and": operator.and_,
    "or": operator.or_,
    "xor": operator.xor,
    "lshift": operator.lshift,
    "rshift": operator.rshift,
}

NEGATION = {"neg": operator.neg}

COMPARE_OPS = {
    "lt": operator.lt,
    "le": operator.le,
    "gt": operator.gt,
    "ge": operator.ge,
    "eq": operator.eq,
    "ne": operator.ne,
}


def operations(
    attr="data",
):  # Main reason is that the result of numeric operations should always be
    # the same class in DatasetProperties
    # is operation deliberately does not work use ==.
    def decorator(cls):

        def normal_operations(op):
            def method(self, other):
                return type(self)(op(getattr(self, attr), other))

            return method

        def right_operations(op):
            def method(self, other):
                return type(self)(op(other, getattr(self, attr)))

            return method

        def inplace_operations(op):
            def method(self, other):
                setattr(self, attr, op(getattr(self, attr), other))
                return self

            return method

        def make(op):
            def method(self):
                return type(self)(op(getattr(self, attr)))

            return method

        def cmp(op):
            def method(self, other):
                return op(getattr(self, attr), other)

            return method

        # arithmetic
        for name, op in OPS.items():
            setattr(cls, f"__{name}__", normal_operations(op))
            setattr(cls, f"__r{name}__", right_operations(op))
            setattr(cls, f"__i{name}__", inplace_operations(op))

        # Just for negation
        for name, op in NEGATION.items():
            setattr(cls, f"__{name}__", make(op))

        # comparisons → MUST return bool
        for name, op in COMPARE_OPS.items():
            setattr(cls, f"__{name}__", cmp(op))

        return cls

    return decorator


@dataclass
class ExternalProperty(ABC):
    """General Property template"""

    data: Any

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return repr(self.data)


class NodeProperty(StaticTypedValueDict, ExternalProperty):
    """Property class for node attributes, should be a dict like structure where the keys are the nodes
    and the values are the property. Because it is a StaticallyTypedValueDict, only one type of
    values can be accepted."""


class TimeProperty(StaticTypedValueDict, ExternalProperty):
    """Property class for node attributes, should be a dict like structure where the keys are the times.
    Because it is a StaticallyTypedValueDict, only one type ofvalues can be accepted.
    """


def DatasetProperty(var, time=False):
    """Automatically returns the correct External Property subclass."""
    if isinstance(var, dict) and time is False:
        return NodeProperty(var)
    elif isinstance(var, dict) and time:
        return TimeProperty(var)
    elif isinstance(var, np.number | numbers.Number):
        return DatasetPropertyNumeric(var)
    elif isinstance(var, str):
        return DatasetPropertyString(var)
    elif isinstance(var, list):
        return DatasetPropertyList(var)
    elif isinstance(var, set):
        return DatasetPropertySet(var)
    elif isinstance(var, tuple):
        return DatasetPropertyTuple(var)
    else:
        raise Warning(
            "Value couldnot be converted to dataset property only numericals, lists, sets, tuples and dictionaries can be converted to dataset properties."
        )


@operations("data")
@dataclass
class DatasetPropertyNumeric(
    ExternalProperty,
):
    """Whole class numeric property. Has all methods implemented."""

    data: numbers.Number | np.number


@operations("data")
@dataclass
class DatasetPropertyString(ExternalProperty):
    """Whole class string property. Has all methods implemented."""

    data: str


class DatasetPropertyList(ExternalProperty, list): ...


class DatasetPropertySet(ExternalProperty, set): ...


class DatasetPropertyTuple(ExternalProperty, tuple): ...


class Properties:
    """The properties object holds different properties that are not part of the core feature set of a
    `LineageTree`, this may be the dictionary that holds the principal components for each timepoint
    the gene expression for each gene and so on. All properties in this object are converted to Exteranl properties,
    and are checked so that all their info corresponds to a node or a timepoint of the dataset.

    """

    property_list = []

    def __repr__(self) -> str:
        ret = [
            prop
            for prop, val in self.__dict__.items()
            if isinstance(val, ExternalProperty)
        ]
        if not ret:
            return "Empty properties"
        return str(ret)

    def pop(self, key: str, default=None):
        return self.__dict__.pop(key, default)

    @property
    def list_properties(self):
        return [
            prop
            for prop, val in self.__dict__.items()
            if isinstance(val, ExternalProperty)
        ]

    def __init__(self, lT: LineageTree) -> None:
        self.__dict__["lT"] = lT

    def __setattr__(self, name: str, value: dict | Any) -> None:
        """Custom setattr so that only valid properties can be added."""

        if isinstance(value, NodeProperty) and not set(value).issubset(
            self.lT.nodes
        ):
            raise ValueError(
                f"All ids in the labelset should correspond to ids in the LineageTree object. `{name}` contans ids for labels not part of the dataset."
            )
        elif isinstance(value, TimeProperty) and not set(value).issubset(
            set(range(self.lT.t_b, self.lT.t_e))
        ):
            raise ValueError(
                f"Timepoints should all be between `t_b` and `t_e`. Range is {self.lT.t_b, self.lT.t_e}. While the property {name} is {value.keys()}"
            )
        super().__setattr__(name, value)
        self.property_list.append(name)


def add_property(lT: LineageTree, name: str, value: Any, time_property: bool):
    """Adds a property to the `lT.properties` object.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree object.
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
    setattr(lT.properties, name, DatasetProperty(value, time_property))


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


def del_property(lT: LineageTree, name: str, default=None):
    return lT.properties.pop(name, default)


def list_all_properties(lT: LineageTree) -> list[str]:
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

    return lT.properties.list_properties
