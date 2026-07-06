from __future__ import annotations
from typing import TYPE_CHECKING, Sequence, Any
import numpy as np
from dataclasses import dataclass
import warnings
from collections.abc import (
    MutableMapping,
)  # I do not need to impement all methods and it works like a dict
from dataclasses import dataclass, field
from abc import ABC
from collections import UserList

if TYPE_CHECKING:
    from .lineage_tree import LineageTree

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


@dataclass
class NodeProperty(ExternalProperty, dict):
    """Property class for node attributes, should be a dict like structure"""

    ...


class TimeProperty(ExternalProperty, dict):
    """Property class for time properties."""

    ...


def DatasetProperty(var, time=False):
    """Automatically returns the correct External Property subclass."""
    if isinstance(var, dict):
        return NodeProperty(var)
    elif isinstance(var, dict) and time:
        return TimeProperty(var)
    elif isinstance(var, np.number):
        return DatasetPropertyNumeric(var)
    elif isinstance(var, str):
        return DatasetPropertyString(var)
    elif isinstance(var, list):
        return DatasetPropertyList(var)


@operations("data")
@dataclass
class DatasetPropertyNumeric(
    ExternalProperty,
):
    """Whole class numeric property. Has all methods implemented."""

    data: int | float | np.number


@operations("data")
@dataclass
class DatasetPropertyString(ExternalProperty):
    """Whole class string property. Has all methods implemented."""

    data: str


@dataclass
class DatasetPropertyList(ExternalProperty, UserList):
    data: list


def add_property(
    lT: LineageTree,
    d: dict | Any,
    name: str,
    time_property=False,
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
    if isinstance(d, dict):
        if (
            all([lT.t_b < i < lT.t_e for i in lT.keys()])
            and time_property is False
        ):
            warnings.warn(
                "Looks like a time property. Maybe try `lT.add_property(val, name, time_property=True)`"
            )

    setattr(lT, name, DatasetProperty(d, time_property))


def get_property(lT: LineageTree, key):
    if key not in lT._property_dict:
        raise KeyError(f"No attribute `{key}` in the properties.")
    return lT._property_dict[key]


def del_property(lT: LineageTree, key):
    lT._property_dict.pop(key)


def list_all_properties(
    lT: LineageTree,
):

    return list(lT._property_dict.keys())
