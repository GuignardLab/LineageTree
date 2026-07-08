from __future__ import annotations
from typing import TYPE_CHECKING, Sequence, Any
import numpy as np
from dataclasses import dataclass
import warnings

from dataclasses import dataclass, field
from abc import ABC
from collections import UserList
from ..util_types import StaticTypedValueDict
import numbers

if TYPE_CHECKING:
    from ..lineage_tree import LineageTree
from typing import Any, Iterable, Mapping
from collections import UserDict


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
    """Property class for node attributes, should be a dict like structure"""

    ...


class TimeProperty(StaticTypedValueDict, ExternalProperty):
    """Property class for time properties."""

    ...


def DatasetProperty(var, time=False):
    """Automatically returns the correct External Property subclass."""
    if isinstance(var, dict) and time == False:
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

    property_list = []

    def __init__(self, lT: LineageTree) -> None:
        self.__dict__["lT"] = lT

    def __setattr__(self, name: str, value: dict | Any) -> None:

        if isinstance(value, NodeProperty) and not set(value).issubset(
            self.lT._nodes
        ):
            raise ValueError(
                f"All ids in the labelset should correspond to ids in the LineageTree object. `{name}` contans ids for labels not part of the dataset."
            )
        # if not self.lT.time and isinstance(value, TimeProperty):
        #     t_b, t_e = self.lT.t_b, self.lT.t_e
        #     self.__dict__["t_b"] = t_b
        #     self.__dict__["t_e"] = t_e
        elif isinstance(value, TimeProperty) and not set(value).issubset(
            set(range(self.lT.t_b, self.lT.t_e))
        ):
            raise ValueError(
                f"Timepoints should all be between `t_b` and `t_e`. Range is {self.lT.t_b, self.lT.t_e}. While the property {name} is {value.keys()}"
            )
        super().__setattr__(name, value)
        self.property_list.append(name)


def add_property(lT: LineageTree, name: str, value: Any, time_property: bool):
    lT.properties._nodes = lT.nodes
    setattr(lT.properties, name, DatasetProperty(value, time_property))


def list_all_properties(lT):

    return [
        prop
        for prop, val in lT.properties.__dict__.items()
        if isinstance(val, ExternalProperty)
    ]
