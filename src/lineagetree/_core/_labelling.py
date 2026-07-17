from __future__ import annotations
from typing import Iterable, Literal
from ..util_types import StaticTypedValueDict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..lineage_tree import LineageTree


class Labels(StaticTypedValueDict):
    """Subclass of `StaticTypedValueDict` that only allows for string values."""

    def __init__(self, iterable: Iterable) -> None:
        super().__init__(iterable, str)


class Labelling:
    """Labelling can hold only `Labels` where all their keys are a subset of the nodes of `LineageTree`.
    An important part of `Labelling` is the default_dict, this property returns a lebelset that exists in labelling,
    you can change the labelset the default dict returns with `change_default_label
    """

    _default_dict: str = None

    @property
    def default_dict(self):
        if not self.list_labels:
            raise ValueError("Label list is empty.")
        if (
            self._default_dict not in self.list_labels
            and "_default_dict" != self._default_dict
        ):
            self.__dict__["_default_dict"] = next(iter(self.list_labels))
        return self.__dict__[self._default_dict]

    def __repr__(self) -> str:
        ret = self.list_labels
        if not ret:
            return "Empty Label set"
        return str(ret)

    @property
    def list_labels(self):
        return [
            prop
            for prop, val in self.__dict__.items()
            if isinstance(val, Labels)
        ]

    def change_default_label(self, name: str):
        if name not in self.list_labels:
            raise ValueError(
                f"Label {name} not part of the labelling object. Please choose one of the following: {self.list_labels}."
            )
        self.__dict__["_default_label"] = name

    def pop(self, key: str, default=None):
        return self.__dict__.pop(key, default)

    def __init__(self, lT: LineageTree) -> None:
        self.__dict__["lT"] = lT

    def __setattr__(self, name: str, value: dict | Labels) -> None:
        lbl = Labels(value)
        if problematic := set(lbl).difference(self.lT.nodes):
            raise ValueError(
                f"All ids in the labelset should correspond to ids in the LineageTree object.{problematic}"
            )
        super().__setattr__(name, lbl)


def list_all_labels(lT: LineageTree) -> list[str]:
    return lT.labelling.list_labels


def add_label(lT: LineageTree, name: str, label: dict):
    setattr(lT.labelling, name, label)


def get_label(lT: LineageTree, name: str | Literal["default"] = "default"):
    return getattr(lT.labelling, name)


def del_label(lT: LineageTree, name: str, default):
    lT.labelling.pop(name, default)
