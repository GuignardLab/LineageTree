from typing import Any, Iterable, Mapping
from collections import UserDict
from ..util_types import StaticTypedValueDict


class Labels(StaticTypedValueDict):
    """Subclass of `StaticTypedValueDict` that only allows for string values."""

    def __init__(self, iterable: Iterable) -> None:
        super().__init__(iterable, str)


class Labelling:
    """Labelling can hold only `Labels` where all their keys are a subset of the nodes of `LineageTree`."""

    default_dict = {}

    def __init__(self, nodes) -> None:
        if isinstance(nodes, Iterable) and not isinstance(nodes, str):
            self.__dict__["_nodes"] = set(nodes)
        else:
            raise ValueError(
                f"Labelling expects an Iterable, not {type(nodes)}"
            )

    def __setattr__(self, name: str, value: dict | Labels) -> None:
        l = Labels(value)
        if not set(l).issubset(self._nodes):
            raise ValueError(
                "All ids in the labelset should correspond to ids in the LineageTree object."
            )
        if (
            not self.default_dict
        ):  # if the default dict has not been set set it up with the first label set available.
            super().__setattr__("default_dict", l)
        # self.list_of_labels.append(name)
        super().__setattr__(name, l)


def list_all_labels(lT) -> list[str]:
    return [
        k for k, v in lT.labelling.__dict__.items() if isinstance(v, Labels)
    ]
