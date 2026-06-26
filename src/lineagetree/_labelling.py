from typing import Any, Iterable, Mapping
from collections import UserDict
import numpy as np
import numbers


class StaticTypedValueDict(UserDict):
    """Dict that allows only one type of values."""

    def __init__(
        self,
        data: Iterable,
        data_type: Any = None,
    ) -> None:
        if not data and data_type is None:
            raise ValueError("data_type cant be `None` if data is empty.")

        if data_type is None:
            if not isinstance(data, Mapping):
                tmp_d = tuple(data)
                if len(tmp_d[0]) != 2:
                    raise ValueError(f"`data` could not be converted to dict.")
                self.data_type = type(next(iter(data))[1])
            else:
                self.data_type = type(next(iter(data.values())))
        else:
            self.data_type = data_type

        super().__init__(data)

    def __setitem__(self, key: Any, item: Any) -> None:
        if not isinstance(item, self.data_type):
            raise TypeError(f"All values must be {self.data_type}")
        return super().__setitem__(key, item)


class Labels(StaticTypedValueDict):
    def __init__(self, iterable: Iterable) -> None:
        super().__init__(iterable, str)


class Labelling:

    default_dict = {}
    list_of_labels = []

    def __init__(self, nodes) -> None:
        if isinstance(nodes, Iterable) and not isinstance(nodes, str):
            self.__dict__["_nodes"] = set(nodes)
        else:
            raise ValueError(
                f"Labelling expects an Iterable, not {type(nodes)}"
            )

    def __setattr__(self, name: str, value: Any) -> None:
        l = Labels(value)
        if not set(l).issubset(self._nodes):
            raise ValueError(
                "All ids in the labelset should correspond to ids in the LineageTree object."
            )
        if (
            not self.default_dict
        ):  # if the default dict has not been set set it up with the first label set available.
            super().__setattr__("default_dict", l)
        self.list_of_labels.append(name)
        super().__setattr__(name, l)
