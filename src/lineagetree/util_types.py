from typing import Any, Iterable, Mapping
from collections import UserDict


class StaticTypedValueDict(UserDict):
    """Dict that allows only one type of values."""

    def __init__(
        self,
        data: Iterable,
        data_type: type | None = None,
    ) -> dict[int, Any]:

        if data is None and not data_type:
            if not isinstance(data, Mapping):
                value = next(iter(data))
                if len(value[0]) != 2:
                    raise ValueError(f"`data` could not be converted to dict.")
                self.data_type = type(next(iter(data))[1])
            else:
                self.data_type = type(next(iter(data.values())))
        else:
            self.data_type = data_type

        super().__init__(data)

    def __setitem__(self, key: int, item: Any) -> None:
        if not self.data_type:
            self.data_type = type(item)
        if not isinstance(item, self.data_type):
            raise TypeError(f"All values must be {self.data_type}")
        super().__setitem__(key, item)
