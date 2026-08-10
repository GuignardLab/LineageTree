from typing import Any, Iterable, Mapping
from collections import UserDict


class StaticTypedValueDict(UserDict):
    """A dict-like container (UserDict subclass) whose values must all share
    one single type.

    The enforced type (`self.data_type`) can be established in one of three
    ways:

    1. **Explicitly**, via the `data_type` argument::

        var = StaticTypedValueDict({}, data_type=int)

    2. **Lazily**, from the first value ever assigned, if created empty
    with no `data_type`::

        var = StaticTypedValueDict()
        var[0] = 5        # data_type becomes int
        var[1] = "1"      # raises TypeError

    3. **Inferred from initial data**, if a non-empty `data` is passed
    without `data_type`::

        var = StaticTypedValueDict({5: 1, 6: 2})  # data_type becomes int

    Once `data_type` is set (by any of the above), every subsequent
    `__setitem__` call is checked against it; mismatched types raise
    `TypeError`. Aside from this constraint, the object behaves exactly
    like a normal dict (iteration, `.get`, `.update`, etc. all work as
    inherited from `UserDict`).

    Parameters
    ----------
    data : Mapping[int, Any] | Iterable[tuple[int, Any]] | None, optional
        Initial contents. Can be a mapping (e.g. a plain dict) or an
        iterable of `(key, value)` pairs. Defaults to an empty dict.
    data_type : type | None, optional
        The single type every value must be an instance of. If not
        given, it is inferred from `data` (when non-empty) or from the
        first value assigned via `__setitem__`.

    Raises
    ------
    ValueError
        If `data` is a non-mapping iterable whose entries are not
        `(key, value)` pairs.
    TypeError
        If a value is assigned that is not an instance of `data_type`.

    Examples
    --------
    >>> d = StaticTypedValueDict({}, data_type=int)
    >>> d[0] = 5
    >>> d[1] = "1"
    Traceback (most recent call last):
        ...
    TypeError: All values must be <class 'int'>
    """

    def __init__(
        self,
        data: Iterable | None = None,
        data_type: type | None = None,
    ):
        if data is None:
            data = {}

        if data and not data_type:
            if not isinstance(data, Mapping):
                value = next(iter(data))
                if len(value) != 2:
                    raise ValueError("`data` could not be converted to dict.")
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
