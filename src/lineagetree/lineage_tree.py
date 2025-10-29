#!python
# This file is subject to the terms and conditions defined in
# file 'LICENCE', which is part of this source code package.
# Author: Leo Guignard (leo.guignard...@AT@...univ-amu.fr)
from __future__ import annotations

import importlib.metadata
import warnings
from collections.abc import Iterable, Sequence
from packaging.version import Version
import numpy as np

from ._core.utils import CompatibleUnpickler
from ._mixins.properties_mixin import PropertiesMixin
from ._mixins.modifier_mixin import ModifierMixin
from ._mixins.navigation_mixin import NavigationMixin
from ._mixins.plot_mixin import PlotMixin
from ._mixins.spatial_mixin import SpatialMixin
from ._mixins.analysis_mixin import AnalysisMixin
from ._mixins.io_mixin import IOMixin
from ._core.validation import TreeValidator


class LineageTree(
    PropertiesMixin,
    ModifierMixin,
    NavigationMixin,
    PlotMixin,
    SpatialMixin,
    AnalysisMixin,
    IOMixin,
):
    """A lineage tree data structure with comprehensive analysis capabilities."""

    def __eq__(self, other) -> bool:
        if isinstance(other, LineageTree):
            return (
                other._successor == self._successor
                and other._predecessor == self._predecessor
                and other._time == self._time
            )
        else:
            return False

    def __setstate__(self, state):
        if "_successor" not in state:
            state["_successor"] = state["successor"]
        if "_predecessor" not in state:
            state["_predecessor"] = state["predecessor"]
        if "_time" not in state:
            state["_time"] = state["time"]
        self.__dict__.update(state)

    @classmethod
    def load(clf, fname: str):
        """Loading a lineage tree from a '.lT' file.

        Parameters
        ----------
        fname : str
            path to and name of the file to read

        Returns
        -------
        LineageTree
            loaded file
        """
        with open(fname, "br") as f:
            lT = CompatibleUnpickler(f).load()
            f.close()
        if not hasattr(lT, "__version__") or Version(lT.__version__) < Version(
            "2.0.0"
        ):
            properties = {
                prop_name: prop
                for prop_name, prop in lT.__dict__.items()
                if (isinstance(prop, dict) or prop_name == "_time_resolution")
                and prop_name
                not in [
                    "successor",
                    "predecessor",
                    "time",
                    "_successor",
                    "_predecessor",
                    "_time",
                    "pos",
                    "labels",
                ]
                + LineageTree._dynamic_properties
                + LineageTree._protected_dynamic_properties
            }
            lT = LineageTree(
                successor=lT._successor,
                time=lT._time,
                pos=lT.pos,
                name=lT.name if hasattr(lT, "name") else None,
                **properties,
            )
        if not hasattr(lT, "time_resolution"):
            lT.time_resolution = 1

        return lT

    def get_subtree(self, node_list: set[int]) -> LineageTree:
        """Create a new lineage tree that has the same edges and properties
        as the given lineage tree. Only the nodes in `node_list` are considered.

        Parameters
        ----------
        node_list : Iterator of int
            Iterator over the nodes to keep

        Returns
        -------
        LineageTree
            The subtree lineage tree
        """
        new_successors = {
            n: tuple(vi for vi in self.successor[n] if vi in node_list)
            for n in node_list
        }
        return LineageTree(
            successor=new_successors,
            time=self._time,
            pos=self.pos,
            name=self.name,
            root_leaf_value=[
                (),
            ],
            **{
                name: self.__dict__[name]
                for name in self._custom_property_list
            },
        )

    def __init__(
        self,
        *,
        successor: dict[int, Sequence] | None = None,
        predecessor: dict[int, int | Sequence] | None = None,
        time: dict[int, int] | None = None,
        starting_time: int | None = None,
        pos: dict[int, Iterable] | None = None,
        name: str | None = None,
        root_leaf_value: Sequence | None = None,
        **kwargs,
    ):
        """Create a LineageTree object from minimal information, without reading from a file.
        Either `successor` or `predecessor` should be specified.

        Parameters
        ----------
        successor : dict mapping int to Iterable
            Dictionary assigning nodes to their successors.
        predecessor : dict mapping int to int or Iterable
            Dictionary assigning nodes to their predecessors.
        time : dict mapping int to int, optional
            Dictionary assigning nodes to the time point they were recorded to.
            Defaults to None, in which case all times are set to `starting_time`.
        starting_time : int, optional
            Starting time of the lineage tree. Defaults to 0.
        pos : dict mapping int to Iterable, optional
            Dictionary assigning nodes to their positions. Defaults to None.
        name : str, optional
            Name of the lineage tree. Defaults to None.
        root_leaf_value : Iterable, optional
            Iterable of values of roots' predecessors and leaves' successors in the successor and predecessor dictionaries.
            Defaults are `[None, (), [], set()]`.
        **kwargs:
            Supported keyword arguments are dictionaries assigning nodes to any custom property.
            The property must be specified for every node, and named differently from LineageTree's own attributes.
        """
        self.__version__ = importlib.metadata.version("lineagetree")
        self.name = str(name) if name is not None else None

        self._validator = TreeValidator(self)

        if successor is not None and predecessor is not None:
            raise ValueError(
                "You cannot have both successors and predecessors."
            )

        if root_leaf_value is None:
            root_leaf_value = [None, (), [], set()]
        elif not isinstance(root_leaf_value, Iterable):
            raise TypeError(
                f"root_leaf_value is of type {type(root_leaf_value)}, expected Iterable."
            )
        elif len(root_leaf_value) < 1:
            raise ValueError(
                "root_leaf_value should have at least one element."
            )
        self._successor = {}
        self._predecessor = {}
        if successor is not None:
            for pred, succs in successor.items():
                if succs in root_leaf_value:
                    self._successor[pred] = ()
                else:
                    if not isinstance(succs, Iterable):
                        raise TypeError(
                            f"Successors should be Iterable, got {type(succs)}."
                        )
                    if len(succs) == 0:
                        raise ValueError(
                            f"{succs} was not declared as a leaf but was found as a successor.\n"
                            "Please lift the ambiguity."
                        )
                    self._successor[pred] = tuple(succs)
                    for succ in succs:
                        if succ in self._predecessor:
                            raise ValueError(
                                "Node can have at most one predecessor."
                            )
                        self._predecessor[succ] = (pred,)
        elif predecessor is not None:
            for succ, pred in predecessor.items():
                if pred in root_leaf_value:
                    self._predecessor[succ] = ()
                else:
                    if isinstance(pred, Sequence):
                        if len(pred) == 0:
                            raise ValueError(
                                f"{pred} was not declared as a leaf but was found as a successor.\n"
                                "Please lift the ambiguity."
                            )
                        if 1 < len(pred):
                            raise ValueError(
                                "Node can have at most one predecessor."
                            )
                        pred = pred[0]
                    self._predecessor[succ] = (pred,)
                    self._successor.setdefault(pred, ())
                    self._successor[pred] += (succ,)
        for root in set(self._successor).difference(self._predecessor):
            self._predecessor[root] = ()
        for leaf in set(self._predecessor).difference(self._successor):
            self._successor[leaf] = ()

        if self._validator.check_for_cycles():
            raise ValueError(
                "Cycles were found in the tree, there should not be any."
            )

        if pos is None or len(pos) == 0:
            self.pos = {}
        else:
            if self.nodes.difference(pos) != set():
                raise ValueError("Please provide the position of all nodes.")
            self.pos = {
                node: np.array(position) for node, position in pos.items()
            }
        if "labels" in kwargs:
            self._labels = kwargs["labels"]
            kwargs.pop("labels")
        if time is None:
            if starting_time is None:
                starting_time = 0
            if not isinstance(starting_time, int):
                warnings.warn(
                    f"Attribute `starting_time` was a `{type(starting_time)}`, has been casted as an `int`.",
                    stacklevel=2,
                )
                starting_time = int(starting_time)
            self._time = dict.fromkeys(self.roots, starting_time)
            queue = list(self.roots)
            for node in queue:
                for succ in self._successor[node]:
                    self._time[succ] = self._time[node] + 1
                    queue.append(succ)
        else:
            if starting_time is not None:
                warnings.warn(
                    "Both `time` and `starting_time` were provided, `starting_time` was ignored.",
                    stacklevel=2,
                )
            self._time = {n: int(time[n]) for n in self.nodes}
            if self._time != time:
                if len(self._time) != len(time):
                    warnings.warn(
                        "The provided `time` dictionary had keys that were not nodes. "
                        "They have been removed",
                        stacklevel=2,
                    )
                else:
                    warnings.warn(
                        "The provided `time` dictionary had values that were not `int`. "
                        "These values have been truncated and converted to `int`",
                        stacklevel=2,
                    )
            if self.nodes.symmetric_difference(self._time) != set():
                raise ValueError(
                    "Please provide the time of all nodes and only existing nodes."
                )
            if not all(
                self._time[node] < self._time[s]
                for node, succ in self._successor.items()
                for s in succ
            ):
                raise ValueError(
                    "Provided times are not strictly increasing. Setting times to default."
                )
        # custom properties
        self._custom_property_list = []
        for name, d in kwargs.items():
            if name in self.__dict__:
                warnings.warn(
                    f"Attribute name {name} is reserved.", stacklevel=2
                )
                continue
            setattr(self, name, d)
            self._custom_property_list.append(name)
        if not hasattr(self, "_comparisons"):
            self._comparisons = {}
