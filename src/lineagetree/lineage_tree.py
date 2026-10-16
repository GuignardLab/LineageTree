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
    """A forest of cell lineage trees, with navigation and analysis methods.

    A ``LineageTree`` is a directed forest (a set of rooted trees). Each node
    is an integer id that stands for one cell at one time point, and each edge
    links a node to its successor at a later time point. A node has at most
    one predecessor and any number of successors: one while the cell lives,
    two or more when it divides, none at a leaf.

    A *chain* is a maximal run of nodes in which every node but the last has
    exactly one successor, i.e. the life of one cell between two divisions.
    Many methods work on chains rather than on single nodes.

    The methods come from mixins, grouped by capability: structural
    properties (roots, leaves, edges, …), tree edits (adding and removing
    nodes, smoothing trajectories, …), navigation (ancestors, subtrees,
    chains, …), plotting, spatial neighbourhoods (Gabriel graph, k nearest
    neighbours, …), tree comparison (unordered tree edit distance, dynamic
    time warping) and writing to disk (pickle, SVG, Tulip).

    To create a ``LineageTree`` from a file, use one of the ``read_from_*``
    functions of the ``lineagetree`` package, or
    [`LineageTree.load`][lineagetree.LineageTree.load] for ``.lT`` files.

    Examples
    --------
    A cell (0) that divides into two daughters, each observed once:

    >>> from lineagetree import LineageTree
    >>> lT = LineageTree(successor={0: [1, 2], 1: [], 2: []})
    >>> sorted(lT.roots), sorted(lT.leaves)
    ([0], [1, 2])
    >>> dict(lT.time)
    {0: 0, 1: 1, 2: 1}
    """

    def __eq__(self, other) -> bool:
        """Compare two ``LineageTree`` objects for structural equality.

        Two trees are considered equal when their successor, predecessor, and
        time dictionaries are identical. Positions, names and custom
        properties are not compared.

        Parameters
        ----------
        other : object
            Object to compare against.

        Returns
        -------
        bool
            ``True`` if ``other`` is a ``LineageTree`` with the same topology
            and time assignment, ``False`` otherwise.
        """
        if isinstance(other, LineageTree):
            return (
                other._successor == self._successor
                and other._predecessor == self._predecessor
                and other._time == self._time
            )
        else:
            return False

    def __setstate__(self, state: dict) -> None:
        """Restore instance state from a pickle, handling legacy attribute names.

        Older pickled ``LineageTree`` objects stored the core dictionaries
        under ``successor``, ``predecessor``, and ``time`` (without the
        leading underscore).  This method remaps those to the current private
        names before calling ``__dict__.update``.

        Parameters
        ----------
        state : dict
            The unpickled ``__dict__`` of the stored object.
        """
        if "_successor" not in state:
            state["_successor"] = state["successor"]
        if "_predecessor" not in state:
            state["_predecessor"] = state["predecessor"]
        if "_time" not in state:
            state["_time"] = state["time"]
        self.__dict__.update(state)

    @classmethod
    def load(clf, fname: str):
        """Load a lineage tree from a ``.lT`` file.

        ``.lT`` files are written by
        [`LineageTree.write`][lineagetree.LineageTree.write]. Files written
        by versions older than 2.0 are converted to the current format, and
        attributes missing from older files (``time_resolution``,
        ``spatial_resolution``, ``temporal``) are set to their defaults.

        Parameters
        ----------
        fname : str
            Path to and name of the file to read.

        Returns
        -------
        LineageTree
            The loaded lineage tree.

        Warnings
        --------
        ``.lT`` files are pickles: only load files from sources you trust.
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
        lT.pos = {
            node: np.array(pos, dtype=float) for node, pos in lT.pos.items()
        }
        if not hasattr(lT, "time_resolution"):
            lT.time_resolution = 1
        if not hasattr(lT, "spatial_resolution"):
            lT.spatial_resolution = np.ones(3)
        if not hasattr(lT, "_temporal"):
            lT._temporal = True

        return lT

    def get_subtree(self, node_list: set[int]) -> LineageTree:
        """Create a new lineage tree restricted to a set of nodes.

        Only the nodes in ``node_list`` and the edges between them are kept.
        Times, positions, the name and the custom properties are carried over.
        Custom properties are passed as they are, so they may still hold
        values for nodes that were dropped. Ids that are not nodes of the
        tree are ignored.

        Parameters
        ----------
        node_list : set of int
            The nodes to keep, for example the output of
            [`get_subtree_nodes`][lineagetree.LineageTree.get_subtree_nodes].

        Returns
        -------
        LineageTree
            A new lineage tree; this one is left unchanged.
        """
        node_list = self.nodes.intersection(node_list)
        new_successors = {
            n: tuple(vi for vi in self.successor[n] if vi in node_list)
            for n in node_list
        }
        return LineageTree(
            successor=new_successors,
            time={n: self._time[n] for n in node_list},
            pos={n: self.pos[n] for n in node_list if n in self.pos},
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
        spatial_resolution: Sequence | None = None,
        temporal: bool = True,
        **kwargs,
    ):
        """Create a LineageTree from dictionaries, without a file.

        The topology is given by either ``successor`` or ``predecessor``, not
        both. Nodes that only appear as values (e.g. leaves missing from the
        keys of ``successor``) are added automatically.

        Parameters
        ----------
        successor : dict of {int: Sequence of int}, optional
            Maps each node to its successors.
        predecessor : dict of {int: int or Sequence of int}, optional
            Maps each node to its predecessor, given either as an int or as a
            sequence with at most one element.
        time : dict of {int: int}, optional
            Maps each node to the time point it was recorded at. Times must
            strictly increase along every edge. If None, roots are placed at
            ``starting_time`` and each other node one time point after its
            predecessor.
        starting_time : int, optional
            Time point of the roots when ``time`` is not given. Defaults to 0.
            Ignored, with a warning, when ``time`` is given.
        pos : dict of {int: Iterable of float}, optional
            Maps each node to its position. If given, every node needs one.
        name : str, optional
            Name of the lineage tree.
        root_leaf_value : Sequence, optional
            Values that mark a missing predecessor (root) or missing
            successors (leaf) in ``successor`` or ``predecessor``. Defaults
            to ``[None, (), [], set()]``.
        spatial_resolution : Sequence of float, optional
            Size of a unit of ``pos`` along each spatial dimension, used by
            the spatial methods. Must have one value per dimension of the
            positions. Defaults to ones.
        temporal : bool, default=True
            Whether the tree has a time dimension. Set to False for static
            trees such as neuron morphologies.
        **kwargs
            Custom node properties, each a dictionary mapping node ids to
            values, e.g. ``volume={0: 12.5, 1: 11.0}``. Each one becomes an
            attribute of the tree (``lT.volume``). A name already used by a
            ``LineageTree`` attribute is skipped with a warning.

        Raises
        ------
        ValueError
            If both ``successor`` and ``predecessor`` are given, if a node has
            more than one predecessor, if the graph has a cycle, if ``pos``
            misses a node, if times do not strictly increase along an edge,
            or if ``spatial_resolution`` does not match the dimension of the
            positions.
        TypeError
            If a successor entry or ``root_leaf_value`` is not iterable.
        """
        self.__version__ = importlib.metadata.version("lineagetree")
        self.name = str(name) if name is not None else None
        self._temporal = temporal

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
                node: np.array(position, dtype=float)
                for node, position in pos.items()
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

        spatia_dimension = (
            len(self.pos[next(iter(self.nodes))])
            if self.nodes and self.pos
            else 3
        )
        if self.nodes and spatial_resolution is not None:
            if len(spatial_resolution) == spatia_dimension:
                self.spatial_resolution = np.array(spatial_resolution)
            else:
                raise ValueError(
                    "The spatial resolution should have the same dimension as the one of the positions:\n"
                    f"{len(spatial_resolution)=}, spatial dimension={spatia_dimension}"
                )
        else:
            self.spatial_resolution = spatial_resolution or np.ones(
                spatia_dimension
            )
