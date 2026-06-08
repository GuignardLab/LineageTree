from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING, Iterable, Callable
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from hashlib import sha256
import pickle

from edist import uted

from .delta import (
    delta_normalized_difference,
    delta_nd_norm,
    delta_difference,
    delta_binary,
)
from warnings import warn

if TYPE_CHECKING:
    from lineagetree import LineageTree
    from edist.alignment import Alignment
    from .approximations import TreeApproximationTemplate
    from .distance_calculator import TreeDistanceTemplate


class TreeComparator:

    def __init__(
        self,
        tree_distance: TreeDistanceTemplate,
        approximator: TreeApproximationTemplate | None = None,
    ):
        self.approximator = approximator() if approximator else lambda x: x
        if approximator is None:
            warn(
                "No approximation was set so the lineageTree object itself will be added to the the Object without any changes."
            )
            self.approximator = lambda x: x
        else:
            self.approximator = approximator

        self.tree_distance = tree_distance
        self.labels = {}
        self.__id_count = 0
        self.trees = {}

    @property
    def cached_distances(
        self,
    ):
        if not hasattr(self, "_cache"):
            self._cache = {}
        return self._cache

    def _get_next_id(self) -> int:
        """Provides and id if it is not provided by `add_tree`.

        Returns
        -------
        int
            The generated id.
        """
        self.__id_count += 1
        return self.__id_count - 1

    def add_trees_from_dif_starts(self, lT, starts, emd_time): ...

    def add_multiple_trees(self, *trees): ...

    def add_tree(
        self,
        lT: LineageTree,
        root: int,
        end_time: int | None = None,
        name: str = None,
    ):
        """Handles adding a tree to the object, by calling the `approximation` method and assigning an id to the tree.
        This function also saves the labels.

        Parameters
        ----------
        lT : LineageTree
            The LineageTree object
        root : int
            The root of the tree to be added, can be any nodein the `lT` object.
        end_time : int | None, optional
            The final timepoint of the subtree to be taken into account, by default None
        time_resolution : float, optional
            The time_resolution, by default None
        name: str, optional
            The name of the tree to be added, if None an id will be generated, which will serve as its name, by default None
        """
        approximated_tree = self.approximator.approximation(
            lT, root, end_time, time_resolution
        )
        if isinstance(name, str):
            id = name
        else:
            id = self._get_next_id()
        self.trees.update({id: approximated_tree})
        self.labels.update({id: lT.labels.get(root, root)})

    def remove_tree(self, tree: int | str):
        if not isinstance(tree, str):
            raise Warning(
                f"Trees are removed form the structure using their names (str)"
            )
        if tree in self.trees:
            self.trees.pop(tree)
