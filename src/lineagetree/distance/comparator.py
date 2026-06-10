from __future__ import annotations

from typing import TYPE_CHECKING, Iterable
import inspect

from warnings import warn

if TYPE_CHECKING:
    from lineagetree import LineageTree
    from .approximations import TreeApproximationTemplate
    from .distance_calculator import TreeDistanceTemplate
    from .approximations import ApproximatedTree
    from edist.alignment import Alignment


class TreeComparator:

    levels_of_comparison = {}

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
        self.trees = {}

    @property
    def _cached_distances(
        self,
    ) -> dict[frozenset[tuple[int, int, int]], Alignment | float | int]:
        if not hasattr(self, "_cache"):
            self._cache = {}
        return self._cache

    @property
    def _cached_approximations(
        self,
    ) -> dict[tuple[int, int, int], ApproximatedTree]:
        if not hasattr(self, "approximations"):
            self.__approximations = {}
        return self.__approximations

    def use_approximation(
        self, tree: tuple[LineageTree, int, int] | ApproximatedTree
    ) -> ApproximatedTree:
        print("peos", tree)
        if isinstance(tree, tuple):
            new_key = (hash(tree[0]), *tree[1:])
            if new_key in self._cached_approximations:
                f_tree = self._cached_approximations[new_key]
            else:
                print("edw", tree)
                f_tree = self.approximator.approximation(*tree)
                self._cached_approximations[new_key] = f_tree
        else:
            f_tree = tree
        return f_tree

    def compare(
        self,
        tree1: tuple[LineageTree, int, int] | ApproximatedTree,
        tree2: tuple[LineageTree, int, int] | ApproximatedTree,
        norm="sum",
    ) -> float | int:
        tree1 = self.use_approximation(tree1)
        tree2 = self.use_approximation(tree2)
        distance = self.tree_distance.compute_distance(
            tree1, tree2, self._cached_distances
        ) / self.tree_distance.get_norm(tree1, tree2, norm)

        return distance
