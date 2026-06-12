from __future__ import annotations

from typing import TYPE_CHECKING, Iterable
import inspect
import pickle
from warnings import warn
from multiprocessing import pool
import types
from .approximations import ApproximatedTree
import itertools

if TYPE_CHECKING:
    from lineagetree import LineageTree
    from .approximations import TreeApproximationTemplate
    from .distance_calculator import TreeDistanceTemplate
    from edist.alignment import Alignment


class TreeComparator:

    def __init__(
        self,
        tree_distance: TreeDistanceTemplate,
        approximator: TreeApproximationTemplate | None = None,
    ):

        if approximator is None:
            warn(
                "No approximation was set so the lineageTree object itself will be added to the the Object without any changes."
            )
            self.approximator = lambda x: x
        else:
            self.approximator = approximator

        self.tree_distance = tree_distance(self.approximator.delta)
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

    def __use_approximation(
        self, tree: tuple[LineageTree, int, int] | ApproximatedTree
    ) -> ApproximatedTree:
        if isinstance(tree, tuple):
            new_key = (hash(tree[0]), *tree[1:])
            if new_key in self._cached_approximations:
                f_tree = self._cached_approximations[new_key]
            else:
                f_tree = self.approximator.approximation(*tree)
                self._cached_approximations[new_key] = f_tree
        else:
            f_tree = tree
        return f_tree

    def compare(
        self,
        tree1: tuple[LineageTree, int, int] | ApproximatedTree,
        tree2: tuple[LineageTree, int, int] | ApproximatedTree,
        norm: {"max", "sum", "tuple"} | Callable = "sum",
    ) -> float | int:
        if not isinstance(tree1, ApproximatedTree):
            tree1 = self.__use_approximation(tree1)
        if not isinstance(tree2, ApproximatedTree):
            tree2 = self.__use_approximation(tree2)
        distance = self.tree_distance.compute_distance(
            tree1, tree2, self._cached_distances
        ) / self.tree_distance.get_norm(tree1, tree2, norm)

        return distance

    def p_compare(self, *trees, norm="sum", n_proccessors: int = 4):
        app_trees = []
        for tree in trees:
            if not isinstance(tree, ApproximatedTree):
                app_trees.append(self.__use_approximation(tree))
        print("finished calc approx")
        combinations = itertools.combinations(app_trees, 2)
        proccesses = (
            (self.tree_distance, comb1, comb2, norm, self._cached_distances)
            for comb1, comb2 in combinations
        )
        with pool.Pool(
            n_proccessors if n_proccessors < len(app_trees) else len(app_trees)
        ) as p:
            res = p.map(_worker, proccesses)
        distances = []
        for r in res:
            print(r)
            self._cached_distances.update(r[0])
            distances.append(r[1])

        return distances


def _worker(args):
    (
        distance,
        app_tree1,
        app_tree2,
        norm,
        cache,
    ) = args
    res = distance.compute_distance_parallel(app_tree1, app_tree2, cache)
    distance = res[1] / distance.get_norm(app_tree1, app_tree2, norm)

    return (res[0], distance)
