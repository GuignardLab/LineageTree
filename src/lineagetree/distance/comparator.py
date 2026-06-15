from __future__ import annotations

from typing import TYPE_CHECKING, Iterable
import inspect
import pickle
from warnings import warn
from multiprocessing import pool
import types
from .approximations import ApproximatedTree
import itertools
import tqdm

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
            new_key = (hash(tree[0]), tree[1], tree[2])
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
        tree1 = self.__use_approximation(tree1)
        tree2 = self.__use_approximation(tree2)
        distance = self.tree_distance.compute_distance(
            tree1, tree2, self._cached_distances
        ) / self.tree_distance.get_norm(tree1, tree2, norm)

        return distance

    def order_processes(self, proc):
        nodes1, nodes2 = len(proc[1].nodes), len(proc[2].nodes)
        return nodes1 + nodes2

    def p_compare(self, *trees, norm="sum", n_proccessors: int = 4):
        app_trees = []
        for tree in tqdm.tqdm(trees, desc="Processing Trees: "):
            app_trees.append(self.__use_approximation(tree))
        combinations = list(itertools.combinations(app_trees, 2))
        processes = (
            (self.tree_distance, comb1, comb2, norm, self._cached_distances)
            for comb1, comb2 in combinations
        )
        distances = []

        ch_size = (
            len(combinations) // n_proccessors
        ) // 4  # Maybe overengineered but it works super good
        processes_b_s = sorted(processes, key=self.order_processes)
        processes_s_b = processes_b_s[::-1]
        mid = len(processes_b_s) // 2
        high = processes_b_s[:mid]
        low = processes_s_b[mid:]
        new_proc = []
        for h, l in zip(high, low):
            new_proc.append(h)
            new_proc.append(l)

        new_proc.extend(high[len(low) :])
        with pool.Pool(min(n_proccessors, len(app_trees))) as p:
            for r in tqdm.tqdm(
                p.imap_unordered(
                    _worker, new_proc, chunksize=ch_size if ch_size > 0 else 1
                ),
                total=len(combinations),
                desc="UTED distances",
            ):
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
