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
from pathlib import Path
import csv

if TYPE_CHECKING:
    from lineagetree import LineageTree
    from .approximations import TreeApproximationTemplate
    from .distance_calculator import TreeDistanceTemplate
    from edist.alignment import Alignment


def _worker(args):
    (
        distance,
        app_tree1,
        app_tree2,
        cache,
    ) = args
    res = distance.compute_distance_parallel(app_tree1, app_tree2, cache)

    return res


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
        if not hasattr(self, "_approximations"):
            self._approximations = {}
        return self._approximations

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

    def _order_processes(self, proc):
        nodes1, nodes2 = len(proc[1].nodes), len(proc[2].nodes)
        return nodes1 + nodes2

    def p_compare(
        self,
        *trees,
        norm="sum",
        n_proccessors: int = 4,
        path: Path | None = None,
    ):
        if path:
            for tree in trees:
                if tree[0].name is None:
                    raise Warning(
                        f"All LineageTrees should have a name for saving the comparisons. Missing name: {tree[0]}"
                    )
        app_trees = []
        for tree in tqdm.tqdm(trees, desc="Processing Trees: "):
            app_trees.append(self.__use_approximation(tree))
        combinations = list(itertools.combinations(app_trees, 2))
        processes = [
            (self.tree_distance, comb1, comb2, self._cached_distances)
            for comb1, comb2 in combinations
        ]
        distances = {}

        ch_size = (
            len(combinations) // n_proccessors
        ) // 4  # Maybe overengineered but it works super good
        processes_b_s = sorted(processes, key=self._order_processes)
        # processes_s_b = processes_b_s[::-1]
        mid = len(processes_b_s) // 2
        high = processes_b_s[:mid]
        low = processes_b_s[mid:]
        new_proc = []
        for h, l in zip(high, low):
            new_proc.append(h)
            new_proc.append(l)

        if len(high) != len(low):
            new_proc.append(low[-1])
        assert len(new_proc) == len(processes)
        with pool.Pool(min(n_proccessors, len(app_trees))) as p:
            for r in tqdm.tqdm(
                p.imap_unordered(
                    _worker, processes, chunksize=ch_size if ch_size > 0 else 1
                ),
                total=len(combinations),
                desc="UTED distances",
            ):
                self._cached_distances.update(r[0])
                distances.update(r[1])

                # if path: # needs discussion
                #     with open(path, "w", newline="") as f:
                #         writer = csv.writer(f)
                #         writer.writerow((r[0].keys(), r[1]))

        return distances

    @property
    def __get_next_lineage(self):
        if not hasattr(self, "__id"):
            self.__id = 0
        self.__id += 1
        return f"lT: {self.__id-1}"  # if you start naming your datasets lT 1, 2 .... fuck you.

    def plot_clustermap(self, *trees, norm, ax=None, **kwargs):
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
        import numpy as np
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        lts = {tree[0] for tree in trees}
        one_lineage = False
        if len(lts) == 1:
            one_lineage = True
        hash_to_name = {hash(lt): lt for lt in lts}
        res = self.p_compare(*trees, **kwargs)
        only_trees = set()
        for tree in res.keys():
            only_trees.update(tree)
        matrix = np.zeros((len(trees), len(trees)))
        for i, t1 in enumerate(only_trees):
            for j, t2 in enumerate(only_trees):
                if i != j:
                    matrix[i, j] = matrix[j, i] = res[frozenset({t1, t2})]
                else:
                    matrix[i, j] = 0
        if one_lineage:
            labels = [
                (
                    hash_to_name[tr.tree_specs.lT].labels.get(
                        tr.tree_specs.root, tr.tree_specs.root
                    )
                )
                for tr in only_trees
            ]
        else:
            labels = [
                str(
                    getattr(
                        hash_to_name[tr.lT], "name", self.__get_next_lineage
                    )
                )
                + (
                    hash_to_name[tr.tree_specs.lT].labels.get(
                        tr.tree_specs.root, tr.tree_specs.root
                    )
                )
                for tr in only_trees
            ]
        condensed_dist_matrix = squareform(matrix)

        linkage_data = linkage(condensed_dist_matrix, method="ward")
        order = dendrogram(linkage_data, no_plot=True)["leaves"]
        labels = [labels[i] for i in order]
        print(labels)
        plot = np.ix_(order, order)
        plt.imshow(matrix[plot], **kwargs)
        ax.set_xticks(np.arange(len(labels)), labels=labels)
        ax.set_yticks(np.arange(len(labels)), labels=labels)
        ax.tick_params(axis="both", labelsize=10)
        plt.setp(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        return fig, ax

    def compare_all_trees_that_start_at_t(
        self, lT: LineageTree, roots, time, end_time
    ):
        if roots:
            new_roots = lT.nodes_at_t(
                time,
                roots,
            )
        else:
            new_roots = lT.time_nodes[time]
        trees = [(lT, r, end_time) for r in new_roots]
        self.p_compare(*trees)

    def plot_tree_distance_graph(
        self,
        tree1,
        tree2,
    ): ...
