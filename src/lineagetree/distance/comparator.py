from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import warn
from multiprocessing import pool
from .approximations import ApproximatedTree
import itertools
import tqdm
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from lineagetree import LineageTree
    from .approximations import TreeApproximationTemplate
    from .distance_calculator import TreeDistanceTemplate
    from edist.alignment import Alignment
    from typing import Callable
    import matplotlib


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
    """Handler class for comparing lineages. The parameters of the initializer
    are the tree distance algorithm (A class that dictates how the lineages are to be compared)
    and the approximator (A class that approximates/transforms the lineage to the format appropriate
    for comparing lineages)
    """

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
    ) -> dict[
        frozenset[tuple[ApproximatedTree, ApproximatedTree]],
        Alignment | float | int,
    ]:
        """This cache is where the distances are saved. For alignment based distances usually the
        alignment is saved instead.

        Returns
        -------
        dict[frozenset[tuple[ApproximatedTree, ApproximatedTree], Alignment | float | int]
            A dict that contains a frozenset of 2 approximated trees mapped to their distance/alignment.
        """
        if not hasattr(self, "_cache"):
            self._cache = {}
        return self._cache

    @property
    def _cached_approximations(
        self,
    ) -> dict[tuple[int, int, int], ApproximatedTree]:
        """The cache of the approximations."""
        if not hasattr(self, "_approximations"):
            self._approximations = {}
        return self._approximations

    def __use_approximation(
        self, tree: tuple[LineageTree, int, int] | ApproximatedTree
    ) -> ApproximatedTree:
        """Helping function that takes a lineagetree a node and an end_time to create
        tne approximated_tree. If an approximated tree is inserted instead just return that.

        Parameters
        ----------
        tree : tuple[LineageTree, int, int] | ApproximatedTree
               - A tuple ``(lineage_tree, root, end_time)`` containing:
                    - ``lineage_tree``: a ``LineageTree`` instance.
                    - ``root``: the root node identifier.
                    - ``end_time``: the final time step.
                - An ``ApproximatedTree`` instance.

        Returns
        -------
        ApproximatedTree
            The approximated tree object.
        """
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
        norm: Literal["max", "sum", "tuple"] | Callable = "sum",
    ) -> float | int:
        """Compares two trees.

        Parameters
        ----------
        tree1 : tuple[LineageTree, int, int] | ApproximatedTree
            tree : tuple[LineageTree, int, int] | ApproximatedTree
               - A tuple ``(lineage_tree, root, end_time)`` containing:
                    - ``lineage_tree``: a ``LineageTree`` instance.
                    - ``root``: the root node identifier.
                    - ``end_time``: the final time step.
                - An ``ApproximatedTree`` instance.
        tree2 : tuple[LineageTree, int, int] | ApproximatedTree
            tree : tuple[LineageTree, int, int] | ApproximatedTree
               - A tuple ``(lineage_tree, root, end_time)`` containing:
                    - ``lineage_tree``: a ``LineageTree`` instance.
                    - ``root``: the root node identifier.
                    - ``end_time``: the final time step.
                - An ``ApproximatedTree`` instance.
        norm : `max`,`sum` or tuple, by default `sum`
            How the distances should be normalized, by default "sum"

        Returns
        -------
        float | int
            The distance between 2 trees
        """
        tree1 = self.__use_approximation(tree1)
        tree2 = self.__use_approximation(tree2)
        distance = self.tree_distance.compute_distance(
            tree1, tree2, self._cached_distances
        ) / self.tree_distance.get_norm(tree1, tree2, norm)

        return distance

    def p_compare(
        self,
        *trees,
        norm: Literal["max", "sum", "tuple"] | Callable = "sum",
        n_processors: int = 4,
    ):
        """Compute pairwise distances between multiple trees in parallel.

        Parameters
        ----------
        *trees : tuple[LineageTree, int, int] | ApproximatedTree
            Trees to compare.

            Each tree can be either:

            - A tuple ``(lineage_tree, root, end_time)``, where:
                - ``lineage_tree`` is a ``LineageTree`` instance.
                - ``root`` is the identifier of the root node.
                - ``end_time`` is the final time step to consider.
            - An ``ApproximatedTree`` instance.

        norm : {"max", "sum", "tuple"} | Callable, default="sum"
            Normalization strategy applied to the computed distances.
            May be one of ``"max"``, ``"sum"``, ``"tuple"``, or a custom
            callable.

        n_proccessors : int, default=4
            Number of worker processes used to perform the comparisons.

        Returns
        -------
        list[float | int]
            Distances computed for all requested tree comparisons.
        """
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
            len(combinations) // n_processors
        ) // 4  # Maybe overengineered but it works super good
        processes_b_s = sorted(processes, key=_order_processes)
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
        with pool.Pool(min(n_processors, len(app_trees))) as p:
            for r in tqdm.tqdm(
                p.imap_unordered(
                    _worker, processes, chunksize=ch_size if ch_size > 0 else 1
                ),
                total=len(combinations),
                desc="UTED distances",
            ):
                self._cached_distances.update(r[0])
                distances.update(r[1])

        return {
            key: dist / self.tree_distance.get_norm(*key, norm)
            for key, dist in distances.items()
        }

    @property
    def __get_next_lineage(self) -> str:
        """Returns a name for a nameless dataset."""
        if not hasattr(self, "__id"):
            self.__id = 0
        self.__id += 1
        return f"lT: {self.__id-1}"  # if you start naming your datasets lT 1, 2 .... fuck you.

    def plot_clustermap(
        self,
        *trees,
        norm: Literal["max", "sum", "tuple"] | Callable = "sum",
        ax=None,
        **kwargs,
    ):
        """Compute pairwise tree distances using `p_compare` and visualize them
         as a hierarchical clustermap.

         Parameters
         ----------
         *trees : tuple[LineageTree, int, int] | ApproximatedTree
             Trees to compare.

             Each tree can be either:

             - A tuple ``(lineage_tree, root, end_time)``, where:
                 - ``lineage_tree`` is a ``LineageTree`` instance.
                 - ``root`` is the identifier of the root node.
                 - ``end_time`` is the final time step.
             - An ``ApproximatedTree`` instance.

         norm : {"max", "sum", "tuple"} | Callable, by default="sum"
             Normalization method, by default `sum

         ax : matplotlib.axes.Axes, optional
             Matplotlib axis to draw the clustermap on. If None, a new figure
             is created, by default `None

         **kwargs :
             Additional keyword arguments forwarded to the plt.imshow function.

         Returns
        fig : matplotlib.figure.Figure
             The matplotlib figure containing the clustermap.

         ax : matplotlib.axes.Axes
             The axes object associated with the clustermap.
        """

        if ax is None:
            fig, ax = plt.subplots()

        lts = {tree[0] for tree in trees}
        one_lineage = False
        if len(lts) == 1:
            one_lineage = True
        hash_to_name = {hash(lt): lt for lt in lts}
        res = self.p_compare(*trees, norm=norm, **kwargs)
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

        plot = np.ix_(order, order)
        _pl = plt.imshow(matrix[plot], **kwargs)
        self.colorbar = fig.colorbar(_pl, ax=ax)
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
        self,
        lT: LineageTree,
        time: int,
        norm: Literal["max", "sum", "tuple"] | Callable = "sum",
        roots: list | None = None,
        end_time: int | None = None,
        n_processors=4,
    ) -> dict[frozenset[tuple[ApproximatedTree, ApproximatedTree]], float]:
        """
        Compare all trees that originate from a given lineage tree at a specific time.

        This function computes pairwise distances between all subtrees of a given
        lineage tree that start at a specified time point.

        Parameters
        ----------
        lT : LineageTree
            The input lineage tree from which comparisons will be derived.

        time : int
            The time point at which to extract subtrees for comparison.

        norm : {"max", "sum", "tuple"} | Callable, by default="sum"
            Normalization method, by default `sum`

        roots : list of int or None, optional
            Specific root node IDs to include in the comparison.
            If None, all valid roots at the given time are used.

        end_time : int or None, optional
            Optional cutoff time for subtree extraction.
            If None, uses the full available time range.

        n_processors : int, default=4
            Number of parallel processes used for computing pairwise distances.

        Returns
        -------
        dict
            A dictionary where keys are frozensets of TreeSpecs pairs and
            values are the corresponding pairwise distances.
        """

        if roots:
            new_roots = lT.nodes_at_t(
                time,
                roots,
            )
        else:
            new_roots = lT.time_nodes[time]
        trees = [(lT, r, end_time) for r in new_roots]
        return self.p_compare(*trees, norm=norm, n_processors=n_processors)

    def clustermap__all_trees_that_start_at_t(
        self,
        lT: LineageTree,
        time,
        norm: Literal["max", "sum", "tuple"] | Callable = "sum",
        roots=None,
        end_time=None,
        n_processors=4,
        **kwargs,
    ):
        """
        Compute and visualize a clustermap of all tree distances starting at a given time.

        Parameters
        ----------
        lT : LineageTree
            Input lineage tree from which subtrees are extracted.

        time : int
            Time point at which subtree extraction begins.

        norm : {"max", "sum", "tuple"} or callable, default="sum"
           Normalization method, by default `sum`

        roots : list, optional
            Specific root nodes to include in the analysis. If None, all
            nodes at a specific timepoint are used, by default None

        end_time : int, optional
            Optional cutoff time for subtree extraction. If None, uses the whole tree, by default None

        n_processors : int, default=4
            Number of parallel workers used for distance computation.

        **kwargs :
            Additional keyword arguments passed to matplotlib.pyplot.imshow function

        Returns
        -------
        matplotlib.axes.Axes
            The axes object containing the clustermap visualization.

        Notes
        -----
        - Internally computes pairwise distances using a parallelized backend.
        - The resulting matrix is hierarchically clustered for visualization.
        - Output ordering is determined by linkage dendrogram reordering.
        """

        if roots:
            new_roots = lT.nodes_at_t(
                time,
                roots,
            )
        else:
            new_roots = lT.time_nodes[time]
        trees = [(lT, r, end_time) for r in new_roots]
        return self.plot_clustermap(
            *trees, norm=norm, n_processors=n_processors, kwargs=kwargs
        )

    def plot_tree_distance_graph(
        self,
        tree1,
        tree2,
    ): ...


def _order_processes(proc):
    """Calculates the total number of nodes each process in p_comparisons has to deal with."""
    nodes1, nodes2 = len(proc[1].nodes), len(proc[2].nodes)
    return nodes1 + nodes2
