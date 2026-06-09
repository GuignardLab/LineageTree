from __future__ import annotations

from typing import TYPE_CHECKING, Iterable
import inspect

from warnings import warn

if TYPE_CHECKING:
    from lineagetree import LineageTree
    from .approximations import TreeApproximationTemplate
    from .distance_calculator import TreeDistanceTemplate


class TreeComparator:

    __adding_trees = True  # Variable that allows adding trees to the object or not, the moment a dataset is added no more trees can be added.
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
        self.__id_count = 0
        self.trees = {}

    @property
    def cached_distances(
        self,
    ):
        if not hasattr(self, "_cache"):
            self._cache = {}
        return self._cache

    def _get_next_id(self) -> str:
        """Provides and id if it is not provided by `add_tree`.

        Returns
        -------
        str
            The generated id.
        """
        self.__id_count += 1
        return f"LineageTree {self.__id_count - 1}"

    def add_trees_one_dataset(
        self,
        lT: LineageTree,
        roots: Iterable,
        starts: int | list[int],
        end_time: int | list[int] = None,
    ):
        """Creates a dataset using ONLY one `Lineagetree` object for comparison.
          The user may choose if they want their dataset to have multiple starts
          or multiple end times but not both.

        Parameters
        ----------
        lT : LineageTree
            The `LineageTree` object.
        roots : Iterable
            The roots of the subtrees to be compared.
        starts : int | list[int]
            The different starting points for the dataset. If `starts` is a list the `end_time` cannot be a list.
        end_time : int | list[int], optional
            The different end times for the dataset. If `end_time` is a list the `starts` cannot be a list. If None the whole subtrees will be taken into account, by default `None`
        """
        if not self.__adding_trees:
            raise Warning("Cannot add any more trees to the Comparator.")
        if isinstance(starts, Iterable) and isinstance(end_time, Iterable):
            raise Warning(
                "Each Comparator can hold either multiple starts or multiple end_times. Not both."
            )
        if isinstance(starts, Iterable) and not isinstance(end_time, Iterable):
            for start in starts:
                selected_roots = lT.nodes_at_t(start, roots)
                self._add_trees(lT, selected_roots, end_time, level=start)
        elif not isinstance(starts, Iterable) and isinstance(
            end_time, Iterable
        ):
            for end in end_time:
                self._add_trees(lT, roots, end, level=end)

        else:
            for root in roots:
                self._add_tree(lT, root, end_time, level=1)

        self.__adding_trees = False

    def _add_trees(
        self,
        lT: LineageTree,
        roots: Iterable,
        end_time: int | list[int] | None = None,
        level=1,
    ):
        for root in roots:
            self._add_tree(lT, root, end_time, level=level)

    def add_trees_multiple_datasets(
        self, *trees: tuple[LineageTree, list[int], int|list[int], int|list[int]]
    ):
        """
        Works the same way as adding trees from one dataset, but its for multiple datasaets.

        Args:
            *trees: Variable-length collection of tuples, where each tuple contains:
                - tree (LineageTree): The lineage tree to add.
                - roots (list[int]): The selected roots of the dataset.
                - end_time (int): The ending time for the subtrees.
        """
        if not self.__adding_trees:
            raise Warning("Cannot add any more trees to the Comparator.")
        lTs = [tr[0] for tr in trees]
        roots = [tr[1] for tr in trees]
        starting_times = [tr[2] for tr in trees]
        for s_t in starting_times:
            
        end_times = [tr[3] for tr in trees]
        

    def _add_tree(
        self,
        lT: LineageTree,
        root: int,
        end_time: int | None = None,
        level: int = None,
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
        level: int, optional
            Used if the dataset has multiple levels of comparison
        """
        approximated_tree = self.approximator.approximation(lT, root, end_time)
        if isinstance(lT.name, str):
            id = lT.name
        else:
            id = self._get_next_id()
        self.trees.update({id: approximated_tree})
        self.labels.update({id: lT.labels.get(root, root)})
        if level is not None:
            self.levels_of_comparison.setdefault(level, []).append(
                approximated_tree
            )
