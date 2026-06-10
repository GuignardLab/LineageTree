from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from edist import uted

if TYPE_CHECKING:
    from edist.alignment import Alignment
    from .approximations import ApproximatedTree
    from lineagetree import LineageTree


class TreeDistanceTemplate(ABC):
    backtrace_based = (
        ...
    )  # If your distance is and edit distance that may provide the alignments between nodes set to `True`, else `False`.

    @abstractmethod
    def compute_distance(
        self,
        approximated_tree1: ApproximatedTree | LineageTree,
        approximated_tree2: ApproximatedTree | LineageTree,
        backtrace: Alignment = None,
    ) -> float: ...

    @abstractmethod
    def get_norm(
        self,
        tree1: ApproximatedTree,
        tree2: ApproximatedTree,
        norm_type: {"max", "sum", "tuple"} = "sum",
    ) -> float: ...


class UnorderedTreeEditDistance(TreeDistanceTemplate):

    backtrace_based = True

    def __init__(self, delta):
        super().__init__()
        self.delta = delta
        self.backtrace = {}

    def _compute_uted_backtrace(
        self,
        approximated_tree1: ApproximatedTree,
        approximated_tree2: ApproximatedTree,
        cache: dict[frozenset[tuple], Alignment] | None = None,
    ) -> Alignment:
        """
        Computes the optimal mapping between
        `approximated_tree1` and `approximated_tree2`
        according to the delta function predetermined

        Parameters
        ----------
        approximated_tree1 : ApproximatedTree
            The first tree to compare
        approximated_tree2 : ApproximatedTree
            The second tree to compare

        Returns
        -------
        backtrace : edist.alignment.Alignment
            The resulting alignment in the edist format
        """
        if cache:
            key = frozenset(
                {approximated_tree1.tree_specs, approximated_tree2.tree_specs}
            )
            if key not in cache:
                cache[key] = uted.uted_backtrace(
                    approximated_tree1.nodes,
                    approximated_tree1.adjacency_list,
                    approximated_tree2.nodes,
                    approximated_tree2.adjacency_list,
                    delta=self.delta,
                )
                backtrace = cache[key]
        else:
            backtrace = uted.uted_backtrace(
                approximated_tree1.nodes,
                approximated_tree1.adjacency_list,
                approximated_tree2.nodes,
                approximated_tree2.adjacency_list,
                delta=self.delta,
            )
        return backtrace

    def compute_distance(
        self,
        approximated_tree1: ApproximatedTree,
        approximated_tree2: ApproximatedTree,
        btrc: dict[frozenset[tuple], Alignment] | None = None,
    ) -> float:
        """
        Computes the unordered edit distance
        between two approximated lineage trees.

        It can take as an input the backtrace
        if it was already computed

        Parameters
        ----------
        approximated_tree1 : ApproximatedTree
            The first tree to compare
        approximated_tree2 : ApproximatedTree
            The second tree to compare
        backtrace : Alignment, optional
            The precomputed alignement between
            the two trees

        Returns
        -------
        float
            The unordered tree edit distance
            between the two trees.
        """
        return self._compute_uted_backtrace(
            approximated_tree1, approximated_tree2, btrc
        ).cost(approximated_tree1.nodes, approximated_tree2.nodes, self.delta)

    def reconstruct_backtrace(self, tree1, tree2, backtrace): ...

    def get_norm(
        self,
        tree1: ApproximatedTree,
        tree2: ApproximatedTree,
        norm_type: {"max", "sum", "tuple"} = "sum",
    ) -> float:
        """
        Computes the normalisation value for the
        unordered tree edit distance between `tree1` and `tree2`

        The normalisation always involve the distance between
        either of the trees to the empty tree (d(tree, ø)).

        If the norm type is "max" then
        the max of d(tree1, ø) and d(ø, tree2) is returned

        If the norm type is "sum", then
        d(tree1, ø) + d(ø, tree2) is returned

        Parameters
        ----------
        tree1 : ApproximatedTree
            The first tree to compute the normalisation
        tree2 : ApproximatedTree
            The second tree to compute the normalisation
        norm_type : {"max", "sum", "tuple"}
            How to combine the two distances (see above)
            If "tuple" is provided, return the two raw values

        Returns
        -------
        float
            The normalisation value
        """
        distance_to_none1 = uted.uted(
            tree1.nodes, tree1.adjacency_list, [], [], delta=self.delta
        )
        distance_to_none2 = uted.uted(
            [], [], tree2.nodes, tree2.adjacency_list, delta=self.delta
        )

        match norm_type.lower():
            case "max":
                return max(distance_to_none1, distance_to_none2)
            case "sum":
                return distance_to_none1 + distance_to_none2
            case "tuple":
                return distance_to_none1, distance_to_none2
            case _:
                raise ValueError(
                    f"Invalid value for `norm_type`. Got {norm_type},"
                    "expected 'max', 'sum' or 'tuple'"
                )
