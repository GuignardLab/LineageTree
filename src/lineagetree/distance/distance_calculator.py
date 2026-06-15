from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable
from typing import TYPE_CHECKING
from edist import uted
from warnings import warn

if TYPE_CHECKING:
    from edist.alignment import Alignment
    from .approximations import ApproximatedTree
    from lineagetree import LineageTree


class TreeDistanceTemplate(ABC):

    def compute_distance_parallel(
        self,
        approximated_tree1: ApproximatedTree | LineageTree,
        approximated_tree2: ApproximatedTree | LineageTree,
        backtrace: Alignment = None,
    ) -> dict:
        warn(
            "Parallel algorithm not implemented, defaulting to single process."
        )
        self.compute_distance(
            approximated_tree1, approximated_tree2, backtrace
        )

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

    def compute_distance_parallel(
        self,
        approximated_tree1: ApproximatedTree,
        approximated_tree2: ApproximatedTree,
        backtrace: Alignment = None,
    ):
        key = frozenset({str(approximated_tree1), str(approximated_tree2)})
        approximated_tree1, approximated_tree2 = sorted(
            (approximated_tree1, approximated_tree2), key=lambda x: str(x)
        )
        align = self._compute_uted_backtrace(
            approximated_tree1, approximated_tree2, backtrace
        )

        return (
            {key: align},
            self.compute_distance(
                approximated_tree1, approximated_tree2, backtrace
            ),
        )

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
        key = frozenset({str(approximated_tree1), str(approximated_tree2)})

        approximated_tree1, approximated_tree2 = sorted(
            (approximated_tree1, approximated_tree2), key=lambda x: str(x)
        )

        if cache is not None and key in cache:
            return cache[key]

        backtrace = uted.uted_backtrace(
            approximated_tree1.nodes,
            approximated_tree1.adjacency_list,
            approximated_tree2.nodes,
            approximated_tree2.adjacency_list,
            delta=self.delta,
        )

        if cache is not None:
            cache[key] = backtrace

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
        approximated_tree1, approximated_tree2 = sorted(
            (approximated_tree1, approximated_tree2), key=lambda x: str(x)
        )
        return self._compute_uted_backtrace(
            approximated_tree1, approximated_tree2, btrc
        ).cost(approximated_tree1.nodes, approximated_tree2.nodes, self.delta)

    def reconstruct_backtrace(self, tree1, tree2, backtrace): ...

    def get_norm(
        self,
        tree1: ApproximatedTree,
        tree2: ApproximatedTree,
        norm_type: {"max", "sum", "tuple"} | Callable = "sum",
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
        if isinstance(norm_type, Callable):
            return norm_type(distance_to_none1, distance_to_none2)
        match norm_type.lower():
            case "max":
                return max(distance_to_none1, distance_to_none2)
            case "sum":
                return distance_to_none1 + distance_to_none2
            case "None" | None:
                return 1
            case "tuple":
                return distance_to_none1, distance_to_none2
            case _:
                raise ValueError(
                    f"Invalid value for `norm_type`. Got {norm_type},"
                    "expected 'max', 'sum' or 'tuple'"
                )
