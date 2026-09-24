from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING



if TYPE_CHECKING:
    from ..lineage_tree import LineageTree


class TreeApproximator(ABC):

    def build_approximated_tree(self):
        """This functionm will create the nod list and the adjacency_list"""

    def get_norm(self):
        """Return the distance of the approximated tree to the null tree."""

    def __init__(
        self,
        lT: LineageTree,
        starting_node: int,
        ending_timepoint: int | None = None,
    ) -> None:
        """_summary_

        Parameters
        ----------
        lT : LineageTree
            _description_
        starting_node : int
            _description_
        ending_timepoint : int | None, optional
            _description_, by default None
        """
