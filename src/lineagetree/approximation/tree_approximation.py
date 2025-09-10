from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING, Iterable, Callable
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

from edist import uted

from .deltas import (
    delta_normalized_difference,
    delta_nd_norm,
    delta_difference,
)

if TYPE_CHECKING:
    from ..lineage_tree import LineageTree
    from edist.alignment import Alignment


@dataclass
class ApproximatedTree:
    adjacency_dict: dict[int, Iterable[int]]
    property_dict: dict[int, float | int] | None

    def _edist_format(
        self,
    ) -> tuple[list, list[list], dict[int, int]]:
        """
        A function that transforms an adjacency list into
        a datastructure understandable by uted

        Returns
        -------
        nodes : list of int
            The list of node ids
        adj_list: list of list of int
            The adjacency list where `nodes[adj_list[nodes[0]][0]]`
            gives you the id of one of the successors of `nodes[0]`.
        list2nid : dict of int to int
            A dictionary that maps the new node ids (in `nodes`) to
            the original ones (from the LineageTree)
        """
        inv_adj = {vi: k for k, v in self.adjacency_dict.items() for vi in v}
        roots = set(self.adjacency_dict).difference(inv_adj)
        nid2list = {}
        list2nid = {}
        nodes = []
        adj_list = []
        curr_id = 0
        for r in roots:
            to_do = deque([r])
            while to_do:
                curr = to_do.popleft()
                nid2list[curr] = curr_id
                list2nid[curr_id] = curr
                nodes.append(self.property_dict.get(curr, 0))
                to_do.extendleft(reversed(self.adjacency_dict.get(curr, [])))
                curr_id += 1
            adj_list = [
                [
                    nid2list[d]
                    for d in self.adjacency_dict.get(list2nid[_id], [])
                ]
                for _id in range(len(nodes))
            ]
        return nodes, adj_list, list2nid

    def __post_init__(self):
        if (
            self.property_dict is not None
            and len(set(self.adjacency_dict).difference(self.property_dict))
            != 0
        ):
            raise ValueError(
                "Mismatch between adjacency_dict keys and property_dict keys.\n"
                "All nodes have to have a property."
            )
        elif self.property_dict is None:
            self.property_dict = {}
        self.nodes, self.adjacency_list, self.correspondency = (
            self._edist_format()
        )


class TreeApproximatorTemplate(ABC):
    delta: Callable = None
    available_norms = {"max", "sum"}

    @abstractmethod
    def build_approximated_tree(
        self,
        lT: LineageTree,
        root: int,
        end_time: int | None = None,
    ) -> ApproximatedTree: ...

    def compute_uted_backtrace(
        self,
        approximated_tree1: ApproximatedTree,
        approximated_tree2: ApproximatedTree,
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
        backtrace = uted.uted_backtrace(
            approximated_tree1.nodes,
            approximated_tree1.adjacency_list,
            approximated_tree2.nodes,
            approximated_tree2.adjacency_list,
            delta=self.delta,
        )
        return backtrace

    def compute_uted_distance(
        self,
        approximated_tree1: ApproximatedTree,
        approximated_tree2: ApproximatedTree,
        backtrace: Alignment = None,
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
        if backtrace is None:
            return uted.uted(
                approximated_tree1.nodes,
                approximated_tree1.adjacency_list,
                approximated_tree2.nodes,
                approximated_tree2.adjacency_list,
                delta=self.delta,
            )

        if self.delta is None:

            def delta(x1, x2):
                return 1 if x1 != x2 else 0

        else:
            delta = self.delta

        return backtrace.cost(
            approximated_tree1.nodes, approximated_tree2.nodes, delta
        )

    def get_norm(
        self,
        tree1: ApproximatedTree,
        tree2: ApproximatedTree,
        norm_type: {"max", "sum", "tuple"} = "max",
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
                    f"Invalide value for `norm_type`. Got {norm_type},"
                    "expected 'max', 'sum' or 'tuple'"
                )


class SimpleTreeTimed(TreeApproximatorTemplate):
    """
    An approximator where a lineage tree is approximated as
    a binary tree where each node holds as property the length
    of the chain it is representing.

    An approximation when cell cycle length is the main/only concern.
    This approximation is one of the fastest.

    The default delta for this approximation is the normalised difference
    """

    def build_approximated_tree(
        self,
        lT: LineageTree,
        root: int,
        end_time: int | None = None,
        time_resolution: float = None,
        *_,
        **__,
    ) -> ApproximatedTree:
        """
        Build the approximation of a given sub lineage of lineage tree.

        Parameters
        ----------
        lT : LineageTree
            The lineage tree from which to approximate
        root : int
            The id of the spawning cell of the sub lineage
            to approximate
        end_time : int
            The last time point to consider
        time_resolution : float
            How much time happens between two consecutive time points
            This is useful when comparing two different lineage trees
            that do not have the same time scale

        Returns
        -------
        ApproximatedTree
            The approximated tree that will be used
            for the computation of the tree edit distance
        """

        if end_time is None:
            end_time = lT.t_e
        if time_resolution is None:
            time_resolution = 1
        out_dict = {}
        final_properties = {}
        to_do = [root]
        while to_do:
            current = to_do.pop()
            cycle = np.array(lT.get_successors(current))
            cycle_times = np.array([lT.time[c] for c in cycle])
            cycle = cycle[cycle_times <= end_time]
            if cycle.size:
                _next = list(lT.successor[cycle[-1]])
                if len(_next) > 1 and lT.time[cycle[-1]] < end_time:
                    out_dict[current] = _next
                    to_do.extend(_next)
                else:
                    out_dict[current] = []
            final_properties[current] = len(cycle) * time_resolution

        return ApproximatedTree(out_dict, final_properties)

    def __init__(self, delta: Callable | None = None):
        self.delta = delta or delta_normalized_difference


class SimpleTreeGeneral(TreeApproximatorTemplate):
    """
    An approximator where a lineage tree is approximated as
    a binary tree where each node holds as property either a
    scalar or a vector of numbers.

    A general approximation that requires more work to take into
    account the cell cycle times but that is more versatile.
    This approximation is slightly slower than SimpleTreeTime
    due to its versatility.

    The default delta for this approximation is the L2 norm
    """

    @staticmethod
    def build_approximated_tree(
        lT: LineageTree,
        root: int,
        end_time: int | None = None,
        time_resolution: float = None,
        properties: dict[int, float] = None,
        aggregator: Callable = np.nanmean,
        *_,
        **__,
    ) -> ApproximatedTree:
        """
        Build the approximation of a given sub lineage of lineage tree.

        Parameters
        ----------
        lT : LineageTree
            The lineage tree from which to approximate
        root : int
            The id of the spawning cell of the sub lineage
            to approximate
        end_time : int
            The last time point to consider
        time_resolution : float
            How much time happens between two consecutive time points
            This is useful when comparing two different lineage trees
            that do not have the same time scale
        properties : dict mapping int to float, optional
            The dictionary that maps a node id to a given value.
            *Not* all the nodes need to have a value.
        aggregator : Callable, default: np.nanmean
            A function that takes as an input an `NxD` array
            where `N` is the number of observations
            and `D` the dimension of the measurements.
            .. warning:: The aggregator function has to have an `axis` argument

        Returns
        -------
        ApproximatedTree
            The approximated tree that will be used
            for the computation of the tree edit distance
        """

        if end_time is None:
            end_time = lT.t_e
        if time_resolution is None:
            time_resolution = 1
        out_dict = {}
        final_properties = {}
        to_do = [root]
        while to_do:
            current = to_do.pop()
            cycle = np.array(lT.get_successors(current))
            cycle_times = np.array([lT.time[c] for c in cycle])
            cycle = cycle[cycle_times <= end_time]
            if 0 < len(cycle):
                _next = list(lT.successor[cycle[-1]])
                if len(_next) > 1 and lT.time[cycle[-1]] < end_time:
                    out_dict[current] = _next
                    to_do.extend(_next)
                else:
                    out_dict[current] = []
                final_properties[current] = aggregator(
                    [properties.get(node, np.nan) for node in cycle], axis=0
                )
                if np.isnan(final_properties[current]).any():
                    final_properties[current] = np.zeros_like(
                        list(properties.values())[0]
                    )

        return ApproximatedTree(out_dict, final_properties)

    def __init__(self, delta: Callable = None):
        self.delta = delta or delta_nd_norm


class DownsampledTree(TreeApproximatorTemplate):
    """
    An approximator where a lineage tree is downsampled by a given
    downsampling value `downsample`. One every `downsample` node
    is kept in the approximated tree.
    When computing the distance, it only looks at difference in topology,
    no property dictionary can be provided.

    This approximation is halfway between the precision of the full lineage
    tree and the rapidity of the SimpleTree.

    .. warning:: This approximation should only be used to compare trees
    that have the same time resolution. When comparing two trees that have
    different time resolution, one should use `ResampledTree`

    The default delta for this approximation is simple delta
    """

    @staticmethod
    def build_approximated_tree(
        lT: LineageTree,
        root: int,
        end_time: float = None,
        downsample: int = None,
        *_,
        **__,
    ):
        """
        Build the approximation of a given sub lineage of lineage tree.

        Parameters
        ----------
        lT : LineageTree
            The lineage tree from which to approximate
        root : int
            The id of the spawning cell of the sub lineage
            to approximate
        end_time : int
            The last time point to consider
        downsample : int
            The downsampling value.
            One node will be conserved every `downsample` node

        Returns
        -------
        ApproximatedTree
            The approximated tree that will be used
            for the computation of the tree edit distance
        """
        if end_time is None:
            end_time = lT.t_e
        if downsample is None:
            downsample = 2
        out_dict = {}
        to_do = [root]
        while to_do:
            current = to_do.pop()
            _next = lT.nodes_at_t(
                r=current,
                t=lT.time[current] + downsample,
            )
            if _next == [current]:
                _next = None
            if _next and lT.time[_next[0]] <= end_time:
                out_dict[current] = _next
                to_do.extend(_next)
            else:
                out_dict[current] = []
        return ApproximatedTree(out_dict, None)

    def __init__(self, delta: Callable = None):
        self.delta = delta


class ResampledTree(TreeApproximatorTemplate):
    """
    An approximator that resample lineage trees to a given time resolution.
    The target time resolution is provided at creation.
    Each time a new approximated tree is built, its original time resolution
    has to be provided. Therefore, the resampling can be upsampling or downsampling.

    Moreover, a dictionary mapping node ids to values can be provided.
    It will be interpolated using a spline interpolation.
    If no property dictionary is provided then the simple delta will be used.

    This approximation is the go to approximation when comparing
    different lineage trees.

    The default delta for this approximation is the difference delta
    """

    def build_approximated_tree(
        self,
        lT: LineageTree,
        root: int,
        sampling_property: dict[int, float] = None,
        end_time: float = None,
        time_resolution: float = None,
        spline_smoothing: int = 3,
        *_,
        **__,
    ):
        """
        Build the approximation of a given sub lineage of lineage tree.

        Parameters
        ----------
        lT : LineageTree
            The lineage tree from which to approximate
        root : int
            The id of the spawning cell of the sub lineage
            to approximate
        sampling_property : dict mapping ints to floats, optional
            A dictionary mapping node ids to their corresponding
            values for the property considered.
            If not provided, no property will be attached to the nodes
            and therefore only the topology of the lineage tree will matter
        end_time : int
            The last time point to consider
        time_resolution : int
            The time resolution of the lineage tree
            to approximate. It has to be in the same
            units than the one provided as `target_time_resolution`.

        Returns
        -------
        ApproximatedTree
            The approximated tree that will be used
            for the computation of the tree edit distance
        """
        if time_resolution is None:
            time_resolution = 1
        if end_time is None:
            end_time = lT.t_e
        if sampling_property is None:
            sampling_property = {}

        # All chains will end up existing, at most reduced to 1 if too short
        chains = lT.get_all_chains_of_subtree(root, end_time=end_time)

        downsampling_rate = self.target_time_resolution / time_resolution
        out_dict = {}  # adjacency dictionary
        final_property = {}  # interpolated property dictionary
        # links remains to add after the first for loop (ie divisions)
        remaining_links = []
        temporary_mapping = {}  # Mapping between old and new ids

        next_id = 0  # Initial id value
        for chain in chains:

            # First computing the times of the original and new chain
            # to use for the spline interpolation
            initial_length = len(chain)
            target_length = round(initial_length / downsampling_rate)
            if target_length <= 0:
                target_length = 1
            new_chain = np.arange(target_length)
            initial_time = np.arange(initial_length) * time_resolution
            new_chain_time = new_chain * self.target_time_resolution

            # Making sure that we don't have missing values in the property values
            # to avoid weird things with the interpolation
            initial_property = []
            smoothing_time = []
            for i, node in enumerate(chain):
                if node in sampling_property:
                    initial_property.append(sampling_property[node])
                    smoothing_time.append(initial_time[i])

            if spline_smoothing < len(smoothing_time):
                interpolator = InterpolatedUnivariateSpline(
                    smoothing_time,
                    initial_property,
                    k=spline_smoothing,
                )
                interpolated_property = interpolator(new_chain_time)
            elif 0 < len(
                smoothing_time
            ):  # case when interpolation give too few nodes
                interpolated_property = [
                    0,
                ] * target_length
            else:
                interpolated_property = initial_property[:target_length]

            if len(interpolated_property) == 0:
                interpolated_property = [
                    0,
                ] * target_length

            # it can happen that the interpolated property does not
            # have enough values, so we pad its with the last value
            if len(interpolated_property) < target_length:
                interpolated_property.extend(
                    [
                        interpolated_property[-1],
                    ]
                    * (target_length - len(interpolated_property))
                )

            new_chain += next_id
            next_id = new_chain[-1] + 1

            out_dict.update(
                {
                    n: [
                        new_chain[i + 1],
                    ]
                    for i, n in enumerate(new_chain[:-1])
                }
            )

            # The strict=True ensure that we have
            # len(new_chain) == len(interpolated_property)
            final_property.update(
                zip(new_chain, interpolated_property, strict=True)
            )
            if lT.time[chain[-1]] <= end_time:
                remaining_links.append(chain[-1])
                temporary_mapping[chain[-1]] = new_chain[-1]
                temporary_mapping[chain[0]] = new_chain[0]

        for node in remaining_links:
            out_dict[temporary_mapping[node]] = list(
                temporary_mapping[nodei] for nodei in lT.successor[node]
            )

        return ApproximatedTree(out_dict, final_property)

    def __init__(
        self,
        target_time_resolution: int = 2,
        delta: Callable = None,
    ):
        self.target_time_resolution = target_time_resolution
        self.delta = delta or delta_difference


class FullTree(TreeApproximatorTemplate):
    """
    An approximator do not really approximate.
    It takes the sub tree to "approximate" as is unless
    end_time is provided.

    This approximation is the go to approximation when comparing
    different lineage trees.

    The default delta for this approximation is the unit delta
    """

    @staticmethod
    def build_approximated_tree(
        lT,
        root,
        end_time: float = None,
        property_dictionary: dict[int, float] = None,
        *_,
        **__,
    ):
        if end_time is None:
            end_time = lT.t_e
        out_dict = {}
        to_do = [root]
        while to_do:
            current = to_do.pop()
            _next = list(lT.successor[current])
            if _next:
                for _n in _next:
                    if lT.time[_n] <= end_time:
                        out_dict.setdefault(current, []).append(_n)
                        to_do.append(_n)
            else:
                out_dict[current] = []
        return ApproximatedTree(out_dict, property_dictionary)

    def __init__(
        self,
        delta: Callable = None,
    ):
        self.delta = delta


TREE_APPROXIMATORS: dict[str, type[TreeApproximatorTemplate]] = {
    "simple": SimpleTreeTimed,
    "full": FullTree,
    "downsampled": DownsampledTree,
    "normalized_simple": SimpleTreeTimed,
}
