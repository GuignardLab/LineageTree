from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING, Iterable, Callable
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


from .delta import (
    delta_normalized_difference,
    delta_nd_norm,
    delta_difference,
    delta_binary,
)

if TYPE_CHECKING:
    from lineagetree import LineageTree
    from .approximations import TreeApproximationTemplate


@dataclass
class TreeSpecs:
    lT: int
    root: int
    end_time: int

    def __str__(self):
        return (
            f"TreeSpecs("
            f"lt={self.lT}, "
            f"root={self.root}, "
            f"end_time={self.end_time}"
            f")"
        )

    def __hash__(self):
        return hash((self.lT, self.root, self.end_time))


@dataclass
class ApproximatedTree:
    adjacency_dict: dict[int, Iterable[int]]
    property_dict: dict[int, float | int | list[int | float]] | None
    tree_specs: TreeSpecs | None = None

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
        if self.property_dict:
            sample = next(iter(self.property_dict.values()))
            if isinstance(sample, Iterable):
                length = len(sample)
                default_value = [0] * length
            elif isinstance(sample, int | float):
                default_value = 0
            else:
                default_value = 0

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
                if self.property_dict:
                    nodes.append(self.property_dict.get(curr, default_value))
                else:
                    nodes.append(0)
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

    def __str__(self):  # For quickly checking how the object was created.
        return str(self.tree_specs)


class TreeApproximationTemplate(ABC):
    default_delta: Callable = ...

    def __init__(
        self,
        delta: Callable | None = None,
    ):
        self.delta = delta if delta else self.__class__.default_delta  # :(

    @abstractmethod
    def approximation(
        self,
        lT: LineageTree,
        root: int,
        end_time: int | None = None,
    ) -> ApproximatedTree: ...


class ReducedTreeTimed(TreeApproximationTemplate):
    default_delta = delta_normalized_difference

    def approximation(
        self,
        lT: LineageTree,
        root: int,
        end_time: int | None = None,
    ) -> ApproximatedTree:
        if end_time is None:
            end_time = lT.t_e
        if lT.time_resolution == 0:
            raise Warning(
                "Time resolution of `LineageTree` object cannot be `0`."
            )
        if lT.time_resolution is None:
            time_resolution = 1
        else:
            time_resolution = lT.time_resolution
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
            final_properties[current] = len(cycle) * (time_resolution)
        return ApproximatedTree(
            out_dict, final_properties, TreeSpecs(hash(lT), root, end_time)
        )


class ReducedTreeProperties(TreeApproximationTemplate):

    default_delta = delta_nd_norm

    def __init__(self, aggregator: Callable = np.nanmean, delta=None):
        super().__init__(delta)
        self.aggregator = aggregator

    def approximation(
        self,
        lT: LineageTree,
        root: int,
        end_time=None,
        properties: dict[int, float | list] | list[str] = None,
    ):
        """Build the approximation of a given sub lineage of lineage tree.


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

        Returns
        -------
        ApproximatedTree
            The approximated tree that will be used
            for the computation of the tree edit distance
        """

        if end_time is None:
            end_time = lT.t_e
        if lT.time_resolution == 0:
            raise Warning(
                "Time resolution of `LineageTree` object cannot be `0`."
            )
        out_dict = {}
        final_properties = {}
        if isinstance(properties, list):
            prop_dicts = [getattr(lT, prop) for prop in properties]
            properties = {
                node: [prop.get(node, np.nan) for prop in prop_dicts]
                for node in lT.nodes
            }

        default_value = np.zeros_like(next(iter(properties.values())))
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
                final_properties[current] = self.aggregator(
                    [properties.get(node, np.nan) for node in cycle],
                    axis=0,
                )
                if np.isnan(final_properties[current]).any():
                    final_properties[current] = default_value

        return ApproximatedTree(
            out_dict, final_properties, TreeSpecs(hash(lT), root, end_time)
        )


class DownsampledTree(TreeApproximationTemplate):
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

    default_delta = delta_binary

    def __init__(self, downsample: int = 2, delta=None):
        super().__init__(delta)
        if not isinstance(downsample, int):
            raise Warning("Please put a valid downsampling value.")
        self.downsample = downsample

    def approximation(
        self,
        lT: LineageTree,
        root: int,
        end_time: int | None = None,
        properties: dict[int, float | list] | list[str] = None,
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
        out_dict = {}
        if isinstance(properties, list):
            prop_dicts = [getattr(lT, prop) for prop in properties]
            properties = {
                node: [prop.get(node, np.nan) for prop in prop_dicts]
                for node in lT.nodes
            }
        to_do = [root]
        while to_do:
            current = to_do.pop()
            _next = lT.nodes_at_t(
                r=current,
                t=lT.time[current] + self.downsample,
            )
            if _next == [current]:
                _next = None
            if _next and lT.time[_next[0]] <= end_time:
                out_dict[current] = _next
                to_do.extend(_next)
            else:
                out_dict[current] = []
        return ApproximatedTree(
            out_dict, properties, TreeSpecs(hash(lT), root, end_time)
        )


class ResampledTree(TreeApproximationTemplate):
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

    default_delta = delta_difference

    def __init__(
        self,
        delta=None,
        target_time_resolution: int = 2,
        spline_smoothing: int = 3,
    ):
        super().__init__(delta)
        self.spline_smoothing = spline_smoothing
        self.target_time_resolution = target_time_resolution

    def approximation(
        self,
        lT: LineageTree,
        root: int,
        end_time: int | None = None,
        properties: dict[int, float | list] | list[str] = None,
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

        if lT.time_resolution is None:
            lT.time_resolution = 1
        if end_time is None:
            end_time = lT.t_e
        if properties is None:
            properties = {}
        if isinstance(properties, list):
            prop_dicts = [getattr(lT, prop) for prop in properties]
            properties = {
                node: [prop.get(node, np.nan) for prop in prop_dicts]
                for node in lT.nodes
            }

        # All chains will end up existing, at most reduced to 1 if too short
        chains = lT.get_all_chains_of_subtree(root, end_time=end_time)

        downsampling_rate = self.target_time_resolution / lT.time_resolution
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
            initial_time = np.arange(initial_length) * lT.time_resolution
            new_chain_time = new_chain * self.target_time_resolution

            # Making sure that we don't have missing values in the property values
            # to avoid weird things with the interpolation
            initial_property = []
            smoothing_time = []
            for i, node in enumerate(chain):
                if node in properties:
                    initial_property.append(properties[node])
                    smoothing_time.append(initial_time[i])

            if self.spline_smoothing < len(smoothing_time):
                interpolator = InterpolatedUnivariateSpline(
                    smoothing_time,
                    initial_property,
                    k=self.spline_smoothing,
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

        return ApproximatedTree(
            out_dict, final_property, TreeSpecs(hash(lT), root, end_time)
        )


class FullTree(TreeApproximationTemplate):
    """
    An approximator do not really approximate.
    It takes the sub tree to "approximate" as is unless
    end_time is provided.

    This approximation is the go to approximation when comparing
    different lineage trees.

    The default delta for this approximation is the unit delta
    """

    default_delta = delta_binary

    def __init__(self, delta=None):
        super().__init__(delta)

    def approximation(
        self,
        lT: LineageTree,
        root: int,
        end_time: int | None = None,
        properties: dict[int, float | list] | list[str] = None,
    ):

        if isinstance(properties, list):
            prop_dicts = [getattr(lT, prop) for prop in properties]
            properties = {
                node: [prop.get(node, np.nan) for prop in prop_dicts]
                for node in lT.nodes
            }
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
        return ApproximatedTree(
            out_dict, properties, TreeSpecs(hash(lT), root, end_time)
        )
