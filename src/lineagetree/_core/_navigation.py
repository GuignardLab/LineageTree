from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING
from ..util_types import StaticTypedValueDict

if TYPE_CHECKING:
    from ..lineage_tree import LineageTree


def get_predecessors(
    lT: LineageTree,
    x: int,
    depth: int | None = None,
    start_time: int | None = None,
    end_time: int | None = None,
) -> list[int]:
    """Compute the predecessors of a node up to a given depth.

    The predecessors of the node ``x`` are collected up to ``depth``
    predecessors or the beginning of the life of ``x``, and returned as an
    ordered list of ids.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    x : int
        Id of the node to compute.
    depth : int, optional
        Maximum number of predecessors to return.
    start_time : int, optional
        Earliest time point to include in the returned chain.
        Defaults to ``lT.t_b`` (beginning of the dataset).
    end_time : int, optional
        Latest time point to include in the returned chain.
        Defaults to ``lT.t_e`` (end of the dataset).

    Returns
    -------
    list of int
        List of ids; the last id is ``x``.
    """
    if start_time is None:
        start_time = lT.t_b
    if end_time is None:
        end_time = lT.t_e
    unconstrained_chain = [x]
    chain = [x] if start_time <= lT._time[x] <= end_time else []
    acc = 0
    while (
        acc != depth
        and start_time < lT._time[unconstrained_chain[0]]
        and (
            lT._predecessor[unconstrained_chain[0]] != ()
            and (
                len(lT._successor[lT._predecessor[unconstrained_chain[0]][0]])
                == 1
            )
        )
    ):
        unconstrained_chain.insert(
            0, lT._predecessor[unconstrained_chain[0]][0]
        )
        acc += 1
        if start_time <= lT._time[unconstrained_chain[0]] <= end_time:
            chain.insert(0, unconstrained_chain[0])

    return chain


def get_successors(
    lT: LineageTree,
    x: int,
    depth: int | None = None,
    end_time: int | None = None,
) -> list[int]:
    """Compute the successors of a node up to a given depth.

    The successors of the node ``x`` are collected up to ``depth`` successors
    or the end of the life of ``x``, and returned as an ordered list of ids.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    x : int
        Id of the node to compute.
    depth : int, optional
        Maximum number of successors to return.
    end_time : int, optional
        Maximum time to consider.

    Returns
    -------
    list of int
        List of ids; the first id is ``x``.
    """
    if end_time is None:
        end_time = lT.t_e
    chain = [x]
    acc = 0
    while (
        len(lT._successor[chain[-1]]) == 1
        and acc != depth
        and lT._time[chain[-1]] < end_time
    ):
        chain += lT._successor[chain[-1]]
        acc += 1

    return chain


def get_chain_of_node(
    lT: LineageTree,
    x: int,
    depth: int | None = None,
    depth_pred: int | None = None,
    depth_succ: int | None = None,
    end_time: int | None = None,
) -> list[int]:
    """Compute the chain of a node from its predecessors and successors.

    The chain gathers up to ``depth_pred`` predecessors plus ``depth_succ``
    successors of the node ``x``, returned as an ordered list of ids. If
    ``depth`` is provided and not None, it overwrites both ``depth_pred`` and
    ``depth_succ``. If all depths are None, the full chain is returned.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    x : int
        Id of the node to compute.
    depth : int, optional
        Maximum number of predecessors and successors to return.
    depth_pred : int, optional
        Maximum number of predecessors to return.
    depth_succ : int, optional
        Maximum number of successors to return.
    end_time : int, optional
        Maximum time to consider.

    Returns
    -------
    list of int
        List of node ids.
    """
    if end_time is None:
        end_time = lT.t_e
    if depth is not None:
        depth_pred = depth_succ = depth
    return lT.get_predecessors(x, depth_pred, end_time=end_time)[
        :-1
    ] + lT.get_successors(x, depth_succ, end_time=end_time)


def get_all_chains_of_subtree(
    lT: LineageTree, node: int, end_time: int | None = None
) -> list[list[int]]:
    """Compute all the chains of the subtree spawned by a given node.

    Similar to :func:`get_all_chains`.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    node : int
        The node from which we want to get its chains.
    end_time : int, optional
        The time at which we want to stop the chains.

    Returns
    -------
    list of list of int
        list of chains
    """
    if not end_time:
        end_time = lT.t_e
    chains = [lT.get_successors(node)]
    to_do = list(lT._successor[chains[0][-1]])
    while to_do:
        current = to_do.pop()
        chain = lT.get_successors(current, end_time=end_time)
        if lT._time[chain[-1]] <= end_time:
            chains += [chain]
            to_do += lT._successor[chain[-1]]
    return chains


def find_leaves(lT: LineageTree, roots: int | Iterable) -> set[int]:
    """Finds the leaves of a tree spawned by one or more nodes.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    roots : int or Iterable
        The roots of the trees spawning the leaves

    Returns
    -------
    set
        The leaves of one or more trees.
    """
    if not isinstance(roots, Iterable):
        to_do = [roots]
    elif isinstance(roots, Iterable):
        to_do = list(roots)
    leaves = set()
    while to_do:
        curr = to_do.pop()
        succ = lT._successor[curr]
        if not succ:
            leaves.add(curr)
        to_do += succ
    return leaves


def get_subtree_nodes(
    lT: LineageTree,
    x: int | Iterable,
    end_time: int | None = None,
    preorder: bool = False,
) -> list[int]:
    """Compute the list of nodes of the subtree spawned by a node.

    The default output order is Breadth First Traversal, unless ``preorder`` is
    True, in which case the order is Depth First Traversal (DFT) preordered.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    x : int or Iterable
        Id of the root node (or an iterable of root ids).
    end_time : int, optional
        The time at which to stop the traversal.
    preorder : bool, default=False
        If True the output order is Depth First Traversal (DFT), otherwise it
        is Breadth First Traversal (BFT).

    Returns
    -------
    list of int
        The ordered list of node ids.
    """
    if not end_time:
        end_time = lT.t_e
    if not isinstance(x, Iterable):
        to_do = [x]
    elif isinstance(x, Iterable):
        to_do = list(x)
    subtree = []
    while to_do:
        curr = to_do.pop()
        succ = lT._successor[curr]
        if succ and end_time < lT._time.get(curr, end_time):
            succ = []
            continue
        if preorder:
            to_do = succ + to_do
        else:
            to_do += succ
        subtree += [curr]
    return subtree


def get_ancestor_at_t(lT: LineageTree, n: int, time: int | None = None) -> int:
    """Find the id of the ancestor of a given node at a given time.

    If there is no ancestor, ``-1`` is returned. If ``time`` is None, the root
    of the subtree that spawns the node ``n`` is returned.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    n : int
        node for which to look the ancestor
    time : int, optional
        time at which the ancestor has to be found.
        If `None` the ancestor at the first time point
        will be found.

    Returns
    -------
    int
        the id of the ancestor at time `time`,
        `-1` if there is no ancestor.
    """
    if n not in lT.nodes:
        return -1
    if time is None:
        time = lT.t_b
    ancestor = n
    while (
        time < lT._time.get(ancestor, lT.t_b - 1) and lT._predecessor[ancestor]
    ):
        ancestor = lT._predecessor[ancestor][0]
    if lT._time.get(ancestor, lT.t_b - 1) == time:
        return ancestor
    else:
        return -1


def get_labelled_ancestor(
    lT: LineageTree, node: int, labels: str | None = None
) -> int:
    """Finds the first labelled ancestor and returns its ID otherwise returns -1

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    node : int
        The id of the node

    Returns
    -------
    int
        Returns the first ancestor found that has a label otherwise `-1`.
    """
    if node not in lT.nodes:
        return -1
    ancestor = node
    if labels is not None and labels not in lT.list_all_labels():
        raise ValueError("Label set not defined.")
    while lT.t_b <= lT._time.get(ancestor, lT.t_b - 1) and ancestor != -1:
        if ancestor in lT.properties.label:
            return ancestor
        ancestor = lT._predecessor.get(ancestor, [-1])[0]
    return -1


def get_ancestor_with_attribute(
    lT: LineageTree, node: int, attribute: str
) -> int:
    """Find the first ancestor (inclusive of ``node``) that appears in a given attribute dict.

    General purpose function to help with searching the first ancestor that
    has an attribute. Similar to :func:`get_labelled_ancestor` and may make
    it redundant.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    node : int
        The id of the node to start the search from (inclusive).
    attribute : str
        Name of the ``LineageTree`` attribute to search in. Must be a ``dict``
        whose keys are node ids.

    Returns
    -------
    int
        Id of the first ancestor (including ``node`` itself) found in
        ``lT.<attribute>``, or ``-1`` if none is found.

    Raises
    ------
    ValueError
        If ``lT.<attribute>`` is not a dictionary.
    """
    attr_dict = getattr(lT.properties, attribute)
    if not isinstance(attr_dict, StaticTypedValueDict):
        raise ValueError("Please select a dict attribute")
    if node not in lT.nodes:
        return -1
    if node in attr_dict:
        return node
    if node in lT.roots:
        return -1
    ancestor = (node,)
    while ancestor and ancestor != [-1]:
        ancestor = ancestor[0]
        if ancestor in attr_dict:
            return ancestor
        ancestor = lT._predecessor.get(ancestor, [-1])
    return -1


def nodes_at_t(
    lT: LineageTree,
    t: int,
    r: int | Iterable[int],
) -> list[int]:
    """Return the nodes at time ``t`` that are spawned by the node(s) ``r``.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    t : int
        Target time. If None, goes as far as possible.
    r : int or Iterable of int
        Id or list of ids of the spawning node(s).

    Returns
    -------
    list of int
        List of ids of the nodes at time ``t`` spawned by ``r``.
    """
    if isinstance(r, Iterable):
        r = list(r)
    else:
        r = [r]
    if t is None:
        t = lT.t_e
    to_do = list(r)
    final_nodes = []
    while 0 < len(to_do):
        curr = to_do.pop()
        if lT._time[curr] == t:
            final_nodes.append(curr)
        elif lT._time[curr] < t:
            to_do.extend(lT.successor[curr])
    return final_nodes


def get_shortest_path_and_last_common_ancestor(
    lT: LineageTree, n1: int, n2: int
) -> tuple[list[int], int]:
    """Returns the minimum path between 2 nodes and the last commmon ancestor

    Parameters
    ----------
    lT : LineageTree
        The LineageTree object
    n1 : int
        The first node
    n2 : int
        The second node.

    Returns
    -------
    tuple which contains
        - list of int
            The shortest path from n1 to n2, containing n1, n2 and ordered, if not shortest path returns [].
        - int
            The last common ancestor of the 2 nodes, if none returns -1.
    """
    left_side = [n1]
    right_side = [n2]
    d1 = lT.depth[n1]
    d2 = lT.depth[n2]
    while d2 > d1:
        n2 = lT.predecessor[n2][0]
        d2 -= 1
        right_side.append(n2)
    while d1 > d2:
        n1 = lT.predecessor[n1][0]
        d1 -= 1
        left_side.append(n1)
    while n1 != n2:
        if d1 == 0:
            return [], -1
        n1 = lT.predecessor[n1][0]
        n2 = lT.predecessor[n2][0]
        d1 -= 1
        d2 -= 1
        left_side.append(n1)
        right_side.append(n2)
    return left_side + right_side[-2::-1], n1
