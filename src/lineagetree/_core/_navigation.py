from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from ..lineage_tree import LineageTree


def get_predecessors(
    lT: LineageTree,
    x: int,
    depth: int | None = None,
    start_time: int | None = None,
    end_time: int | None = None,
) -> list[int]:
    """Return the predecessors of a node within its chain.

    Predecessors are collected backwards from ``x`` until the first node of
    its chain (the node right after the last division), a root, or ``depth``
    steps, whichever comes first.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    x : int
        Id of the node to start from.
    depth : int, optional
        Maximum number of predecessors to collect. If None, there is no
        limit.
    start_time : int, optional
        Earliest time point to include. Defaults to ``lT.t_b``.
    end_time : int, optional
        Latest time point to include. Defaults to ``lT.t_e``.

    Returns
    -------
    list of int
        Node ids in time order. The last one is ``x``, unless ``x`` is
        outside ``[start_time, end_time]``.
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
    """Return the successors of a node within its chain.

    Successors are collected forwards from ``x`` until the last node of its
    chain (a division or a leaf), ``end_time``, or ``depth`` steps, whichever
    comes first.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    x : int
        Id of the node to start from.
    depth : int, optional
        Maximum number of successors to collect. If None, there is no limit.
    end_time : int, optional
        Latest time point to include. Defaults to ``lT.t_e``.

    Returns
    -------
    list of int
        Node ids in time order; the first one is ``x``.
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
    """Return the chain a node belongs to.

    A chain is the life of one cell between two divisions: a run of nodes in
    which every node but the last has exactly one successor. The chain is
    the concatenation of
    [`get_predecessors`][lineagetree.LineageTree.get_predecessors] and
    [`get_successors`][lineagetree.LineageTree.get_successors] of ``x``.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    x : int
        Id of a node of the chain.
    depth : int, optional
        Maximum number of predecessors and of successors of ``x`` to
        include. Overrides ``depth_pred`` and ``depth_succ`` when given.
    depth_pred : int, optional
        Maximum number of predecessors of ``x`` to include.
    depth_succ : int, optional
        Maximum number of successors of ``x`` to include.
    end_time : int, optional
        Latest time point to include. Defaults to ``lT.t_e``.

    Returns
    -------
    list of int
        Node ids of the chain, in time order. With no depth limit, this is
        the whole chain.

    Examples
    --------
    >>> from lineagetree import LineageTree
    >>> lT = LineageTree(successor={0: [1], 1: [2], 2: [3, 4], 3: [], 4: []})
    >>> lT.get_chain_of_node(1)
    [0, 1, 2]
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
    """Return all the chains of the subtree spawned by a node.

    This is the subtree version of the ``all_chains`` property.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    node : int
        The node spawning the subtree. Its chain starts at ``node``, which
        need not be the first node of a chain.
    end_time : int, optional
        Latest time point to include. Defaults to ``lT.t_e``.

    Returns
    -------
    list of list of int
        The chains, each a list of node ids in time order. The chain of
        ``node`` comes first.
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
    """Find the leaves of the subtrees spawned by one or more nodes.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    roots : int or Iterable of int
        The node(s) spawning the subtrees.

    Returns
    -------
    set of int
        The leaves of the subtrees.
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
    """Return the nodes of the subtree spawned by one or more nodes.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    x : int or Iterable of int
        The node(s) spawning the subtree(s). They are included in the
        output.
    end_time : int, optional
        Latest time point to traverse. Defaults to ``lT.t_e``.
    preorder : bool, default=False
        Change the traversal order: nodes are then taken from the front of
        the list of nodes still to visit instead of its back.

    Returns
    -------
    list of int
        The node ids. With the default ``preorder=False``, the order is a
        depth-first pre-order: a node comes before its successors.
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
    """Find the ancestor of a node at a given time point.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    n : int
        The node whose ancestor is searched.
    time : int, optional
        Time point of the ancestor. Defaults to ``lT.t_b``, the first time
        point of the dataset.

    Returns
    -------
    int
        Id of the ancestor of ``n`` at ``time`` (``n`` itself if it is at
        ``time``), or ``-1`` if ``n`` has no ancestor at that time or is not
        a node of the tree.
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


def get_labelled_ancestor(lT: LineageTree, node: int) -> int:
    """Find the closest ancestor of a node that has a label.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    node : int
        The node to start from. It is returned if it has a label itself.

    Returns
    -------
    int
        Id of the closest labelled ancestor (see
        [`labels`][lineagetree.LineageTree.labels]), or ``-1`` if there is
        none.
    """
    if node not in lT.nodes:
        return -1
    ancestor = node
    while lT.t_b <= lT._time.get(ancestor, lT.t_b - 1) and ancestor != -1:
        if ancestor in lT.labels:
            return ancestor
        ancestor = lT._predecessor.get(ancestor, [-1])[0]
    return -1


def get_ancestor_with_attribute(
    lT: LineageTree, node: int, attribute: str
) -> int:
    """Find the closest ancestor of a node that has a value for a property.

    General version of
    [`get_labelled_ancestor`][lineagetree.LineageTree.get_labelled_ancestor]
    for any custom property.

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
    attr_dict = lT.__getattribute__(attribute)
    if not isinstance(attr_dict, dict):
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
    t: int | None,
    r: int | Iterable[int],
) -> list[int]:
    """Return the descendants of one or more nodes at a given time point.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    t : int or None
        The time point. If None, the last time point of the dataset
        (``lT.t_e``) is used.
    r : int or Iterable of int
        The node(s) whose descendants are returned.

    Returns
    -------
    list of int
        The descendants of ``r`` at time ``t``, including ``r`` itself if it
        is at time ``t``.
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


def get_available_labels(lT: LineageTree) -> list[str]:
    """List the properties that can be used as node labels.

    A property qualifies when it is a non-empty dictionary that maps node ids
    (``int``) to strings.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.

    Returns
    -------
    list of str
        The names of these properties, to pass to
        [`change_labels`][lineagetree.LineageTree.change_labels].
    """
    available_labels = []
    for prop_name, prop in lT.__dict__.items():
        if (
            0 < len(prop_name)
            and prop_name[0] != "_"
            and isinstance(prop, dict)
            and 0 < len(prop)
            and all(isinstance(k, int) for k in prop.keys())
            and all(isinstance(v, str) for v in prop.values())
        ):
            available_labels.append(prop_name)
    return available_labels


def change_labels(
    lT: LineageTree,
    new_labels_name: str | None = None,
    new_labels_dict: dict[int, str] | None = None,
    only_first_node_in_chain: bool = False,
) -> None:
    """Change the property used as node labels.

    The labels (see [`labels`][lineagetree.LineageTree.labels]) are taken
    from the property ``new_labels_name``, or from ``new_labels_dict``,
    which is then stored as a new property under that name.

    Labelling only the first node of each chain can help readability, for
    example in the napari plugin reLAX.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    new_labels_name : str, optional
        Name of the property to use as labels; see
        [`get_available_labels`][lineagetree.LineageTree.get_available_labels]
        for the candidates. If None, the labels are reset to the default
        ``"Unlabeled"`` labels and ``new_labels_dict`` is ignored.
    new_labels_dict : dict of {int: str}, optional
        Labels to use, mapping node ids to strings. If None, the existing
        property ``new_labels_name`` is used.
    only_first_node_in_chain : bool, default=False
        If True, only the first node of each chain is labelled.

    Raises
    ------
    AttributeError
        If ``new_labels_dict`` is None and the tree has no property named
        ``new_labels_name``.
    TypeError
        If a label is not a ``str``.

    Warns
    -----
    UserWarning
        If no node of the tree gets a label; the labels are then left
        unchanged.
    """
    store_new_labels = True
    if new_labels_name is not None:
        lT.labels_name = new_labels_name
        if new_labels_dict is None:
            if new_labels_name in lT.__dict__:
                new_labels_dict = lT.__dict__[new_labels_name]
                store_new_labels = False
            else:
                raise AttributeError(
                    f"{new_labels_name} is not in the properties of {lT.name}"
                )
        if any(not isinstance(v, str) for v in new_labels_dict.values()):
            raise TypeError(
                "All values of new_labels dictionary should be `str`"
            )

        labelled_cells = lT.nodes.intersection(new_labels_dict)
        if only_first_node_in_chain:
            labelled_cells = labelled_cells.intersection(
                {chain[0] for chain in lT.all_chains}
            )

        if len(labelled_cells) < 1:
            warnings.warn(
                "The labeling dictionary does not have any node labels.\n"
                'Defaulting to the "Unlabeled" labeling'
            )
        else:
            lT._labels = {n: new_labels_dict[n] for n in labelled_cells}
            if store_new_labels:
                lT.__dict__[new_labels_name] = lT._labels
    else:
        lT.labels_name = ""
        lT._labels = {
            root: "Unlabeled"
            for root in lT.roots
            for leaf in lT.find_leaves(root)
            if abs(lT._time[leaf] - lT._time[root]) >= abs(lT.t_e - lT.t_b) / 4
        }


def get_shortest_path_and_last_common_ancestor(
    lT: LineageTree, n1: int, n2: int
) -> tuple[list[int], int]:
    """Return the path between two nodes and their last common ancestor.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    n1 : int
        The first node.
    n2 : int
        The second node.

    Returns
    -------
    list of int
        The nodes on the path from ``n1`` to ``n2``, both included, going
        through their last common ancestor. Empty if the nodes are in
        different trees.
    int
        The last common ancestor of ``n1`` and ``n2``, or ``-1`` if the
        nodes are in different trees.
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
