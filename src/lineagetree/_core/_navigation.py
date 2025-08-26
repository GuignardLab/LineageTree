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
    """Computes the predecessors of the node `x` up to
    `depth` predecessors or the begining of the life of `x`.
    The ordered list of ids is returned.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    x : int
        id of the node to compute
    depth : int
        maximum number of predecessors to return
    start_time : int, optional
        maximum time to consider, if not provided the beginning of the life of `x` is used
    end_time : int, optional
        maximum time to consider, if not provided the end of the life of `x` is used

    Returns
    -------
    list of int
        list of ids, the last id is `x`
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
    """Computes the successors of the node `x` up to
    `depth` successors or the end of the life of `x`.
    The ordered list of ids is returned.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    x : int
        id of the node to compute
    depth : int, optional
        maximum number of predecessors to return
    end_time : int, optional
        maximum time to consider

    Returns
    -------
    list of int
        list of ids, the first id is `x`
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
    """Computes the predecessors and successors of the node `x` up to
    `depth_pred` predecessors plus `depth_succ` successors.
    If the value `depth` is provided and not None,
    `depth_pred` and `depth_succ` are overwriten by `depth`.
    The ordered list of ids is returned.
    If all `depth` are None, the full chain is returned.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    x : int
        id of the node to compute
    depth : int, optional
        maximum number of predecessors and successor to return
    depth_pred : int, optional
        maximum number of predecessors to return
    depth_succ : int, optional
        maximum number of successors to return

    Returns
    -------
    list of int
        list of node ids
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
    """Computes all the chains of the subtree spawn by a given node.
    Similar to get_all_chains().

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
    """Computes the list of nodes from the subtree spawned by *x*
    The default output order is Breadth First Traversal.
    Unless preorder is `True` in that case the order is
    Depth First Traversal (DFT) preordered.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    x : int
        id of root node
    preorder : bool, default=False
        if True the output preorder is Depth First Traversal (DFT)
        Otherwise it is Breadth First Traversal (BFT)

    Returns
    -------
    list of int
        the ordered list of node ids
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
    """Find the id of the ancestor of a give node `n`
    at a given time `time`.

    If there is no ancestor, returns `None`
    If time is None return the root of the subtree that spawns
    the node n.

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


def get_labelled_ancestor(lT: LineageTree, node: int) -> int:
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
    while lT.t_b <= lT._time.get(ancestor, lT.t_b - 1) and ancestor != -1:
        if ancestor in lT.labels:
            return ancestor
        ancestor = lT._predecessor.get(ancestor, [-1])[0]
    return -1


def get_ancestor_with_attribute(
    lT: LineageTree, node: int, attribute: str
) -> int:
    """General purpose function to help with searching the first ancestor that has an attribute.
    Similar to get_labeled_ancestor and may make it redundant.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    node : int
        The id of the node

    Returns
    -------
    int
        Returns the first ancestor found that has an attribute otherwise `-1`.
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
    t: int,
    r: int | Iterable[int],
) -> list[int]:
    """
    Returns the list of nodes at time `t` that are spawn by the node(s) `r`.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    t : int
        target time, if `None` goes as far as possible
    r : int or Iterable of int
        id or list of ids of the spawning node

    Returns
    -------
    list of int
        list of ids of the nodes at time `t` spawned by `r`
    """
    if isinstance(r, int):
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
    """Returns the list all the available label dictionaries

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.

    Returns
    -------
    list of string
        list of the names of all the available properties
        to label the nodes
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
    """Change the dictionary that serves at labels with
    the `LineageTree` attribute `new_labels_name`.
    It has to be a dictionary mapping node id to string.

    If `new_labels_dict` is provided, it will be used to
    label the cells.
    If `new_labels_name` is not specified, the labels are reset.

    One can decide to only label the first node of the chain
    instead of all the nodes of the chain.
    That can help readability in the napari plugin reLAX.

    Parameters
    ----------
    lT : LineageTree
    new_labels_name : string, optional
        The name of the dictionary to use
        (the list of potential dictionaries can be found
        with `lT.available_labels`)
        If `new_labels_name` is not provided,
        the labels are reset to `"Unlabeleld"`
    new_labels_dict : dictionary mapping integers to strings, optional
        The new names as a dictionary mapping each named node id to its string label
        If not provided and lT has a fitting attribute named `new_labels_name`,
        it will therefore be used
    only_first_node_in_chain : bool, default=True
        If `True` only labels the first node of the chains
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
