from __future__ import annotations

from collections.abc import Callable, Iterable
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..lineage_tree import LineageTree


def modifier(wrapped_func):
    @wraps(wrapped_func)
    def raising_flag(self, *args, **kwargs):
        should_reset = (
            not hasattr(self, "_has_been_reset") or not self._has_been_reset
        )
        out_func = wrapped_func(self, *args, **kwargs)
        if should_reset:
            for prop in self._protected_dynamic_properties:
                self.__dict__[prop] = None
            self._has_been_reset = True
        return out_func

    return raising_flag


###TODO pos can be callable and stay motionless (copy the position of the succ node, use something like optical flow)
@modifier
def add_chain(
    lT: LineageTree,
    node: int,
    length: int,
    downstream: bool,
    pos: Callable | None = None,
) -> int:
    """Adds a chain of specific length to a node either as a successor or as a predecessor.
    If it is placed on top of a tree all the nodes will move timepoints #length down.

    Parameters
    ----------
    node : int
        Id of the successor (predecessor if `downstream==False`)
    length : int
        The length of the new chain.
    downstream : bool, default=True
        If `True` will create a chain that goes forwards in time otherwise backwards.
    pos : np.ndarray, optional
        The new position of the chain. Defaults to None.

    Returns
    -------
    int
        Id of the first node of the sublineage.
    """
    if length == 0:
        return node
    if length < 1:
        raise ValueError("Length cannot be <1")
    if downstream:
        for _ in range(int(length)):
            old_node = node
            node = lT._add_node(pred=[old_node])
            lT._time[node] = lT._time[old_node] + 1
    else:
        if lT._predecessor[node]:
            raise Warning("The node already has a predecessor.")
        if lT._time[node] - length < lT.t_b:
            raise Warning(
                "A node cannot created outside the lower bound of the dataset. (It is possible to change it by lT.t_b = int(...))"
            )
        for _ in range(int(length)):
            old_node = node
            node = lT._add_node(succ=[old_node])
            lT._time[node] = lT._time[old_node] - 1
    return node


@modifier
def add_root(lT: LineageTree, t: int, pos: list | None = None) -> int:
    """Adds a root to a specific timepoint.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    t :int
        The timepoint the node is going to be added.
    pos : list
        The position of the new node.
    Returns
    -------
    int
        The id of the new root.
    """
    C_next = lT.get_next_id()
    lT._successor[C_next] = ()
    lT._predecessor[C_next] = ()
    lT._time[C_next] = t
    lT.pos[C_next] = pos if isinstance(pos, list) else []
    lT._changed_roots = True
    return C_next


def get_next_id(lT) -> int:
    """Computes the next authorized id and assign it.

    Returns
    -------
    int
        next authorized id
    """
    if not hasattr(lT, "max_id") or (lT.max_id == -1 and lT.nodes):
        lT.max_id = max(lT.nodes) if len(lT.nodes) else 0
    if not hasattr(lT, "next_id") or lT.next_id == []:
        lT.max_id += 1
        return lT.max_id
    else:
        return lT.next_id.pop()


@modifier
def _add_node(
    lT: LineageTree,
    succ: list | None = None,
    pred: list | None = None,
    pos: Iterable | None = None,
    nid: int | None = None,
) -> int:
    """Adds a node to the LineageTree object that is either a successor or a predecessor of another node.
    Does not handle time! You cannot enter both a successor and a predecessor.

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    succ : list
        list of ids of the nodes the new node is a successor to
    pred : list
        list of ids of the nodes the new node is a predecessor to
    pos : np.ndarray, optional
        position of the new node
    nid : int, optional
        id value of the new node, to be used carefully,
        if None is provided the new id is automatically computed.

    Returns
    -------
    int
        id of the new node.
    """
    if not succ and not pred:
        raise Warning(
            "Please enter a successor or a predecessor, otherwise use the add_roots() function."
        )
    C_next = lT.get_next_id() if nid is None else nid
    if succ:
        lT._successor[C_next] = succ
        for suc in succ:
            lT._predecessor[suc] = (C_next,)
    else:
        lT._successor[C_next] = ()
    if pred:
        lT._predecessor[C_next] = pred
        lT._successor[pred[0]] = lT._successor.setdefault(pred[0], ()) + (
            C_next,
        )
    else:
        lT._predecessor[C_next] = ()
    if isinstance(pos, list):
        lT.pos[C_next] = pos
    return C_next


@modifier
def remove_nodes(lT: LineageTree, group: int | set | list) -> None:
    """Removes a group of nodes from the LineageTree

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    group : set of int or list of int or int
        One or more nodes that are to be removed.
    """
    if isinstance(group, int | float):
        group = {group}
    if isinstance(group, list):
        group = set(group)
    group = lT.nodes.intersection(group)
    for node in group:
        for attr in lT.__dict__:
            attr_value = lT.__getattribute__(attr)
            if isinstance(attr_value, dict) and attr not in [
                "successor",
                "predecessor",
                "_successor",
                "_predecessor",
            ]:
                attr_value.pop(node, ())
        if lT._predecessor.get(node):
            lT._successor[lT._predecessor[node][0]] = tuple(
                set(lT._successor[lT._predecessor[node][0]]).difference(group)
            )
        for p_node in lT._successor.get(node, []):
            lT._predecessor[p_node] = ()
        lT._predecessor.pop(node, ())
        lT._successor.pop(node, ())
