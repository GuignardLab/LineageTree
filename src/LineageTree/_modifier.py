from collections.abc import Callable, Iterable
from functools import wraps


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
    self,
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
            node = self._add_node(pred=[old_node])
            self._time[node] = self._time[old_node] + 1
    else:
        if self._predecessor[node]:
            raise Warning("The node already has a predecessor.")
        if self._time[node] - length < self.t_b:
            raise Warning(
                "A node cannot created outside the lower bound of the dataset. (It is possible to change it by lT.t_b = int(...))"
            )
        for _ in range(int(length)):
            old_node = node
            node = self._add_node(succ=[old_node])
            self._time[node] = self._time[old_node] - 1
    return node


@modifier
def add_root(self, t: int, pos: list | None = None) -> int:
    """Adds a root to a specific timepoint.

    Parameters
    ----------
    t :int
        The timepoint the node is going to be added.
    pos : list
        The position of the new node.
    Returns
    -------
    int
        The id of the new root.
    """
    C_next = self.get_next_id()
    self._successor[C_next] = ()
    self._predecessor[C_next] = ()
    self._time[C_next] = t
    self.pos[C_next] = pos if isinstance(pos, list) else []
    self._changed_roots = True
    return C_next


def get_next_id(self) -> int:
    """Computes the next authorized id and assign it.

    Returns
    -------
    int
        next authorized id
    """
    if not hasattr(self, "max_id") or (self.max_id == -1 and self.nodes):
        self.max_id = max(self.nodes) if len(self.nodes) else 0
    if not hasattr(self, "next_id") or self.next_id == []:
        self.max_id += 1
        return self.max_id
    else:
        return self.next_id.pop()


@modifier
def _add_node(
    self,
    succ: list | None = None,
    pred: list | None = None,
    pos: Iterable | None = None,
    nid: int | None = None,
) -> int:
    """Adds a node to the LineageTree object that is either a successor or a predecessor of another node.
    Does not handle time! You cannot enter both a successor and a predecessor.

    Parameters
    ----------
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
    C_next = self.get_next_id() if nid is None else nid
    if succ:
        self._successor[C_next] = succ
        for suc in succ:
            self._predecessor[suc] = (C_next,)
    else:
        self._successor[C_next] = ()
    if pred:
        self._predecessor[C_next] = pred
        self._successor[pred[0]] = self._successor.setdefault(pred[0], ()) + (
            C_next,
        )
    else:
        self._predecessor[C_next] = ()
    if isinstance(pos, list):
        self.pos[C_next] = pos
    return C_next


@modifier
def remove_nodes(self, group: int | set | list) -> None:
    """Removes a group of nodes from the LineageTree

    Parameters
    ----------
    group : set of int or list of int or int
        One or more nodes that are to be removed.
    """
    if isinstance(group, int | float):
        group = {group}
    if isinstance(group, list):
        group = set(group)
    group = self.nodes.intersection(group)
    for node in group:
        for attr in self.__dict__:
            attr_value = self.__getattribute__(attr)
            if isinstance(attr_value, dict) and attr not in [
                "successor",
                "predecessor",
                "_successor",
                "_predecessor",
            ]:
                attr_value.pop(node, ())
        if self._predecessor.get(node):
            self._successor[self._predecessor[node][0]] = tuple(
                set(self._successor[self._predecessor[node][0]]).difference(
                    group
                )
            )
        for p_node in self._successor.get(node, []):
            self._predecessor[p_node] = ()
        self._predecessor.pop(node, ())
        self._successor.pop(node, ())
