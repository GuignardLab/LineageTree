class TreeValidator:
    """Handles cycle-detection and structural validation for a LineageTree."""

    def __init__(self, tree_instance):
        """
        Parameters
        ----------
        tree_instance : LineageTree
            The tree instance to validate.
        """
        self.tree = tree_instance

    def check_cc_cycles(self, n: int) -> tuple[bool, set[int]]:
        """Check whether the connected component containing node `n` has a cycle.

        Traverses both forward (via successors) and backward (via predecessors)
        from `n`, collecting visited nodes. A cycle is detected if any node is
        visited twice.

        Parameters
        ----------
        n : int
            Node id to start the search from.

        Returns
        -------
        bool
            ``True`` if a cycle was found, ``False`` otherwise.
        set of int
            The set of all nodes visited during the traversal (the connected
            component of `n`).
        """
        to_do = [n]
        no_cycle = True
        already_done = set()
        while to_do and no_cycle:
            current = to_do.pop(-1)
            if current not in already_done:
                already_done.add(current)
            else:
                no_cycle = False
            to_do.extend(self.tree._successor[current])
        to_do = list(self.tree._predecessor[n])
        while to_do and no_cycle:
            current = to_do.pop(-1)
            if current not in already_done:
                already_done.add(current)
            else:
                no_cycle = False
            to_do.extend(self.tree._predecessor[current])
        return not no_cycle, already_done

    def check_for_cycles(self) -> bool:
        """Check whether the tree contains any cycles.

        Iterates over all nodes and calls :meth:`check_cc_cycles` on each
        unvisited connected component.

        Returns
        -------
        bool
            ``True`` if at least one cycle was found, ``False`` otherwise.
        """
        to_do = set(self.tree.nodes)
        found_cycle = False
        while to_do and not found_cycle:
            current = to_do.pop()
            found_cycle, done = self.check_cc_cycles(current)
            to_do.difference_update(done)
        return found_cycle
