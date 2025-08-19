class TreeValidator:
    """Handles tree validation logic."""

    def __init__(self, tree_instance):
        self.tree = tree_instance

    def check_cc_cycles(self, n: int) -> tuple[bool, set[int]]:
        """Check if the connected component of a given node `n` has a cycle."""
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
        """Check if the tree has cycles."""
        to_do = set(self.tree.nodes)
        found_cycle = False
        while to_do and not found_cycle:
            current = to_do.pop()
            found_cycle, done = self.check_cc_cycles(current)
            to_do.difference_update(done)
        return found_cycle
