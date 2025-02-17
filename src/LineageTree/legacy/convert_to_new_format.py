from ..lineageTree import lineageTree


class tmp_lt(lineageTree):

    @property
    def successor(self):
        self._successor = self.successor
        return self._successor

    @property
    def predecessor(self):
        self._predecessor = self.predecessor
        return self.predecessor
