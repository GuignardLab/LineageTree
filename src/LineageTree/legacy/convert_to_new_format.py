import argparse
from pathlib import Path
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


def open_old_lT(path):
    path = Path(path)
    lT = tmp_lt.load(path)
    # lT.write("test/data/test-mamut(1).lT")
