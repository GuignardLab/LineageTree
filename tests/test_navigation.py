"""Traversal of subtrees and chains, and restriction to a subtree."""

import warnings

import numpy as np
import pytest

from lineagetree import LineageTree


@pytest.fixture
def lT():
    """0 -> 1 -> 2, which divides into 3 and 4; 3 divides into 5 and 6.

    Times are the depths: 0, 1, 2, 3 (nodes 3, 4) and 4 (nodes 5, 6, 7).
    """
    successor = {
        0: [1],
        1: [2],
        2: [3, 4],
        3: [5, 6],
        4: [7],
        5: [],
        6: [],
        7: [],
    }
    pos = {n: [n, 0.0, 0.0] for n in successor}
    return LineageTree(successor=successor, pos=pos, volume={0: 1.0})


def _is_preorder(lT, order):
    """Each node comes after its predecessor and each subtree is contiguous."""
    index = {n: i for i, n in enumerate(order)}
    for n in order:
        subtree = [index[d] for d in lT.get_subtree_nodes(n)]
        if sorted(subtree) != list(range(index[n], index[n] + len(subtree))):
            return False
    return True


def test_subtree_nodes_default_order(lT):
    order = lT.get_subtree_nodes(0)
    assert sorted(order) == sorted(lT.nodes)
    assert order[0] == 0
    assert _is_preorder(lT, order)


def test_subtree_nodes_preorder_follows_successor_order(lT):
    """`preorder=True` used to raise `TypeError`."""
    assert lT.get_subtree_nodes(0, preorder=True) == [0, 1, 2, 3, 5, 6, 4, 7]


def test_subtree_nodes_preorder_with_several_roots(lT):
    assert lT.get_subtree_nodes([3, 4], preorder=True) == [3, 5, 6, 4, 7]


def test_subtree_nodes_stop_at_end_time(lT):
    assert lT.get_subtree_nodes(0, end_time=2) == [0, 1, 2]
    assert set(lT.get_subtree_nodes(0, end_time=3)) == {0, 1, 2, 3, 4}


def test_subtree_nodes_leave_out_leaves_after_end_time():
    """A leaf right after `end_time` used to be kept."""
    lT = LineageTree(successor={0: [1, 2], 1: [], 2: [3], 3: []})
    assert set(lT.get_subtree_nodes(0, end_time=0)) == {0}


def test_subtree_nodes_end_time_zero_is_a_limit():
    """`end_time=0` used to be read as "no limit"."""
    lT = LineageTree(successor={0: [1], 1: []}, starting_time=0)
    assert lT.get_subtree_nodes(0, end_time=0) == [0]


def test_all_chains_of_subtree(lT):
    chains = lT.get_all_chains_of_subtree(0)
    assert chains[0] == [0, 1, 2]
    assert sorted(map(tuple, chains)) == sorted(lT.all_chains)


def test_all_chains_of_subtree_cut_the_first_chain(lT):
    """The first chain used to ignore `end_time`."""
    assert lT.get_all_chains_of_subtree(0, end_time=1) == [[0, 1]]
    assert lT.get_all_chains_of_subtree(0, end_time=0) == [[0]]


def test_all_chains_of_subtree_cut_later_chains(lT):
    chains = lT.get_all_chains_of_subtree(0, end_time=3)
    assert sorted(map(tuple, chains)) == [(0, 1, 2), (3,), (4,)]


def test_all_chains_of_subtree_after_end_time(lT):
    assert lT.get_all_chains_of_subtree(3, end_time=2) == []


def test_get_subtree_does_not_warn(lT):
    """`get_subtree` used to pass the whole `time` dict and warn."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sub = lT.get_subtree(lT.get_subtree_nodes(3))
    assert sub.nodes == {3, 5, 6}
    assert dict(sub.time) == {3: 3, 5: 4, 6: 4}
    assert set(sub.pos) == {3, 5, 6}
    np.testing.assert_array_equal(sub.pos[5], lT.pos[5])


def test_get_subtree_ignores_unknown_ids(lT):
    sub = lT.get_subtree({3, 5, 6, 1000})
    assert sub.nodes == {3, 5, 6}


def test_get_subtree_keeps_custom_properties(lT):
    assert lT.get_subtree({0, 1}).volume == {0: 1.0}
