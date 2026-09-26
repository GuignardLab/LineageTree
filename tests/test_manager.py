"""Adding trees to a manager, and comparing datasets of different time
resolutions."""

import pytest

from lineagetree import LineageTree, LineageTreeManager


def _division(chain_length, time_resolution):
    """A chain of `chain_length` nodes that divides into two such chains."""
    lT = LineageTree()
    root = lT.add_root(0)
    end = lT.add_chain(root, chain_length - 1, downstream=True)
    lT.add_chain(end, chain_length, downstream=True)
    lT.add_chain(end, chain_length, downstream=True)
    lT.time_resolution = time_resolution
    return lT, root


def test_identical_trees_are_both_added():
    """Equal trees under different names used to be skipped silently."""
    lT1 = LineageTree(successor={0: [1, 2]}, name="embryo1")
    lT2 = LineageTree(successor={0: [1, 2]}, name="embryo2")
    lTm = LineageTreeManager([lT1, lT2])
    assert len(lTm) == 2
    assert lTm["embryo2"] is lT2


def test_same_tree_is_added_once():
    lT = LineageTree(successor={0: [1, 2]}, name="embryo")
    lTm = LineageTreeManager([lT])
    assert lTm.add(lT, "again") is False
    assert len(lTm) == 1


def test_add_operator_returns_the_manager():
    """`lTm + lT` used to return None."""
    lTm = LineageTreeManager()
    lT1 = LineageTree(successor={0: [1]}, name="a")
    lT2 = LineageTree(successor={0: [1]}, name="b")
    assert (lTm + lT1) is lTm
    lTm += lT2
    assert isinstance(lTm, LineageTreeManager)
    assert len(lTm) == 2


def test_unnamed_trees_get_a_name():
    lTm = LineageTreeManager([LineageTree(successor={0: [1]})])
    assert list(lTm.lineagetrees) == ["Lineagetree 0"]


@pytest.fixture
def two_resolutions():
    """The same development, recorded every 5 and every 10 time units."""
    lT_fine, root_fine = _division(10, time_resolution=5)
    lT_coarse, root_coarse = _division(5, time_resolution=10)
    lTm = LineageTreeManager()
    lTm.add(lT_fine, "fine")
    lTm.add(lT_coarse, "coarse")
    return lTm, root_fine, root_coarse


@pytest.mark.parametrize("style", ["simple", "normalized_simple", "full"])
def test_same_development_at_two_resolutions(two_resolutions, style):
    lTm, root_fine, root_coarse = two_resolutions
    distance = lTm.cross_lineage_edit_distance(
        root_fine, "fine", root_coarse, "coarse", style=style
    )
    assert distance == 0


@pytest.mark.parametrize(
    ("downsample", "kept_nodes"), [(10, 15), (20, 7), (50, 3)]
)
def test_downsample_is_a_duration(two_resolutions, downsample, kept_nodes):
    """`downsample` used to have no effect across datasets.

    Each chain lasts 50 time units, so sampling every `downsample` units
    keeps `50 / downsample` nodes per chain (plus the last one), whatever
    the time resolution of the dataset.
    """
    lTm, root_fine, root_coarse = two_resolutions
    cost, norms = lTm.cross_lineage_edit_distance(
        root_fine,
        "fine",
        root_coarse,
        "coarse",
        style="downsampled",
        downsample=downsample,
        return_norms=True,
    )
    assert cost == 0
    assert norms == (kept_nodes, kept_nodes)


def test_downsample_must_match_the_resolutions(two_resolutions):
    lTm, root_fine, root_coarse = two_resolutions
    with pytest.raises(Exception, match="multiple of"):
        lTm.cross_lineage_edit_distance(
            root_fine,
            "fine",
            root_coarse,
            "coarse",
            style="downsampled",
            downsample=15,
        )


def test_downsample_with_a_fractional_time_resolution():
    """A time resolution below 1 used to give a time scale of 0."""
    lT_a, root_a = _division(8, time_resolution=0.5)
    lT_b, root_b = _division(4, time_resolution=1)
    lTm = LineageTreeManager()
    lTm.add(lT_a, "a")
    lTm.add(lT_b, "b")
    cost, norms = lTm.cross_lineage_edit_distance(
        root_a,
        "a",
        root_b,
        "b",
        style="downsampled",
        downsample=2,
        return_norms=True,
    )
    assert cost == 0
    assert norms == (6, 6)
