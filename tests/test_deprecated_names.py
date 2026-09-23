"""The 3.2 method names keep working, with a deprecation warning."""

import warnings

import pytest

from lineagetree import LineageTree, read_from_mamut_xml


@pytest.fixture(scope="module")
def lT():
    return read_from_mamut_xml("tests/data/test-mamut.xml")


def test_time_alone_does_not_warn():
    """Providing `time` without `starting_time` is not worth a warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        LineageTree(successor={0: [1]}, time={0: 0, 1: 1})


@pytest.mark.parametrize(
    ("old", "new", "kwargs"),
    [
        ("get_idx3d", "idx3d", {"t": 10}),
        ("get_gabriel_graph", "gabriel_graph", {"time": 10}),
        ("compute_k_nearest_neighbours", "k_nearest_neighbours", {"k": 3}),
        ("compute_spatial_edges", "spatial_edges", {"th": 30}),
        (
            "compute_spatial_density",
            "spatial_density",
            {"t_b": 10, "t_e": 12, "th": 50},
        ),
    ],
)
def test_32_names_warn_and_forward(lT, old, new, kwargs):
    with pytest.warns(DeprecationWarning, match=new):
        old_result = getattr(lT, old)(**kwargs)
    new_result = getattr(lT, new)(**kwargs)

    assert type(old_result) is type(new_result)
    if old == "get_idx3d":
        assert (old_result[1] == new_result[1]).all()
    elif old == "compute_k_nearest_neighbours":
        assert old_result[0].keys() == new_result[0].keys()
    else:
        assert old_result == new_result


def test_calculate_dtw_warns_and_forwards(lT):
    roots = sorted(lT.roots, key=lambda r: -len(lT.get_subtree_nodes(r)))[:2]
    with pytest.warns(DeprecationWarning, match="dtw"):
        old_result = lT.calculate_dtw(*roots)
    assert old_result[0] == lT.dtw(*roots)[0]


def test_compute_neighbours_in_radius_keeps_counting(lT):
    """The 3.2 name counted neighbours; the new one returns them."""
    with pytest.warns(DeprecationWarning, match="neighbours_in_radius"):
        counts = lT.compute_neighbours_in_radius(t_b=0, t_e=20, th=200)
    neighbours = lT.neighbours_in_radius(t_b=0, t_e=20, th=200)

    assert counts == {n: len(v) for n, v in neighbours.items()}
