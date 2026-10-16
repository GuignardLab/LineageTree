"""`pos` is `raw_pos` multiplied by `spatial_resolution`, at all times."""

import pickle

import numpy as np
import pytest

from lineagetree import LineageTree, read_from_mamut_xml, read_from_swc

RESOLUTION = (0.5, 2.0, 3.0)


@pytest.fixture
def lT():
    """Two cells at time 0, one dividing into two at time 1."""
    return LineageTree(
        successor={0: [2, 3], 1: [], 2: [], 3: []},
        pos={0: [0, 0, 0], 1: [4, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]},
        spatial_resolution=RESOLUTION,
    )


def _assert_consistent(lT):
    assert lT.pos.keys() == lT.raw_pos.keys()
    for node, position in lT.pos.items():
        np.testing.assert_allclose(
            position, lT.raw_pos[node] * lT.spatial_resolution
        )


def test_init_scales_positions(lT):
    np.testing.assert_array_equal(lT.raw_pos[1], [4, 0, 0])
    np.testing.assert_array_equal(lT.pos[1], [2, 0, 0])
    np.testing.assert_array_equal(lT.spatial_resolution, RESOLUTION)
    _assert_consistent(lT)


def test_default_resolution_keeps_positions():
    lT = LineageTree(successor={0: []}, pos={0: [1, 2, 3]})
    np.testing.assert_array_equal(lT.spatial_resolution, [1, 1, 1])
    np.testing.assert_array_equal(lT.pos[0], lT.raw_pos[0])


def test_two_dimensional_positions():
    lT = LineageTree(
        successor={0: []}, pos={0: [1, 2]}, spatial_resolution=(2, 3)
    )
    np.testing.assert_array_equal(lT.pos[0], [2, 6])


def test_tree_without_positions():
    lT = LineageTree(successor={0: [1]}, spatial_resolution=RESOLUTION)
    assert lT.pos == lT.raw_pos == {}


def test_setting_the_resolution_rescales_positions(lT):
    lT.spatial_resolution = (1, 1, 10)
    np.testing.assert_array_equal(lT.pos[3], [0, 0, 10])
    np.testing.assert_array_equal(lT.raw_pos[3], [0, 0, 1])
    _assert_consistent(lT)


def test_setting_the_resolution_clears_spatial_caches(lT):
    lT.idx3d(0)
    lT.gabriel_graph()
    lT.k_nearest_neighbours(k=1)
    lT.spatial_edges(th=1)
    lT.spatial_resolution = (1, 1, 1)
    for cache in ("kdtrees", "Gabriel_graph", "kn_graph", "kn_distances"):
        assert cache not in lT.__dict__, cache
    assert "th_edges" not in lT.__dict__
    _, distances = lT.k_nearest_neighbours(k=1)
    assert distances[0][0] == pytest.approx(4)


@pytest.mark.parametrize("resolution", [(1, 1), (1, 1, 1, 1)])
def test_resolution_must_match_the_dimension(lT, resolution):
    with pytest.raises(ValueError, match="same dimension"):
        lT.spatial_resolution = resolution
    np.testing.assert_array_equal(lT.spatial_resolution, RESOLUTION)


@pytest.mark.parametrize("resolution", [(1, 0, 1), (1, -1, 1)])
def test_resolution_must_be_positive(lT, resolution):
    with pytest.raises(ValueError, match="positive"):
        lT.spatial_resolution = resolution


def test_spatial_methods_use_physical_units(lT):
    """At time 0, nodes 0 and 1 are 4 voxels apart, i.e. 2 units in x."""
    _, distances = lT.k_nearest_neighbours(k=1)
    assert distances[0][0] == pytest.approx(2)
    assert lT.neighbours_in_radius(0, 1, th=2.5)[0] == {1}
    assert lT.neighbours_in_radius(0, 1, th=1.5)[0] == set()


def test_added_nodes_are_placed_in_physical_units(lT):
    node = lT.add_root(0, pos=[1.0, 2.0, 3.0])
    np.testing.assert_array_equal(lT.pos[node], [1, 2, 3])
    np.testing.assert_allclose(lT.raw_pos[node], [2, 1, 1])
    _assert_consistent(lT)


def test_removed_nodes_leave_raw_pos(lT):
    lT.remove_nodes(3)
    assert 3 not in lT.raw_pos
    _assert_consistent(lT)


def test_get_subtree_keeps_resolution(lT):
    sub = lT.get_subtree({0, 2, 3})
    np.testing.assert_array_equal(sub.spatial_resolution, RESOLUTION)
    np.testing.assert_array_equal(sub.raw_pos[2], lT.raw_pos[2])
    np.testing.assert_array_equal(sub.pos[2], lT.pos[2])


def test_write_and_load_keep_raw_pos(lT, tmp_path):
    path = tmp_path / "tree.lT"
    lT.write(str(path))
    loaded = LineageTree.load(str(path))
    np.testing.assert_array_equal(loaded.spatial_resolution, RESOLUTION)
    for node in lT.nodes:
        np.testing.assert_array_equal(loaded.raw_pos[node], lT.raw_pos[node])
        np.testing.assert_array_equal(loaded.pos[node], lT.pos[node])


def test_load_file_written_before_raw_pos(tmp_path):
    """Older files stored the unscaled positions in `pos`."""
    lT = LineageTree(
        successor={0: [1], 1: []}, pos={0: [1, 1, 1], 1: [2, 2, 2]}
    )
    state = lT.__dict__
    state["pos"] = {0: [1, 1, 1], 1: [2, 2, 2]}
    state["spatial_resolution"] = np.array([1.0, 1.0, 4.0])
    del state["raw_pos"], state["_spatial_resolution"]
    path = tmp_path / "old.lT"
    with open(path, "wb") as f:
        pickle.dump(lT, f)

    loaded = LineageTree.load(str(path))
    np.testing.assert_array_equal(loaded.raw_pos[1], [2, 2, 2])
    np.testing.assert_array_equal(loaded.pos[1], [2, 2, 8])
    np.testing.assert_array_equal(loaded.spatial_resolution, [1, 1, 4])


def test_load_bundled_legacy_file():
    lT = LineageTree.load("tests/data/test-mamut.lT")
    np.testing.assert_array_equal(lT.spatial_resolution, [1, 1, 1])
    _assert_consistent(lT)


def test_loaders_accept_a_resolution():
    plain = read_from_mamut_xml("tests/data/test-mamut.xml")
    scaled = read_from_mamut_xml(
        "tests/data/test-mamut.xml", spatial_resolution=RESOLUTION
    )
    node = next(iter(plain.nodes))
    np.testing.assert_array_equal(scaled.raw_pos[node], plain.pos[node])
    np.testing.assert_allclose(
        scaled.pos[node], plain.pos[node] * np.array(RESOLUTION)
    )


def test_swc_loader_accepts_a_resolution():
    lT = read_from_swc(
        "tests/data/204-2-6nj.CNG.swc", spatial_resolution=RESOLUTION
    )
    _assert_consistent(lT)
    np.testing.assert_array_equal(lT.spatial_resolution, RESOLUTION)
