import warnings

import numpy as np
import pytest

from lineagetree import (
    LineageTree,
    LineageTreeManager,
    read_from_mamut_xml,
    read_from_mastodon,
    tree_approximation,
)

lT1 = read_from_mamut_xml("tests/data/test-mamut.xml")
lT2 = read_from_mastodon("tests/data/test.mastodon")
lt = LineageTree.load("tests/data/demo.lT")


def test_read_MaMuT_xml():
    assert lT1.name == "test-mamut"
    assert len(lT1.nodes) == 2430
    assert len(lT1.successor) == 2430
    assert lT2.name == "test"
    assert len(lT2.roots) == 3
    assert len(lT2.nodes) == 41
    assert len(lT2.successor) == 41
    assert len(lT2.find_leaves(40)) == 2


@pytest.fixture(scope="session")
def test_write(tmp_path_factory):
    tmp_path = str(tmp_path_factory.mktemp("lineagetree")) + ".lT"
    lt.labels[list(lt.nodes)[0]] = "test"
    lt._comparisons = {(1, 2): 30}
    lt.write(str(tmp_path))
    return tmp_path


def test_load(test_write):
    lt2 = LineageTree.load(str(test_write))
    assert lt2._comparisons == {(1, 2): 30}
    assert lt.labels[list(lt.nodes)[0]] == "test"
    assert lt == lt2


def test_all_chains():
    assert len(lT1.all_chains) == 18


def test_uted_2levels_vs_3levels():
    lT = LineageTree()
    t1 = lT.add_root(0)
    first_level_end = lT.add_chain(t1, 10, True)

    second_level_1 = lT.add_chain(first_level_end, 10, downstream=True)
    second_level_2 = lT.add_chain(first_level_end, 10, downstream=True)

    lT.add_chain(second_level_1, 10, downstream=True)
    lT.add_chain(second_level_1, 10, downstream=True)
    lT.add_chain(second_level_2, 10, downstream=True)
    lT.add_chain(second_level_2, 10, downstream=True)

    t2 = lT.add_root(0)
    first_level_end = lT.add_chain(t2, 10, downstream=True)

    second_level_1 = lT.add_chain(first_level_end, 10, downstream=True)
    second_level_2 = lT.add_chain(first_level_end, 10, downstream=True)

    assert (
        lT.unordered_tree_edit_distance(t1, t2, style="simple", norm=None)
        == 40
    )
    assert lT.unordered_tree_edit_distance(t1, t2, style="downsampled")
    assert (
        lT.unordered_tree_edit_distance(t1, t2, style="full", norm=None) == 40
    )
    assert (
        lT.unordered_tree_edit_distance(t1, t2, style="mini", norm=None) == 4
    )
    assert lT.unordered_tree_edit_distance(
        t1, t2, style="normalized_simple", norm="max"
    )
    assert lT.plot_tree_distance_graphs(t1, t2, style="simple", norm=None)
    assert lT.plot_tree_distance_graphs(
        t1, t2, style="normalized_simple", norm=None
    )
    assert lT.plot_tree_distance_graphs(t1, t2, style="full", norm=None)
    assert lT.plot_tree_distance_graphs(
        t1, t2, style="downsampled", downsample=4, norm=None
    )
    assert lT.unordered_tree_edit_distances_at_time_t(10)
    assert lT.labelled_mappings(t1, t2)


def test_adding_nodes():
    lT = LineageTree()
    t1 = lT.add_root(0)
    first_level_end = lT.add_chain(t1, 9, downstream=True)

    lT.add_chain(first_level_end, 10, downstream=True)
    lT.add_chain(first_level_end, 10, downstream=True)

    assert len(lT.get_subtree_nodes(t1)) == 30


def test_removing_nodes():
    lT = LineageTree()
    t1 = lT.add_root(0)
    first_level_end = lT.add_chain(t1, 9, downstream=True)

    second_level_1 = lT.add_chain(first_level_end, 10, downstream=True)
    lT.add_chain(first_level_end, 10, downstream=True)
    lT.remove_nodes(lT.get_chain_of_node(second_level_1))
    assert len(lT.get_subtree_nodes(t1)) == 20


def test_time_resolution():
    lT = LineageTree()
    lT.time_resolution = 3
    assert lT.time_resolution == 3


def test_loading():
    lT = LineageTree.load("tests/data/test-mamut.lT")
    assert lT.time_resolution == 0
    lT.time_resolution = 1.51
    assert lT.time_resolution == 1.5


def test_cross_comparison():
    lT_1 = LineageTree()
    t1 = lT_1.add_root(0)
    first_level_end = lT_1.add_chain(t1, 9, downstream=True)
    node_1 = lT_1.get_chain_of_node(t1)[0]

    second_level_1 = lT_1.add_chain(first_level_end, 10, downstream=True)
    second_level_2 = lT_1.add_chain(first_level_end, 10, downstream=True)

    lT_1.add_chain(second_level_1, 10, downstream=True)
    lT_1.add_chain(second_level_1, 10, downstream=True)
    lT_1.add_chain(second_level_2, 10, downstream=True)
    lT_1.add_chain(second_level_2, 10, downstream=True)
    lT_1.time_resolution = 5

    lT_2 = LineageTree()
    t2 = lT_2.add_root(0)
    first_level_end = lT_2.add_chain(t2, 4, downstream=True)
    node_2 = lT_2.get_chain_of_node(t2)[0]

    second_level_1 = lT_2.add_chain(first_level_end, 5, downstream=True)
    second_level_2 = lT_2.add_chain(first_level_end, 5, downstream=True)

    lT_2.add_chain(second_level_1, 5, downstream=True)
    lT_2.add_chain(second_level_1, 5, downstream=True)
    lT_2.add_chain(second_level_2, 5, downstream=True)
    lT_2.add_chain(second_level_2, 5, downstream=True)
    lT_2.time_resolution = 10

    lTm1 = LineageTreeManager()
    lTm1.add(lT_1, name="embryo_1")
    lTm1.add(lT_2, name="embryo_2")
    assert lT_2.time_resolution == lT_2._time_resolution / 10
    assert (
        len(lT_1.get_subtree_nodes(node_1))
        == len(lT_2.get_subtree_nodes(node_2)) * 2
    )
    assert (
        lTm1.cross_lineage_edit_distance(
            t1,
            "embryo_1",
            t2,
            "embryo_2",
            100,
            100,
            style="full",
        )
        == 0
    )
    assert (
        lTm1.cross_lineage_edit_distance(
            node_1,
            "embryo_1",
            node_2,
            "embryo_2",
            100,
            100,
            style="simple",
        )
        == 0
    )
    assert (
        lTm1.cross_lineage_edit_distance(
            node_1,
            "embryo_1",
            node_2,
            "embryo_2",
            100,
            100,
            style="normalized_simple",
        )
        == 0
    )
    assert (
        lTm1.cross_lineage_edit_distance(
            node_1,
            "embryo_1",
            node_2,
            "embryo_2",
            100,
            100,
            style="downsampled",
            downsample=20,
        )
        == 0
    )
    lT_3 = LineageTree()
    t1 = lT_3.add_root(0)
    first_level_end = lT_3.add_chain(t1, 4, downstream=True)
    node_3 = lT_3.get_chain_of_node(t1)[0]

    second_level_1 = lT_3.add_chain(first_level_end, 5, downstream=True)
    second_level_2 = lT_3.add_chain(first_level_end, 5, downstream=True)
    lT_3.time_resolution = 10
    lTm1.add(lT_3, "embryo_3")
    assert (
        0
        < lTm1.cross_lineage_edit_distance(
            node_1,
            "embryo_1",
            node_3,
            "embryo_3",
            100,
            100,
            style="simple",
            downsample=20,
        )
        < 1
    )
    assert lTm1.plot_tree_distance_graphs(
        t1,
        "embryo_1",
        t2,
        "embryo_2",
        100,
        100,
        style="full",
    )
    assert lTm1.plot_tree_distance_graphs(
        t1,
        "embryo_1",
        t2,
        "embryo_2",
        100,
        100,
        style="simple",
    )
    assert lTm1.plot_tree_distance_graphs(
        t1,
        "embryo_1",
        t2,
        "embryo_2",
        100,
        100,
        style="downsampled",
        downsample=10,
    )
    assert lTm1.labelled_mappings(
        t1,
        "embryo_1",
        t2,
        "embryo_2",
        100,
        100,
        style="full",
    )
    assert lTm1.labelled_mappings(
        t1,
        "embryo_1",
        t2,
        "embryo_2",
        100,
        100,
        style="simple",
    )
    lTm1.clear_comparisons()
    assert lTm1._comparisons == {}


def test_plots():
    assert len(lT2.plot_all_lineages()) == 3
    assert len(lT2.plot_subtree(40)) == 2


def test_removing_embryos_from_manager():
    lT_1 = LineageTree()
    t1 = lT_1.add_root(0)
    first_level_end = lT_1.add_chain(t1, 9, downstream=True)

    second_level_1 = lT_1.add_chain(first_level_end, 10, downstream=True)
    second_level_2 = lT_1.add_chain(first_level_end, 10, downstream=True)

    lT_1.add_chain(second_level_1, 10, downstream=True)
    lT_1.add_chain(second_level_1, 10, downstream=True)
    lT_1.add_chain(second_level_2, 10, downstream=True)
    lT_1.add_chain(second_level_2, 10, downstream=True)
    lT_1.time_resolution = 5

    lT_2 = LineageTree()
    t2 = lT_2.add_root(0)
    first_level_end = lT_2.add_chain(t2, 4, downstream=True)

    second_level_1 = lT_2.add_chain(first_level_end, 5, downstream=True)
    second_level_2 = lT_2.add_chain(first_level_end, 5, downstream=True)

    lT_2.add_chain(second_level_1, 5, downstream=True)
    lT_2.add_chain(second_level_1, 5, downstream=True)
    lT_2.add_chain(second_level_2, 5, downstream=True)
    lT_2.add_chain(second_level_2, 5, downstream=True)
    lT_2.time_resolution = 10

    lTm1 = LineageTreeManager()
    lTm1.add(lT_1, name="embryo_1")
    lTm1.add(lT_2, name="embryo_2")
    lTm1.remove_embryo("embryo_1")
    assert len(lTm1.lineagetrees) == 1
    for k, _e in lTm1:
        assert k == "embryo_2"
    assert lTm1["embryo_2"]


def test_successor():
    test_lT = LineageTree(
        successor={
            1: (2,),
            2: (3, 100),
            100: (101,),
            0: (1,),
            10: (0,),
            5: (),
            3: (),
            4: (),
            101: (),
        }
    )
    lT = LineageTree(
        successor={
            1: (2,),
            2: (3, 100),
            100: [
                101,
            ],
            3: (),
            4: None,
            5: set(),
            0: (1,),
            10: (0,),
        }
    )
    assert lT == test_lT


def test_predecessor():
    test_lT = LineageTree(
        successor={
            1: (2,),
            2: (3, 100),
            100: (101,),
            0: (1,),
            10: (0,),
            5: (),
            3: (),
            4: (),
            101: (),
        }
    )
    lT = LineageTree(
        predecessor={
            2: (1,),
            3: [2],
            100: 2,
            101: (100,),
            4: set(),
            5: None,
            1: 0,
            0: 10,
        }
    )
    assert lT == test_lT


def test_empty():
    LineageTree()


def test_time_warning():
    warnings.filterwarnings(
        "error"
    )  # raises warnings as errors so we can catch them when expected
    with pytest.raises(UserWarning) as excinfo:
        LineageTree(successor={0: (1,)}, time={0: 1, 1: 2}, starting_time=3)
    assert (
        str(excinfo.value)
        == "Both `time` and `starting_time` were provided, `starting_time` was ignored."
    )
    warnings.filterwarnings("default")


def test_bad_leaf():
    with pytest.raises(ValueError) as excinfo:
        LineageTree(
            successor={
                1: (2,),
                2: (3, 100),
                100: [
                    101,
                ],
                3: (),
                4: None,
                5: set(),
                0: (1,),
                10: (0,),
            },
            root_leaf_value=[None],
        )
    assert (
        str(excinfo.value)
        == "() was not declared as a leaf but was found as a successor.\nPlease lift the ambiguity."
    )


def test_multiple_predecessors():
    with pytest.raises(ValueError) as excinfo:
        LineageTree(successor={2: (1,), 3: (2,), 4: (2,)})
    assert str(excinfo.value) == "Node can have at most one predecessor."


def test_bad_root_leaf_value():
    with pytest.raises(ValueError) as excinfo:
        LineageTree(successor={1: (2,), 2: set()}, root_leaf_value=set())
    assert (
        str(excinfo.value)
        == "root_leaf_value should have at least one element."
    )


def test_successor_and_predecessor():
    with pytest.raises(ValueError) as excinfo:
        LineageTree(successor={1: (2, 3)}, predecessor={2: 1, 3: 1})
    assert (
        str(excinfo.value)
        == "You cannot have both successors and predecessors."
    )


def test_cycles():
    with pytest.raises(ValueError) as excinfo:
        LineageTree(successor={0: (1,), 1: (0,)})
    assert (
        str(excinfo.value)
        == "Cycles were found in the tree, there should not be any."
    )


def test_time_nodes():
    assert lT1.time_nodes[131] == {
        108735,
        114627,
        129407,
        138526,
        148274,
        165742,
        169927,
        178305,
    }
    all_cells = set(lT1.nodes)
    no_cells = set()
    for c in lT1.time_nodes.values():
        all_cells.difference_update(c)
        no_cells.update(set(c))

    assert no_cells == lT1.nodes
    assert all_cells == set()


def test_depth():
    assert lT1.depth[128223] == 99


def test_leaves():
    assert list(lT1.leaves)[0] == 181669


def test_edges():
    assert lT1.edges[0] == (106632, 106589)


def test_parenting():
    assert lT2.parenting[0, 2] == 1


def test_equality():
    assert lT1 == lT1
    assert lT2 == lT2
    assert lT1 != lT2


def test_next_id():
    assert lT1.get_next_id() == 182893
    assert lT1.get_next_id() == 182894


def test_dynamic_property():
    lT = LineageTree()
    assert lT.nodes == frozenset()
    t1 = lT.add_root(0)
    assert lT.nodes == frozenset({1})
    lT.add_chain(t1, 10, True)
    assert lT.nodes == frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})


def test_idx3d():
    kdtree, idxs = lT1.get_idx3d(0)
    assert np.isclose(kdtree.query((0, 0, 0))[0], 1131.2352660153383)
    assert idxs[kdtree.query((0, 0, 0))[1]] == 110826
    assert idxs[kdtree.query((1000, 2000, 1000))[1]] == 132063


def test_gabriel_graph():
    gg = lT1.get_gabriel_graph(0)
    assert gg[173618] == {110832, 168322}


def test_get_chain_of_node():
    chain = lT1.get_chain_of_node(173618)
    assert len(chain) == 273
    assert chain[-1] == 181669


def test_get_all_chains_of_subtree():
    assert (
        lT1.get_chain_of_node(173618)
        == lT1.get_all_chains_of_subtree(173618)[0]
    )


def test_get_ancestor_with_attribute():
    lT1.label.pop(178353)
    assert lT1.get_ancestor_with_attribute(178353, "label") == 178336


def test_get_subtree():
    assert lT1.get_subtree(lT1.nodes) == lT1


def test_find_leaves():
    assert lT1.find_leaves(173618) == {lT1.get_chain_of_node(173618)[-1]}


def test_get_subtree_nodes():
    assert lT1.get_chain_of_node(173618) == lT1.get_subtree_nodes(173618)


def test_spatial_density():
    density = list(lT1.compute_spatial_density(0, th=40).values())
    assert np.count_nonzero(density) == 1669


def test_compute_k_nearest_neighbours():
    assert lT1.compute_k_nearest_neighbours()[169994] == {
        108588,
        114722,
        129276,
        139163,
        148361,
        165681,
        169994,
        178396,
    }


def test_compute_spatial_edges():
    assert lT1.compute_spatial_edges()[129294] == {139162, 148358}


def test_get_ancestor_at_t():
    assert lT1.get_ancestor_at_t(175903, 0) == 173618


def get_labelled_ancestor():
    assert lT1.get_labelled_ancestor(175903) == 173618


def test_unordered_tree_edit_distances_at_time_t():
    assert np.isclose(
        lT1.unordered_tree_edit_distances_at_time_t(0, style="simple")[
            (110832, 132129)
        ],
        0.7321711568938193,
    )


def test_unordered_tree_edit_distance():
    assert np.isclose(
        lT1.unordered_tree_edit_distance(110832, 132129), 0.7321711568938193
    )


def test_non_return_functions():
    lT1.plot_all_lineages()
    lT1.plot_subtree(110832)
    lT1.plot_dtw_heatmap(110832, 132129)
    lT1.plot_dtw_trajectory(110832, 132129)


def test_nodes_at_t():
    assert lT1.nodes_at_t(None, 110832) == [123641]
    assert lT1.nodes_at_t(65, 110832) == [112436]


def test_calculate_dtw():
    assert np.isclose(lT1.calculate_dtw(110832, 132129)[0], 25.550036305019194)


def test_create_new_style():
    class new_tree(tree_approximation.simple_tree):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def delta(self, x, y, corres1, corres2, times1, times2):
            if x is None:
                return 1
            if y is None:
                return 1
            return abs(times1[corres1[x]] - times2[corres2[y]]) / (
                times1[corres1[x]] + times2[corres2[y]]
            )

        def get_norm(self, root) -> int:
            return len(
                self.lT.get_all_chains_of_subtree(root, end_time=self.end_time)
            )

    assert lt.unordered_tree_edit_distance(
        176, 29345, style=new_tree
    ) == lt.unordered_tree_edit_distance(176, 29345, style="normalized_simple")


def test_get_ancestor_with():
    assert lt.get_labelled_ancestor(
        list(lt.nodes)[0]
    ) == lt.get_ancestor_with_attribute(list(lt.nodes)[0], "labels")


def test_mastodon_labeling():
    assert lT2.labels[25] == "p"
    assert lT2.labels[40] == "p(2)"
    assert lT2.labels_name == "E"


def test_available_labels():
    assert lT2.get_available_labels() == ["E", "Ep", "Er", "El", "Extoderms"]


def test_change_labels():
    lT2.change_labels("Ep")
    assert lT2.labels[19] == "alla"
    assert lT2.labels[9] == "right1"

    lT2.change_labels("test", {19: "a", 9: "b"})
    assert lT2.labels[19] == "a"
    assert lT2.labels[9] == "b"

    lT2.change_labels("Ep", only_first_node_in_chain=True)
    assert lT2.labels == {
        0: "right1",
        1: "right1",
        40: "right1",
        16: "left",
        19: "alla",
        24: "left",
        25: "left",
    }
