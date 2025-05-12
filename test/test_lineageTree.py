import warnings

import pytest

from LineageTree import (
    lineageTree,
    lineageTreeManager,
    read_from_mamut_xml,
    read_from_mastodon,
)


def test_read_MaMuT_xml():
    lT = read_from_mastodon("test/data/test.mastodon")
    assert lT.name == "test"
    assert len(lT.roots) == 3
    assert len(lT.nodes) == 41
    assert len(lT.successor) == 41
    assert len(lT.find_leaves(40)) == 2
    lT = read_from_mamut_xml("test/data/test-mamut.xml")
    assert lT.name == "test-mamut"
    assert len(lT.nodes) == 2430
    assert len(lT.successor) == 2430


def test_all_chains():
    lT = read_from_mamut_xml("test/data/test-mamut.xml")
    assert len(lT.all_chains) == 18


def test_uted_2levels_vs_3levels():
    lT = lineageTree()
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
    assert (
        lT.unordered_tree_edit_distance(t1, t2, style="full", norm=None) == 40
    )
    assert (
        lT.unordered_tree_edit_distance(t1, t2, style="mini", norm=None) == 4
    )
    assert lT.unordered_tree_edit_distance(
        t1, t2, style="normalized_simple", norm="max"
    )


def test_adding_nodes():
    lT = lineageTree()
    t1 = lT.add_root(0)
    first_level_end = lT.add_chain(t1, 9, downstream=True)

    lT.add_chain(first_level_end, 10, downstream=True)
    lT.add_chain(first_level_end, 10, downstream=True)

    assert len(lT.get_subtree_nodes(t1)) == 30


def test_removing_nodes():
    lT = lineageTree()
    t1 = lT.add_root(0)
    first_level_end = lT.add_chain(t1, 9, downstream=True)

    second_level_1 = lT.add_chain(first_level_end, 10, downstream=True)
    lT.add_chain(first_level_end, 10, downstream=True)
    lT.remove_nodes(lT.get_node_chain(second_level_1))
    assert len(lT.get_subtree_nodes(t1)) == 20


def test_time_resolution():
    lT = lineageTree()
    lT.time_resolution = 3
    assert lT.time_resolution == 3


def test_loading():
    lT = lineageTree.load("test/data/test-mamut.lT")
    assert lT.time_resolution == 0
    lT.time_resolution = 1.51
    assert lT.time_resolution == 1.5


def test_cross_comparison():
    lT_1 = lineageTree()
    t1 = lT_1.add_root(0)
    first_level_end = lT_1.add_chain(t1, 9, downstream=True)
    node_1 = lT_1.get_node_chain(t1)[0]

    second_level_1 = lT_1.add_chain(first_level_end, 10, downstream=True)
    second_level_2 = lT_1.add_chain(first_level_end, 10, downstream=True)

    lT_1.add_chain(second_level_1, 10, downstream=True)
    lT_1.add_chain(second_level_1, 10, downstream=True)
    lT_1.add_chain(second_level_2, 10, downstream=True)
    lT_1.add_chain(second_level_2, 10, downstream=True)
    lT_1.time_resolution = 5

    lT_2 = lineageTree()
    t2 = lT_2.add_root(0)
    first_level_end = lT_2.add_chain(t2, 4, downstream=True)
    node_2 = lT_2.get_node_chain(t2)[0]

    second_level_1 = lT_2.add_chain(first_level_end, 5, downstream=True)
    second_level_2 = lT_2.add_chain(first_level_end, 5, downstream=True)

    lT_2.add_chain(second_level_1, 5, downstream=True)
    lT_2.add_chain(second_level_1, 5, downstream=True)
    lT_2.add_chain(second_level_2, 5, downstream=True)
    lT_2.add_chain(second_level_2, 5, downstream=True)
    lT_2.time_resolution = 10

    lTm1 = lineageTreeManager()
    lTm1.add(lT_1, name="embryo_1")
    lTm1.add(lT_2, name="embryo_2")
    assert lT_2.time_resolution == lT_2._time_resolution / 10
    assert len(lT_1.get_subtree_nodes(node_1)) == len(lT_2.get_subtree_nodes(node_2)) * 2
    assert (
        lTm1.cross_lineage_edit_distance(
            t1,
            "embryo_1",
            100,
            t2,
            "embryo_2",
            100,
            style="full",
        )
        == 0
    )
    assert (
        lTm1.cross_lineage_edit_distance(
            node_1,
            "embryo_1",
            100,
            node_2,
            "embryo_2",
            100,
            style="simple",
        )
        == 0
    )
    assert (
        lTm1.cross_lineage_edit_distance(
            node_1,
            "embryo_1",
            100,
            node_2,
            "embryo_2",
            100,
            style="normalized_simple",
        )
        == 0
    )
    assert (
        lTm1.cross_lineage_edit_distance(
            node_1,
            "embryo_1",
            100,
            node_2,
            "embryo_2",
            100,
            style="downsampled",
            downsample=20,
        )
        == 0
    )
    lT_3 = lineageTree()
    t1 = lT_3.add_root(0)
    first_level_end = lT_3.add_chain(t1, 4, downstream=True)
    node_3 = lT_3.get_node_chain(t1)[0]

    second_level_1 = lT_3.add_chain(first_level_end, 5, downstream=True)
    second_level_2 = lT_3.add_chain(first_level_end, 5, downstream=True)
    lT_3.time_resolution = 10
    lTm1.add(lT_3, "embryo_3")
    assert (
        lTm1.cross_lineage_edit_distance(
            node_1,
            "embryo_1",
            100,
            node_3,
            "embryo_2",
            100,
            style="downsampled",
            downsample=20,
        )
        < 1
    )


def test_plots():
    lT = read_from_mastodon("test/data/test.mastodon")
    assert len(lT.plot_all_lineages()) == 3
    assert len(lT.plot_subtree(40)) == 2


def test_removing_embryos_from_manager():
    lT_1 = lineageTree()
    t1 = lT_1.add_root(0)
    first_level_end = lT_1.add_chain(t1, 9, downstream=True)

    second_level_1 = lT_1.add_chain(first_level_end, 10, downstream=True)
    second_level_2 = lT_1.add_chain(first_level_end, 10, downstream=True)

    lT_1.add_chain(second_level_1, 10, downstream=True)
    lT_1.add_chain(second_level_1, 10, downstream=True)
    lT_1.add_chain(second_level_2, 10, downstream=True)
    lT_1.add_chain(second_level_2, 10, downstream=True)
    lT_1.time_resolution = 5

    lT_2 = lineageTree()
    t2 = lT_2.add_root(0)
    first_level_end = lT_2.add_chain(t2, 4, downstream=True)

    second_level_1 = lT_2.add_chain(first_level_end, 5, downstream=True)
    second_level_2 = lT_2.add_chain(first_level_end, 5, downstream=True)

    lT_2.add_chain(second_level_1, 5, downstream=True)
    lT_2.add_chain(second_level_1, 5, downstream=True)
    lT_2.add_chain(second_level_2, 5, downstream=True)
    lT_2.add_chain(second_level_2, 5, downstream=True)
    lT_2.time_resolution = 10

    lTm1 = lineageTreeManager()
    lTm1.add(lT_1, name="embryo_1")
    lTm1.add(lT_2, name="embryo_2")
    lTm1.remove_embryo("embryo_1")
    assert len(lTm1.lineagetrees) == 1


def test_successor():
    test_lT = lineageTree(
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
    lT = lineageTree(
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
    test_lT = lineageTree(
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
    lT = lineageTree(
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
    lT = lineageTree()


def test_time_warning():
    warnings.filterwarnings(
        "error"
    )  # raises warnings as errors so we can catch them when expected
    with pytest.raises(UserWarning) as excinfo:
        lT = lineageTree(
            successor={0: (1,)}, time={0: 1, 1: 2}, starting_time=3
        )
    assert (
        str(excinfo.value)
        == "Both `time` and `starting_time` were provided, `starting_time` was ignored."
    )
    warnings.filterwarnings("default")


def test_bad_leaf():
    with pytest.raises(ValueError) as excinfo:
        lT = lineageTree(
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
        lT = lineageTree(successor={2: (1,), 3: (2,), 4: (2,)})
    assert str(excinfo.value) == "Node can have at most one predecessor."


def test_bad_root_leaf_value():
    with pytest.raises(ValueError) as excinfo:
        lT = lineageTree(successor={1: (2,), 2: set()}, root_leaf_value=set())
    assert (
        str(excinfo.value)
        == "root_leaf_value should have at least one element."
    )


def test_successor_and_predecessor():
    with pytest.raises(ValueError) as excinfo:
        lT = lineageTree(successor={1: (2, 3)}, predecessor={2: 1, 3: 1})
    assert (
        str(excinfo.value)
        == "You cannot have both successors and predecessors."
    )


def test_cycles():
    with pytest.raises(ValueError) as excinfo:
        lT = lineageTree(successor={0: (1,), 1: (0,)})
    assert (
        str(excinfo.value)
        == "Cycles were found in the tree, there should not be any."
    )
