from LineageTree import (
    lineageTree,
    lineageTreeManager,
    read_from_mamut_xml,
    read_from_mastodon,
)


def test_read_MaMuT_xml():
    lT = read_from_mastodon("test/data/test.mastodon")
    assert len(lT.roots) == 3
    assert len(lT.nodes) == 41
    assert len(lT.successor) == 41
    assert len(lT.find_leaves(40)) == 2
    lT = read_from_mamut_xml("test/data/test-mamut.xml")
    assert len(lT.nodes) == 2430
    assert len(lT.successor) == 2430


def test_writting_svg():
    lT = read_from_mamut_xml("test/data/test-mamut.xml")
    lT = read_from_mastodon("test/data/test.mastodon")
    lT.write_to_svg("test/test.svg")


def test_all_tracks():
    lT = read_from_mamut_xml("test/data/test-mamut.xml")
    assert len(lT.all_tracks) == 18


def test_uted_2levels_vs_3levels():
    lT = lineageTree()
    t1 = lT.add_root(0)
    lT.roots.add(0)
    lT.t_e = 0
    lT.t_b = 0
    first_level_end = lT.add_branch(t1, 10, True)

    second_level_1 = lT.add_branch(first_level_end, 10, downstream=True)
    second_level_2 = lT.add_branch(first_level_end, 10, downstream=True)

    lT.add_branch(second_level_1, 10, downstream=True)
    lT.add_branch(second_level_1, 10, downstream=True)
    lT.add_branch(second_level_2, 10, downstream=True)
    lT.add_branch(second_level_2, 10, downstream=True)

    t2 = lT.add_root(0)
    lT.roots.add(0)
    lT.t_e = 0
    lT.t_b = 0
    first_level_end = lT.add_branch(t2, 10, downstream=True)

    second_level_1 = lT.add_branch(first_level_end, 10, downstream=True)
    second_level_2 = lT.add_branch(first_level_end, 10, downstream=True)

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


def test_fusion():
    lT = lineageTree()
    t1 = lT.add_root(0)
    lT.roots.add(0)
    lT.t_e = 0
    lT.t_b = 0
    first_level_end = lT.add_branch(t1, 10, downstream=True)

    second_level_1 = lT.add_branch(first_level_end, 10, downstream=True)
    second_level_2 = lT.add_branch(first_level_end, 10, downstream=True)

    lT.add_branch(second_level_1, 10, downstream=True)
    lT.add_branch(second_level_1, 10, downstream=True)
    lT.add_branch(second_level_2, 10, downstream=True)
    lT.add_branch(second_level_2, 10, downstream=True)

    t2 = lT.add_root(0)
    lT.roots.add(0)
    lT.t_e = 0
    lT.t_b = 0
    first_level_end = lT.add_branch(t2, 10, downstream=True)

    second_level_1 = lT.add_branch(first_level_end, 10, downstream=True)
    second_level_2 = lT.add_branch(first_level_end, 10, downstream=True)

    new = lT.fuse_lineage_tree(t1, t2, length=10)
    assert len(lT.get_sub_tree(new)) == 110
    assert len(lT.get_cycle(new)) == 10

    new2 = lT.cut_tree(new)
    assert len(lT.get_sub_tree(new)) + 40 == len(lT.get_sub_tree(new2))


def test_adding_nodes():
    lT = lineageTree()
    t1 = lT.add_root(0)
    lT.roots.add(0)
    lT.t_e = 100
    lT.t_b = 0
    first_level_end = lT.add_branch(t1, 9, downstream=True)

    lT.add_branch(first_level_end, 10, downstream=True)
    lT.add_branch(first_level_end, 10, downstream=True)

    assert len(lT.get_sub_tree(t1)) == 30


def test_removing_nodes():
    lT = lineageTree()
    t1 = lT.add_root(0)
    lT.roots.add(t1)
    lT.t_e = 0
    lT.t_b = 0
    first_level_end = lT.add_branch(t1, 9, downstream=True)

    second_level_1 = lT.add_branch(first_level_end, 10, downstream=True)
    lT.add_branch(first_level_end, 10, downstream=True)
    lT.remove_nodes(lT.get_cycle(second_level_1))
    assert len(lT.get_sub_tree(t1)) == 20


def test_modifying_nodes():
    lT = lineageTree()
    t1 = lT.add_root(0)
    lT.roots.add(t1)
    lT.t_e = 0
    lT.t_b = 0
    lT.modify_branch(t1, 100)
    assert len(lT.get_cycle(t1)) == 100


def test_modifying_nodes_2():
    lT = lineageTree()
    t1 = lT.add_root(0)
    lT.roots.add(t1)
    lT.t_e = 0
    lT.t_b = 0
    lT.add_branch(t1, 9, downstream=True)
    lT.modify_branch(t1, 100)
    assert len(lT.get_sub_tree(t1)) == 100


def test_time_resolution():
    lT = lineageTree()
    lT.time_resolution = 3
    assert lT.time_resolution == 3


def test_loading():
    lT = lineageTree.load("test/data/test-mamut.lT")
    assert lT.time_resolution == 1


def test_complete_lineage():
    lT = lineageTree()
    t1 = lT.add_root(0)
    lT.roots.add(t1)
    lT.t_b = 0
    lT.t_e = 0
    lT.add_branch(t1, 10, downstream=True)

    t2 = lT.add_root(0)
    lT.roots.add(t2)
    lT.add_branch(t2, 11, downstream=True)

    lT.t_e = 40
    lT.complete_lineage()
    assert len(lT.nodes) == 82


def test_cross_comparison():
    lT_1 = lineageTree()
    t1 = lT_1.add_root(0)
    lT_1.t_e = 0
    lT_1.t_b = 0
    first_level_end = lT_1.add_branch(t1, 9, downstream=True)
    node_1 = lT_1.get_cycle(t1)[0]

    second_level_1 = lT_1.add_branch(first_level_end, 10, downstream=True)
    second_level_2 = lT_1.add_branch(first_level_end, 10, downstream=True)

    lT_1.add_branch(second_level_1, 10, downstream=True)
    lT_1.add_branch(second_level_1, 10, downstream=True)
    lT_1.add_branch(second_level_2, 10, downstream=True)
    lT_1.add_branch(second_level_2, 10, downstream=True)
    lT_1.time_resolution = 5

    lT_2 = lineageTree()
    t2 = lT_2.add_root(0)
    lT_2.t_e = 0
    lT_2.t_b = 0
    first_level_end = lT_2.add_branch(t2, 4, downstream=True)
    node_2 = lT_2.get_cycle(t2)[0]

    second_level_1 = lT_2.add_branch(first_level_end, 5, downstream=True)
    second_level_2 = lT_2.add_branch(first_level_end, 5, downstream=True)

    lT_2.add_branch(second_level_1, 5, downstream=True)
    lT_2.add_branch(second_level_1, 5, downstream=True)
    lT_2.add_branch(second_level_2, 5, downstream=True)
    lT_2.add_branch(second_level_2, 5, downstream=True)
    lT_2.time_resolution = 10

    lTm1 = lineageTreeManager()
    lTm1.add(lT_1, name="embryo_1")
    lTm1.add(lT_2, name="embryo_2")
    assert lT_2.time_resolution == lT_2._time_resolution / 10
    assert len(lT_1.get_sub_tree(node_1)) == len(lT_2.get_sub_tree(node_2)) * 2
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
    lT_3.t_e = 0
    lT_3.t_b = 0
    first_level_end = lT_3.add_branch(t1, 4, downstream=True)
    node_3 = lT_3.get_cycle(t1)[0]

    second_level_1 = lT_3.add_branch(first_level_end, 5, downstream=True)
    second_level_2 = lT_3.add_branch(first_level_end, 5, downstream=True)
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
    assert len(lT.plot_node(40)) == 2
