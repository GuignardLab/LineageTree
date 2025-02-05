from LineageTree import lineageTree, lineageTreeManager


def test_read_MaMuT_xml():
    lT = lineageTree("test/data/test.mastodon", file_type="mastodon")
    assert len(lT.roots) == 3
    assert len(lT.nodes) == 41
    assert len(lT.successor) == 36
    assert len(lT.find_leaves(40)) == 2
    lT = lineageTree("test/data/test-mamut.xml", file_type="MaMuT")
    assert len(lT.nodes) == 2430
    assert len(lT.successor) == 2418


def test_writting_svg():
    lT = lineageTree("test/data/test-mamut.xml", file_type="MaMuT")
    lT = lineageTree("test/data/test.mastodon", file_type="mastodon")
    lT.write_to_svg("test/test.svg")


def test_all_tracks():
    lT = lineageTree("test/data/test-mamut.xml", file_type="MaMuT")
    assert len(lT.all_tracks) == 18


def test_uted_2levels_vs_3levels():
    lT = lineageTree()
    t1 = lT.add_node(0)
    lT.roots.add(0)
    lT.t_e = 0
    lT.t_b = 0
    first_level_end = lT.add_branch(t1, 10, reverse=True, move_timepoints=True)

    second_level_1 = lT.add_branch(first_level_end, 10, reverse=True)
    second_level_2 = lT.add_branch(first_level_end, 10, reverse=True)

    lT.add_branch(second_level_1, 10, reverse=True)
    lT.add_branch(second_level_1, 10, reverse=True)
    lT.add_branch(second_level_2, 10, reverse=True)
    lT.add_branch(second_level_2, 10, reverse=True)

    t2 = lT.add_node(0)
    lT.roots.add(0)
    lT.t_e = 0
    lT.t_b = 0
    first_level_end = lT.add_branch(t2, 10, reverse=True, move_timepoints=True)

    second_level_1 = lT.add_branch(first_level_end, 10, reverse=True)
    second_level_2 = lT.add_branch(first_level_end, 10, reverse=True)

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
    t1 = lT.add_node(0)
    lT.roots.add(0)
    lT.t_e = 0
    lT.t_b = 0
    first_level_end = lT.add_branch(t1, 10, reverse=True, move_timepoints=True)

    second_level_1 = lT.add_branch(first_level_end, 10, reverse=True)
    second_level_2 = lT.add_branch(first_level_end, 10, reverse=True)

    lT.add_branch(second_level_1, 10, reverse=True)
    lT.add_branch(second_level_1, 10, reverse=True)
    lT.add_branch(second_level_2, 10, reverse=True)
    lT.add_branch(second_level_2, 10, reverse=True)

    t2 = lT.add_node(0)
    lT.roots.add(0)
    lT.t_e = 0
    lT.t_b = 0
    first_level_end = lT.add_branch(t2, 10, reverse=True, move_timepoints=True)

    second_level_1 = lT.add_branch(first_level_end, 10, reverse=True)
    second_level_2 = lT.add_branch(first_level_end, 10, reverse=True)

    new = lT.fuse_lineage_tree(t1, t2, length=10)
    assert len(lT.get_sub_tree(new)) == 110
    assert len(lT.get_cycle(new)) == 10

    new2 = lT.cut_tree(new)
    assert len(lT.get_sub_tree(new)) + 40 == len(lT.get_sub_tree(new2))


def test_adding_nodes():
    lT = lineageTree()
    t1 = lT.add_node(0)
    lT.roots.add(0)
    lT.t_e = 0
    lT.t_b = 0
    first_level_end = lT.add_branch(t1, 9, reverse=True, move_timepoints=True)

    lT.add_branch(first_level_end, 10, reverse=True)
    lT.add_branch(first_level_end, 10, reverse=True)

    assert len(lT.get_sub_tree(t1)) == 30


def test_removing_nodes():
    lT = lineageTree()
    t1 = lT.add_node(0)
    lT.roots.add(t1)
    lT.t_e = 0
    lT.t_b = 0
    first_level_end = lT.add_branch(t1, 9, reverse=True, move_timepoints=True)

    second_level_1 = lT.add_branch(first_level_end, 10, reverse=True)
    lT.add_branch(first_level_end, 10, reverse=True)
    lT.remove_nodes(lT.get_cycle(second_level_1))
    assert len(lT.get_sub_tree(t1)) == 20


def test_modifying_nodes():
    lT = lineageTree()
    t1 = lT.add_node(0)
    lT.roots.add(t1)
    lT.t_e = 0
    lT.t_b = 0
    lT.modify_branch(t1, 100)
    assert len(lT.get_cycle(t1)) == 100


def test_modifying_nodes_2():
    lT = lineageTree()
    t1 = lT.add_node(0)
    lT.roots.add(t1)
    lT.t_e = 0
    lT.t_b = 0
    lT.add_branch(t1, 9, reverse=True, move_timepoints=True)
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
    t1 = lT.add_node(0)
    lT.roots.add(t1)
    lT.t_b = 0
    lT.t_e = 0
    lT.add_branch(t1, 10, reverse=True, move_timepoints=True)

    t2 = lT.add_node(0)
    lT.roots.add(t2)
    lT.add_branch(t2, 11, reverse=True, move_timepoints=True)

    lT.t_e = 40
    lT.complete_lineage()
    assert len(lT.nodes) == 82


def test_cross_comparison():
    lT_1 = lineageTree(
        "/home/giannis/programs/tree_stuff/LineageTree/test/data/test-mamut.xml",
        file_type="MaMuT",
    )
    lT_1.time_resolution = 1
    lT_2 = lineageTree(
        "/home/giannis/programs/tree_stuff/LineageTree/test/data/test-mamut.xml",
        file_type="MaMuT",
    )
    lT_2.remove_nodes(lT_2.get_sub_tree(168322))
    lT_2.time_resolution = 10

    lTm1 = lineageTreeManager()
    lTm1.add(lT_1, name="embryo_1")
    lTm1.add(lT_2, name="embryo_2")

    lT_1 = lineageTree(
        "test/data/test-mamut.xml",
        file_type="MaMuT",
    )
    lT_1.time_resolution = 5
    lT_2 = lineageTree(
        "test/data/test-mamut.xml",
        file_type="MaMuT",
    )
    lT_2.remove_nodes(lT_2.get_sub_tree(168322))
    lT_2.time_resolution = 10
    assert lT_2.time_resolution == lT_2._time_resolution / 10

    lTm2 = lineageTreeManager()
    lTm2.add(lT_1, name="embryo_1")
    lTm2.add(lT_2, name="embryo_2")

    assert lTm2.cross_lineage_edit_distance(
        110832,
        "embryo_1",
        1,
        110832,
        "embryo_2",
        100,
        style="full",
    ) != lTm1.cross_lineage_edit_distance(
        110832,
        "embryo_1",
        1,
        110832,
        "embryo_2",
        100,
        style="full",
    )
    assert lTm2.cross_lineage_edit_distance(
        110832,
        "embryo_1",
        1,
        110832,
        "embryo_2",
        100,
        style="simple",
    ) != lTm1.cross_lineage_edit_distance(
        110832,
        "embryo_1",
        1,
        110832,
        "embryo_2",
        100,
        style="simple",
    )
    assert lTm2.cross_lineage_edit_distance(
        110832,
        "embryo_1",
        1,
        110832,
        "embryo_2",
        100,
        style="normalized_simple",
    ) != lTm1.cross_lineage_edit_distance(
        110832,
        "embryo_1",
        1,
        110832,
        "embryo_2",
        100,
        style="normalized_simple",
    )
    assert lTm2.cross_lineage_edit_distance(
        110832,
        "embryo_1",
        1,
        110832,
        "embryo_2",
        100,
        style="downsampled",
        downsample=100,
    ) != lTm1.cross_lineage_edit_distance(
        110832,
        "embryo_1",
        1,
        110832,
        "embryo_2",
        100,
        style="downsampled",
        downsample=100,
    )


def test_plots():
    lT = lineageTree("test/data/test.mastodon", file_type="mastodon")
    assert len(lT.plot_all_lineages()) == 3
    assert len(lT.plot_node(40)) == 2
