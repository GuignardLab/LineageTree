from LineageTree import lineageTree


def test_read_MaMuT_xml():
    lT = lineageTree("test/data/test.mastodon", file_type="mastodon")
    assert len(lT.nodes) == 41
    assert len(lT.successor) == 36
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
