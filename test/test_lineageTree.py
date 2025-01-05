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

    third_level_1 = lT.add_branch(second_level_1, 10, reverse=True)
    third_level_2 = lT.add_branch(second_level_1, 10, reverse=True)
    third_level_3 = lT.add_branch(second_level_2, 10, reverse=True)
    third_level_4 = lT.add_branch(second_level_2, 10, reverse=True)

    # lT_2 =lineageTree()
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


def test_adding_nodes():
    lT = lineageTree()
    t1 = lT.add_node(0)
    lT.roots.add(0)
    lT.t_e = 0
    lT.t_b = 0
    first_level_end = lT.add_branch(t1, 9, reverse=True, move_timepoints=True)

    second_level_1 = lT.add_branch(first_level_end, 10, reverse=True)
    second_level_2 = lT.add_branch(first_level_end, 10, reverse=True)

    assert len(lT.get_sub_tree(t1)) == 30


def test_removing_nodes():
    lT = lineageTree()
    t1 = lT.add_node(0)
    lT.roots.add(0)
    lT.t_e = 0
    lT.t_b = 0
    first_level_end = lT.add_branch(t1, 9, reverse=True, move_timepoints=True)

    second_level_1 = lT.add_branch(first_level_end, 10, reverse=True)
    second_level_2 = lT.add_branch(first_level_end, 10, reverse=True)
    lT.remove_nodes(lT.get_cycle(second_level_1))
    assert len(lT.get_sub_tree(t1)) == 20
