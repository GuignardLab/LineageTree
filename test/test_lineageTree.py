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
