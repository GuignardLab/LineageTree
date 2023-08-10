from LineageTree import lineageTree

lT = lineageTree("test/data/test-mamut.xml", file_type="MaMuT")
lT = lineageTree("test/data/test.mastodon", file_type="mastodon")

def test_read_MaMuT_xml():
    assert len(lT.nodes) == 2430
    assert len(lT.successor) == 2418


def test_writting_svg():
    lT.write_to_svg("test/test.svg")
