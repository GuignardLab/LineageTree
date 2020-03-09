from LineageTree import lineageTree

def test_read_MaMuT_xml():
    lT = lineageTree('test/data/test-mamut.xml', MaMuT=True)
    assert len(lT.nodes) == 2430
    assert len(lT.successor) == 2418