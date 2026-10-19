import xml.etree.ElementTree as ET
from collections import defaultdict

from lineagetree import LineageTree

SVG = "{http://www.w3.org/2000/svg}"


def _tree():
    """A tree whose leaves end at different times.

    The root divides at time 1. Its first daughter divides at time 3 and
    both of its daughters last until time 6. Its second daughter divides at
    time 2: one of its daughters stops at time 3, before every other leaf,
    the other one lasts until time 6.
    """
    successor = {
        0: [1],
        1: [2, 3],
        2: [4],
        4: [6, 7],
        6: [8],
        8: [10],
        10: [],
        7: [9],
        9: [11],
        11: [],
        3: [12, 13],
        12: [],
        13: [14],
        14: [15],
        15: [16],
        16: [],
    }
    return LineageTree(successor=successor)


def _draw(tmp_path, lT, **kwargs):
    path = tmp_path / "tree.svg"
    lT.write_to_svg(
        str(path), vert_space_factor=1, horizontal_space=1, **kwargs
    )
    root = ET.parse(path).getroot()
    circles = [
        (float(c.get("cx")), float(c.get("cy")), float(c.get("r")))
        for c in root.iter(f"{SVG}circle")
    ]
    lines = [
        tuple(float(line.get(k)) for k in ("x1", "y1", "x2", "y2"))
        for line in root.iter(f"{SVG}line")
    ]
    return float(root.get("width")), float(root.get("height")), circles, lines


def test_write_to_svg_nodes_do_not_overlap(tmp_path):
    lT = _tree()
    _, _, circles, _ = _draw(tmp_path, lT)
    assert len(circles) == len(lT.nodes)
    per_time = defaultdict(list)
    for x, y, _ in circles:
        per_time[y].append(x)
    for xs in per_time.values():
        assert len(set(xs)) == len(xs)


def test_write_to_svg_edges_do_not_cross(tmp_path):
    _, _, _, lines = _draw(tmp_path, _tree())
    for x1, y1, x2, y2 in lines:
        for u1, v1, u2, v2 in lines:
            if (y1, v1) == (y2, v2) or y1 != v1:
                continue
            # two edges starting at the same time keep their left-right order
            assert (x1 - u1) * (x2 - u2) >= 0


def test_write_to_svg_canvas_fits_the_drawing(tmp_path):
    width, height, circles, _ = _draw(tmp_path, _tree())
    for x, y, r in circles:
        assert r <= x <= width - r
        assert r <= y <= height - r
    # the drawing starts at the first node, not at x = 0 or time 0
    assert min(x - r for x, _, r in circles) < 2
    assert min(y - r for _, y, r in circles) < 2


def test_write_to_svg_does_not_change_the_tree(tmp_path):
    lT = _tree()
    _draw(tmp_path, lT)
    assert not hasattr(lT, "vert_space_factor")
