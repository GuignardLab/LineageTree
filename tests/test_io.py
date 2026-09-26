"""Loaders and writers, on small synthetic files."""

import pickle
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from lineagetree import (
    LineageTree,
    read_from_ASTEC,
    read_from_csv,
    read_from_mamut_xml,
    read_from_mastodon,
    read_from_tgmm_xml,
    read_from_txt_for_celegans_CAO,
)

# ---------------------------------------------------------------- loaders


def test_read_celegans_cao(tmp_path):
    """The loader used `np.float` and overwrote its label dictionary."""
    path = tmp_path / "cao.txt"
    path.write_text(
        "cell time z x y\n"
        "ABa 1 1.0 2.0 3.0\n"
        "ABa 2 1.5 2.5 3.5\n"
        "ABaa 3 1.0 1.0 1.0\n"
        "ABap 3 2.0 2.0 2.0\n"
    )
    lT = read_from_txt_for_celegans_CAO(str(path))
    assert len(lT.nodes) == 4
    assert len(lT.roots) == 1
    (root,) = lT.roots
    assert lT.label[root] == "ABa"
    assert sorted(lT.label.values()) == ["ABa", "ABa", "ABaa", "ABap"]
    chain = lT.get_chain_of_node(root)
    assert len(chain) == 2
    assert sorted(lT.label[s] for s in lT.successor[chain[-1]]) == [
        "ABaa",
        "ABap",
    ]
    np.testing.assert_array_equal(lT.pos[root], [2.0, 3.0, 1.0])
    assert lT.name == "cao"


def _tgmm_cell(cell_id, parent, m, lineage=0):
    return ET.Element(
        "GaussianMixtureModel",
        {
            "id": str(cell_id),
            "parent": str(parent),
            "lineage": str(lineage),
            "m": " ".join(map(str, m)),
            "svIdx": "1 2",
            "alpha": "5",
            "alphaPrior": "1",
            "nu": "2",
            "W": " ".join(["1"] * 9),
        },
    )


def test_read_tgmm_xml_gives_unique_ids(tmp_path):
    """Ids used to restart at 0 in each time point file."""
    frames = {
        0: [_tgmm_cell(0, -1, [0, 0, 0]), _tgmm_cell(1, -1, [9, 9, 9], 1)],
        1: [_tgmm_cell(0, 0, [1, 0, 0]), _tgmm_cell(1, 0, [0, 1, 0])],
    }
    for t, cells in frames.items():
        root = ET.Element("document")
        root.extend(cells)
        ET.ElementTree(root).write(tmp_path / f"frame_t{t:03d}.xml")

    lT = read_from_tgmm_xml(
        str(tmp_path / "frame_t{t:03d}.xml"), 0, 1, z_mult=2
    )
    assert len(lT.nodes) == 4
    assert sorted(lT.time.values()) == [0, 0, 1, 1]
    dividing = [n for n in lT.nodes if len(lT.successor[n]) == 2]
    assert len(dividing) == 1
    np.testing.assert_array_equal(lT.pos[dividing[0]], [0, 0, 0])
    assert len(lT.leaves) == 3
    assert lT.intensity[dividing[0]] == 4


def test_read_csv_keeps_isolated_nodes(tmp_path):
    """Nodes with no predecessor and no successor used to be dropped."""
    path = tmp_path / "tracks.csv"
    # id, time, z, y, x, pred_id, lin_id
    path.write_text(
        "1, 0, 0, 0, 0, -1, 0\n"
        "2, 1, 0, 0, 1, 1, 0\n"
        "3, 0, 5, 5, 5, -1, 1\n"
    )
    lT = read_from_csv(str(path))
    assert len(lT.nodes) == 3
    assert len(lT.roots) == 2
    assert len(lT.edges) == 1


def _mamut_xml(path):
    spots = {0: [(1, 0.0), (2, 9.0)], 1: [(3, 0.0)]}
    root = ET.Element("TrackMate")
    model = ET.SubElement(root, "Model")
    ET.SubElement(model, "FeatureDeclarations")
    all_spots = ET.SubElement(model, "AllSpots")
    for frame, frame_spots in spots.items():
        in_frame = ET.SubElement(
            all_spots, "SpotsInFrame", {"frame": str(frame)}
        )
        for spot_id, x in frame_spots:
            ET.SubElement(
                in_frame,
                "Spot",
                {
                    "ID": str(spot_id),
                    "name": f"spot{spot_id}",
                    "POSITION_X": str(x),
                    "POSITION_Y": "0",
                    "POSITION_Z": "0",
                },
            )
    all_tracks = ET.SubElement(model, "AllTracks")
    track = ET.SubElement(
        all_tracks, "Track", {"TRACK_ID": "0", "name": "track0"}
    )
    ET.SubElement(
        track, "Edge", {"SPOT_SOURCE_ID": "1", "SPOT_TARGET_ID": "3"}
    )
    ET.SubElement(model, "FilteredTracks")
    ET.ElementTree(root).write(path)


def test_read_mamut_keeps_spots_without_track(tmp_path):
    """Spots that belong to no track used to be dropped."""
    path = tmp_path / "mamut.xml"
    _mamut_xml(path)
    lT = read_from_mamut_xml(str(path))
    assert lT.nodes == {1, 2, 3}
    assert lT.roots == {1, 2}
    assert lT.successor[1] == (3,)
    assert lT.label[2] == "spot2"


def _write_astec(path, with_barycenters):
    data = {
        "cell_lineage": {10002: [20002, 20003], 20002: [30002]},
        "cell_volume": {10002: 10.0, 20002: 5.0, 20003: 5.0, 30002: 6.0},
        "cell_name": {10002: "a1.0001*", 20002: "a2.0001*"},
    }
    if with_barycenters:
        data["cell_barycenter"] = {
            n: np.array([n, 0.0, 0.0]) for n in data["cell_volume"]
        }
    with open(path, "wb") as f:
        pickle.dump(data, f)


@pytest.mark.parametrize("with_barycenters", [True, False])
def test_read_astec_pkl(tmp_path, with_barycenters):
    """Files without barycenters used to raise `NameError`."""
    path = tmp_path / "astec.pkl"
    _write_astec(path, with_barycenters)
    lT = read_from_ASTEC(str(path))
    assert len(lT.nodes) == 4
    assert sorted(lT.time.values()) == [1, 2, 2, 3]
    assert sorted(lT.image_label.values()) == [2, 2, 2, 3]
    assert sorted(lT.volume.values()) == [5.0, 5.0, 6.0, 10.0]
    (root,) = lT.roots
    assert lT.label[root] == "a1.0001*"
    assert len(lT.successor[root]) == 2
    assert bool(lT.pos) is with_barycenters


# ---------------------------------------------------------------- writers


@pytest.fixture
def small_lT():
    return read_from_mastodon("tests/data/test.mastodon")


def test_write_to_svg_draws_every_node_and_edge(small_lT, tmp_path):
    """Only the roots used to be drawn."""
    path = tmp_path / "tree.svg"
    small_lT.write_to_svg(str(path))
    svg = path.read_text()
    assert svg.count("<circle") == len(small_lT.nodes)
    assert svg.count("<line") == len(small_lT.edges)


def test_write_to_svg_with_property_names(small_lT, tmp_path):
    """Property names used to be looked up in the successor dictionary."""
    small_lT.volume = {n: float(n) for n in small_lT.nodes}
    path = tmp_path / "tree.svg"
    small_lT.write_to_svg(
        str(path),
        node_size="volume",
        node_color="volume",
        order_key=lambda n: n,
    )
    assert path.read_text().count("<circle") == len(small_lT.nodes)


def test_write_to_svg_deep_tree(tmp_path):
    """The layout used to recurse once per time point."""
    lT = LineageTree(successor={i: [i + 1] for i in range(3000)})
    path = tmp_path / "deep.svg"
    lT.write_to_svg(str(path))
    assert path.read_text().count("<circle") == 3001


@pytest.fixture
def spatial_lT():
    lT = LineageTree(
        successor={0: [2], 1: [3], 2: [], 3: []},
        pos={0: [0, 0, 0], 1: [1, 0, 0], 2: [0, 0, 0], 3: [1, 0, 0]},
    )
    return lT


def _tlp_edges(path):
    edges = set()
    for line in path.read_text().splitlines():
        if line.startswith("(edge "):
            _, _, a, b = line.strip("()").split()
            edges.add(tuple(sorted((int(a), int(b)))))
    return edges


@pytest.mark.parametrize(
    ("spatial", "compute"),
    [
        ("GG", lambda lT: lT.gabriel_graph()),
        ("kn", lambda lT: lT.k_nearest_neighbours(k=1)),
        ("ball", lambda lT: lT.spatial_edges(th=2)),
    ],
)
def test_write_to_tlp_spatial_edges(spatial_lT, tmp_path, spatial, compute):
    """Spatial edges used to crash the writer."""
    compute(spatial_lT)
    path = tmp_path / "tree.tlp"
    spatial_lT.write_to_tlp(str(path), spatial=spatial)
    assert _tlp_edges(path) == {(0, 2), (1, 3), (0, 1), (2, 3)}


def test_write_to_tlp_spatial_edges_of_selected_nodes(spatial_lT, tmp_path):
    spatial_lT.gabriel_graph()
    path = tmp_path / "tree.tlp"
    spatial_lT.write_to_tlp(str(path), nodes_to_use=[0, 1], spatial="GG")
    assert _tlp_edges(path) == {(0, 1)}


def test_write_to_tlp_needs_the_spatial_graph(spatial_lT, tmp_path):
    with pytest.raises(ValueError, match="gabriel_graph"):
        spatial_lT.write_to_tlp(str(tmp_path / "t.tlp"), spatial="GG")
    with pytest.raises(ValueError, match="one of"):
        spatial_lT.write_to_tlp(str(tmp_path / "t.tlp"), spatial="delaunay")
