"""Spatial neighbourhoods honour `spatial_resolution`."""

from itertools import combinations

import numpy as np
import pytest

from lineagetree import LineageTree


def _tree(resolution, seed=0, n=30):
    """`n` isolated nodes at time 0 and 2 at time 1, at random positions."""
    rng = np.random.default_rng(seed)
    pos = {i: rng.uniform(0, 100, 3) for i in range(n + 2)}
    time = {i: 0 for i in range(n)} | {n: 1, n + 1: 1}
    return LineageTree(
        successor={i: [] for i in range(n + 2)},
        time=time,
        pos=pos,
        spatial_resolution=resolution,
    )


def _brute_force_knn(lT, node, k):
    t = lT.time[node]
    scaled = {
        n: lT.pos[n] * lT.spatial_resolution
        for n in lT.time_nodes[t]
        if n != node
    }
    own = lT.pos[node] * lT.spatial_resolution
    distances = {n: np.linalg.norm(p - own) for n, p in scaled.items()}
    closest = sorted(distances, key=distances.get)[:k]
    return closest, [distances[n] for n in closest]


@pytest.mark.parametrize("resolution", [(1, 1, 1), (1, 1, 5), (0.5, 2, 3)])
def test_k_nearest_neighbours_match_brute_force(resolution):
    """The query used unscaled positions against a scaled KDTree."""
    lT = _tree(resolution)
    neighbours, distances = lT.k_nearest_neighbours(k=4)
    for node in lT.time_nodes[0]:
        expected, expected_distances = _brute_force_knn(lT, node, 4)
        assert list(neighbours[node]) == expected
        np.testing.assert_allclose(distances[node], expected_distances)


def test_k_nearest_neighbours_with_few_nodes():
    lT = _tree((1, 1, 1))
    neighbours, _ = lT.k_nearest_neighbours(k=4)
    assert list(neighbours[30]) == [31]
    assert list(neighbours[31]) == [30]


def _brute_force_gabriel(lT, t):
    nodes = sorted(lT.time_nodes[t])
    pos = {n: lT.pos[n] * lT.spatial_resolution for n in nodes}
    graph = {n: set() for n in nodes}
    for a, b in combinations(nodes, 2):
        centre = (pos[a] + pos[b]) / 2
        radius = np.linalg.norm(pos[a] - pos[b]) / 2
        if all(
            np.linalg.norm(pos[c] - centre) >= radius
            for c in nodes
            if c not in (a, b)
        ):
            graph[a].add(b)
            graph[b].add(a)
    return graph


@pytest.mark.parametrize("resolution", [(1, 1, 1), (1, 1, 10)])
def test_gabriel_graph_uses_spatial_resolution(resolution):
    lT = _tree(resolution, seed=1)
    expected = _brute_force_gabriel(lT, 0)
    graph = lT.gabriel_graph(0)
    assert {n: graph[n] for n in expected} == expected


def test_anisotropy_changes_the_gabriel_graph():
    """Stretching z must change the graph, or the test above proves little."""
    flat = _tree((1, 1, 1), seed=1).gabriel_graph(0)
    stretched = _tree((1, 1, 10), seed=1).gabriel_graph(0)
    assert flat != stretched


def test_neighbours_in_radius_use_physical_distances():
    lT = LineageTree(
        successor={0: [], 1: [], 2: []},
        time={0: 0, 1: 0, 2: 1},
        pos={0: [0, 0, 0], 1: [0, 0, 1], 2: [0, 0, 0]},
        spatial_resolution=(1, 1, 10),
    )
    assert lT.neighbours_in_radius(0, 1, th=5) == {0: set(), 1: set()}
    assert lT.neighbours_in_radius(0, 1, th=11) == {0: {1}, 1: {0}}
