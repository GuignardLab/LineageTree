"""Plotting functions that depend on optional packages."""

import sys

import matplotlib
import matplotlib.pyplot as plt
import pytest

from lineagetree import LineageTree

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def two_chains():
    successor = {0: [1], 1: [2], 2: [], 10: [11], 11: [12], 12: []}
    pos = {n: [n % 10, n // 10, float(n)] for n in successor}
    return LineageTree(successor=successor, pos=pos)


def test_dtw_trajectory_pca_without_scikit_learn(two_chains, monkeypatch):
    """A missing scikit-learn used to surface as `NameError: PCA`."""
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.decomposition", None)
    with pytest.raises(ImportError, match="scikit-learn"):
        two_chains.plot_dtw_trajectory(0, 10, projection="pca")


@pytest.mark.parametrize("projection", [None, "3d", "xz", "yz"])
def test_dtw_trajectory_projections(two_chains, projection):
    distance, fig = two_chains.plot_dtw_trajectory(
        0, 10, projection=projection
    )
    assert distance >= 0
    assert fig.axes


def test_dtw_trajectory_unknown_projection(two_chains):
    with pytest.raises(ValueError, match="available projections"):
        two_chains.plot_dtw_trajectory(0, 10, projection="xyz")
