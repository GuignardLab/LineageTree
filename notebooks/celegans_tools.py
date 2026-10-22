"""Helpers of the C. elegans notebooks of lineagetree.

``lineagetree_showcase.ipynb`` and ``lineage_tree_examples.ipynb`` import
their functions from this module, so that the notebooks show the analyses
rather than the code behind them. The functions are grouped by theme:

- datasets: the registry ``data/datasets.json`` and ``fetch``, which
  downloads a dataset on first use and checks it, as napari-relax does for
  its demo datasets;
- cell names: founders, generations and sublineages, from the names of the
  Sulston nomenclature;
- timing: cell cycles and divisions;
- space: body axes, depth below the surface, contacts and mirror symmetry;
- reporters: z-scores, fate calls and the profiles of the knockdowns;
- tree distances: a UTED style that adds reporter values to the cycle
  lengths, and its uses;
- comparison of two embryos: ``LineageComparison`` and
  ``ComparisonFigure``, which split the UTED between two lineages over
  their subtrees.
"""

from __future__ import annotations

import hashlib
import json
import lzma
import shutil
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

from lineagetree import LineageTree
from lineagetree.tree_approximation import simple_tree

# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"


def load_registry(path=DATA_DIR / "datasets.json"):
    """Read the dataset registry.

    Parameters
    ----------
    path : pathlib.Path, default=DATA_DIR / "datasets.json"
        The registry, a JSON file in the format of the demo datasets of
        napari-relax: each dataset has a ``filename``, a download ``url``
        (None until the file is online), an ``md5`` checksum, a
        ``description`` and a ``size_mb``.

    Returns
    -------
    dict of {str: dict}
        The entries of the registry, by dataset name.
    """
    return json.loads(Path(path).read_text())


def md5sum(path):
    """Compute the MD5 checksum of a file.

    Parameters
    ----------
    path : pathlib.Path
        The file.

    Returns
    -------
    str
        The checksum, in hexadecimal.
    """
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(name, data_dir=DATA_DIR):
    """Return the local path of a dataset, downloading it on first use.

    The file of the dataset is looked for in ``data_dir``. If it is missing,
    or if its MD5 checksum does not match the registry, it is downloaded
    from the URL of the registry and checked, as ``ensure_demo_data`` of
    napari-relax does. A file compressed with xz (``.xz``) is then
    decompressed next to it, once, and the decompressed file is returned.

    Parameters
    ----------
    name : str
        Name of the dataset in the registry, ``data_dir / "datasets.json"``.
    data_dir : pathlib.Path, default=DATA_DIR
        Where the datasets are kept.

    Returns
    -------
    pathlib.Path
        Path to the local file, ready to be loaded.

    Raises
    ------
    KeyError
        If the dataset is not in the registry.
    RuntimeError
        If the file is not available locally and has no URL yet, or if the
        downloaded file does not match its checksum.
    """
    data_dir = Path(data_dir)
    entry = load_registry(data_dir / "datasets.json")[name]
    path = data_dir / entry["filename"]
    target = path.with_suffix("") if path.suffix == ".xz" else path
    if target != path and target.exists():
        return target
    if not (path.exists() and entry["md5"] in (None, md5sum(path))):
        if entry["url"] is None:
            raise RuntimeError(
                f"{entry['filename']} is not in {data_dir} "
                "and is not online yet"
            )
        print(f"downloading {name} ({entry['size_mb']:.0f} MB) ...")
        partial = path.with_name(path.name + ".part")
        urllib.request.urlretrieve(entry["url"], partial)
        if entry["md5"] not in (None, md5sum(partial)):
            partial.unlink()
            raise RuntimeError(
                f"the downloaded file of {name} does not match "
                "its MD5 checksum"
            )
        partial.rename(path)
    if target != path:
        partial = target.with_name(target.name + ".part")
        with lzma.open(path) as source, open(partial, "wb") as sink:
            shutil.copyfileobj(source, sink, 1 << 20)
        partial.rename(target)
    return target


# --------------------------------------------------------------------------
# Cell names
# --------------------------------------------------------------------------

FOUNDERS = {
    "AB": "tab:blue",
    "MS": "tab:orange",
    "E": "tab:green",
    "C": "tab:red",
    "D": "tab:purple",
    "P4": "tab:brown",
}
"""The founder lineages, with their colours in the figures."""

SUBLINEAGES = [
    "ABala",
    "ABalp",
    "ABara",
    "ABarp",
    "ABpla",
    "ABplp",
    "ABpra",
    "ABprp",
    "MSa",
    "MSp",
    "Ea",
    "Ep",
    "Ca",
    "Cp",
    "D",
    "P4",
]
"""The sublineages of the 8-cell to 28-cell stage used to sum up the fates."""

TISSUES = {"PHA": "tab:green", "HYP": "tab:orange", "NEU": "tab:purple"}
"""The tissue reporters (pharynx, hypodermis, neurons), with their colours."""


def founder(name):
    """Return the founder lineage of a cell, from its name.

    Parameters
    ----------
    name : str
        Sulston name of the cell, such as ``"ABala"`` or ``"MSpa"``.

    Returns
    -------
    str
        The founder: one of the keys of ``FOUNDERS``, or ``"early"`` for the
        cells born before the founders (P0, P1, P2, P3 and EMS). The
        germline precursors Z2 and Z3 belong to P4.
    """
    if name in ("P0", "P1", "P2", "P3", "EMS"):
        return "early"
    if name.startswith(("Z2", "Z3")):
        return "P4"
    return next((f for f in FOUNDERS if name.startswith(f)), "early")


def founder_of(lT, node):
    """Return the founder lineage of a node.

    Parameters
    ----------
    lT : LineageTree
        The lineage, with cell names as labels.
    node : int
        A node of the lineage.

    Returns
    -------
    str
        The founder of the cell the node belongs to; see ``founder``.
    """
    return founder(lT.labels[node])


def generation(name):
    """Count the divisions between a cell and its founder.

    Parameters
    ----------
    name : str
        Sulston name of the cell.

    Returns
    -------
    int or None
        0 for a founder cell (AB, MS, E, C, D, P4), 1 for its daughters, and
        so on: ABala is generation 3 of AB, Z2 and Z3 generation 1 of P4.
        None for the cells born before the founders.
    """
    f = founder(name)
    if f not in FOUNDERS:
        return None
    return int(name != "P4") if f == "P4" else len(name) - len(f)


def sublineage(name, sublineages=SUBLINEAGES):
    """Return the sublineage a cell belongs to.

    Parameters
    ----------
    name : str
        Sulston name of the cell.
    sublineages : list of str, default=SUBLINEAGES
        The sublineages, by the names of their first cells.

    Returns
    -------
    str or None
        The element of ``sublineages`` the name starts with (Z2 and Z3
        count as P4), or None for the cells born before them.
    """
    return next(
        (
            s
            for s in sublineages
            if name.startswith(s) or (s == "P4" and name[:2] in ("Z2", "Z3"))
        ),
        None,
    )


def homologue(name):
    """Return the left/right homologue of an ABpl or ABpr cell.

    Parameters
    ----------
    name : str
        Name of a cell of the ABpl or ABpr lineage, e.g. ``"ABplaap"``.

    Returns
    -------
    str
        The name with the l and r of its fourth letter swapped, e.g.
        ``"ABpraap"``.
    """
    return name[:3] + {"l": "r", "r": "l"}[name[3]] + name[4:]


def first_nodes(lT):
    """Map the name of each cell to the first node of its chain.

    Parameters
    ----------
    lT : LineageTree
        The lineage, with cell names as labels.

    Returns
    -------
    dict of {str: int}
        Maps each cell name to the first node of its chain.
    """
    return {lT.labels[chain[0]]: chain[0] for chain in lT.all_chains}


def divides(lT, chain):
    """Tell whether a cell divides during the recording.

    Parameters
    ----------
    lT : LineageTree
        The lineage.
    chain : list of int
        The nodes of the cell, in time order.

    Returns
    -------
    bool
        True if the last node of the chain has two successors.
    """
    return len(lT.successor[chain[-1]]) == 2


def daughters(lT, chain):
    """Return the daughters of a cell.

    Parameters
    ----------
    lT : LineageTree
        The lineage, with cell names as labels.
    chain : list of int
        The nodes of the mother cell, in time order.

    Returns
    -------
    dict of {str: list of int}
        Maps the name of each daughter to its chain; empty if the cell does
        not divide.
    """
    return {
        lT.labels[d]: lT.get_chain_of_node(d) for d in lT.successor[chain[-1]]
    }


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------


def cycle_lengths(lT):
    """Measure the complete cell cycles of an embryo.

    Parameters
    ----------
    lT : LineageTree
        The lineage of the embryo, with cell names as labels.

    Returns
    -------
    dict of {str: int}
        Maps the name of every cell that is born and divides during the
        recording to the number of time points of its cycle.
    """
    return {
        lT.labels[c[0]]: len(c)
        for c in lT.all_chains
        if lT.predecessor[c[0]] and len(lT.successor[c[-1]]) == 2
    }


def division_cycles(lT):
    """Measure the cycle of every cell that divides.

    Unlike ``cycle_lengths``, the cells born before the recording are
    included.

    Parameters
    ----------
    lT : LineageTree
        The lineage of an embryo, with cell names as labels.

    Returns
    -------
    dict of {str: int}
        Maps the name of each cell that divides to the number of frames of
        its chain.
    """
    return {
        lT.labels[chain[0]]: len(chain)
        for chain in lT.all_chains
        if len(lT.successor[chain[-1]]) >= 2
    }


def tracked_until(lT, root):
    """Return the last frame at which the whole subtree of a node is tracked.

    Parameters
    ----------
    lT : LineageTree
        The lineage of an embryo.
    root : int
        The node spawning the subtree.

    Returns
    -------
    int
        The earliest time at which a branch of the subtree ends: up to this
        frame, every cell of the subtree is followed.
    """
    return min(
        lT.time[chain[-1]]
        for chain in lT.get_all_chains_of_subtree(root)
        if not lT.successor[chain[-1]]
    )


def typical_cycles(cycles_by_embryo, min_embryos=10):
    """Return the typical length of each cell cycle in a set of embryos.

    Parameters
    ----------
    cycles_by_embryo : dict of {str: dict of {str: int}}
        The cycle lengths of each embryo, from ``cycle_lengths``.
    min_embryos : int, default=10
        Minimum number of embryos in which a cycle must be measured.

    Returns
    -------
    dict of {str: float}
        Maps each cell measured in at least ``min_embryos`` embryos to the
        median of its cycle lengths.
    """
    measured = defaultdict(list)
    for cycles in cycles_by_embryo.values():
        for name, length in cycles.items():
            measured[name].append(length)
    return {
        name: np.median(lengths)
        for name, lengths in measured.items()
        if len(lengths) >= min_embryos
    }


def pace(cycles, typical, prefix=""):
    """Estimate the pace of an embryo from its cell cycles.

    Parameters
    ----------
    cycles : dict of {str: float}
        The cycle lengths of the embryo, by cell name.
    typical : dict of {str: float}
        The typical cycle lengths, from ``typical_cycles``.
    prefix : str, default=""
        Only the cells whose name starts with it are used, e.g. ``"AB"``.

    Returns
    -------
    float
        The median ratio of the cycles of the embryo to the typical ones:
        above 1 for an embryo slower than usual.
    """
    return np.median(
        [
            cycles[c] / typical[c]
            for c in cycles
            if c in typical and c.startswith(prefix)
        ]
    )


def corrected_cycles(lT, typical):
    """Measure the cell cycles of an embryo, corrected for its pace.

    Parameters
    ----------
    lT : LineageTree
        The lineage of the embryo.
    typical : dict of {str: float}
        The typical cycle lengths, from ``typical_cycles``.

    Returns
    -------
    dict of {str: float}
        Maps the name of each complete cell cycle to its length divided by
        the pace of the embryo, estimated on its AB cells.
    """
    cycles = cycle_lengths(lT)
    ratio = pace(cycles, typical, prefix="AB")
    return {c: v / ratio for c, v in cycles.items()}


def cycle_variation(cycles_by_embryo, cells, paces=None):
    """Measure how much the cycle of each cell varies between embryos.

    Parameters
    ----------
    cycles_by_embryo : dict of {str: dict of {str: int}}
        The cycle lengths of each embryo, from ``cycle_lengths``.
    cells : list of str
        Names of the cells.
    paces : dict of {str: float}, optional
        The pace of each embryo: its cycle lengths are divided by it first.

    Returns
    -------
    dict of {str: float}
        Maps each cell to the coefficient of variation (standard deviation
        over mean) of its cycle length across the embryos.
    """
    out = {}
    for c in cells:
        v = np.array(
            [
                cycles[c] / (paces[e] if paces else 1)
                for e, cycles in cycles_by_embryo.items()
                if c in cycles
            ]
        )
        out[c] = v.std(ddof=1) / v.mean()
    return out


def division_time(lT, prefix, gen):
    """Return the mean division time of a sublineage at one generation.

    Parameters
    ----------
    lT : LineageTree
        The lineage of an embryo.
    prefix : str
        The sublineage, e.g. ``"ABpl"``.
    gen : int
        The generation of the AB cells.

    Returns
    -------
    float
        The mean time of division of the cells of the sublineage at this
        generation, NaN unless all of them divide during the recording.
    """
    times = [
        lT.time[c[-1]]
        for c in lT.all_chains
        if lT.labels[c[0]].startswith(prefix)
        and len(lT.labels[c[0]]) == 2 + gen
        and len(lT.successor[c[-1]]) == 2
    ]
    return np.mean(times) if len(times) == 2 ** (gen - 2) else np.nan


def ab_division_times(lT):
    """Return when each AB cell of an embryo divides.

    Parameters
    ----------
    lT : LineageTree
        The lineage of an embryo.

    Returns
    -------
    dict of {str: int}
        Maps the name of each AB cell that divides during the recording to
        the time point of its division.
    """
    return {
        lT.labels[c[0]]: lT.time[c[-1]]
        for c in lT.all_chains
        if lT.labels[c[0]].startswith("AB") and len(lT.successor[c[-1]]) == 2
    }


def wild_type_cycles(manager, wild_types):
    """Describe the cycle of each cell in the wild types.

    Parameters
    ----------
    manager : LineageTreeManager
        The manager holding the embryos.
    wild_types : list of str
        Names of the wild-type embryos.

    Returns
    -------
    dict of {str: tuple of (float, float)}
        Maps each cell measured in at least three wild types to the mean and
        the standard deviation of its cycle length, in frames.
    """
    all_cycles = [division_cycles(manager[w]) for w in wild_types]
    reference = {}
    for c in set().union(*all_cycles):
        values = [wc[c] for wc in all_cycles if c in wc]
        if len(values) >= 3:
            reference[c] = (np.mean(values), np.std(values, ddof=1))
    return reference


def cycle_table(lT, cell_names, reference, speed=1):
    """Print the cycles of some cells of an embryo against the wild type.

    Cycles more than 4 standard deviations away from the wild type are
    flagged.

    Parameters
    ----------
    lT : LineageTree
        The lineage of the embryo.
    cell_names : list of str
        The cells to print.
    reference : dict of {str: tuple of (float, float)}
        Mean and standard deviation of the wild-type cycles, from
        ``wild_type_cycles``.
    speed : float, default=1
        Pace of the embryo relative to the wild types: its cycles are
        divided by it.
    """
    observed, born = division_cycles(lT), first_nodes(lT)
    print(f"{'cell':6} {'born':>5} {'wild type':>11} {lT.name:>20}")
    for c in cell_names:
        mean, sd = reference[c]
        value = observed[c] / speed
        z = (value - mean) / max(sd, 1)
        print(
            f"{c:6} {lT.time[born[c]]:5} {mean:6.1f} ± {sd:<3.1f} "
            f"{value:12.1f}" + (f"   {z:+.0f} SD" if abs(z) > 4 else "")
        )


# --------------------------------------------------------------------------
# Space
# --------------------------------------------------------------------------


def cell_spacing(lT):
    """Measure the typical distance between neighbouring cells over time.

    Parameters
    ----------
    lT : LineageTree
        The lineage, with positions.

    Returns
    -------
    dict of {int: float}
        For each time point with at least two cells, the median over its
        cells of the distance to the closest other cell. Distances divided
        by it are in "cell spacings", which compensates for the cells
        getting smaller.
    """
    _, nearest = lT.k_nearest_neighbours(k=1)
    return {
        t: np.median([nearest[n][0] for n in nodes])
        for t, nodes in lT.time_nodes.items()
        if len(nodes) > 1
    }


def body_axes(lT, t):
    """Return the centre and the body axes of the embryo at a time point.

    Parameters
    ----------
    lT : LineageTree
        The lineage, with positions and cell names as labels.
    t : int
        The time point. The ABa, ABpl and ABpr lineages must exist, i.e.
        ``t`` must be after the 8-cell stage.

    Returns
    -------
    numpy.ndarray
        The centre of the cells of the time point.
    numpy.ndarray
        Unit anterior-posterior vector: the long axis of the embryo (its
        first principal axis), pointing to the ABa side.
    numpy.ndarray
        Unit left-right vector: from the centre of the ABpl cells to the
        centre of the ABpr cells, made orthogonal to the first axis.
    numpy.ndarray
        Unit dorsal-ventral vector, orthogonal to the other two.
    """
    nodes_t = list(lT.time_nodes[t])
    xyz = np.array([lT.pos[n] for n in nodes_t])
    centre = xyz.mean(axis=0)

    def centroid(prefix):
        """Return the centre of the cells whose name starts with ``prefix``.

        Parameters
        ----------
        prefix : str
            Beginning of the names, e.g. ``"ABpl"``.

        Returns
        -------
        numpy.ndarray
            The mean position of these cells.
        """
        return np.mean(
            [lT.pos[n] for n in nodes_t if lT.labels[n].startswith(prefix)],
            axis=0,
        )

    ap = np.linalg.svd(xyz - centre)[2][0]
    if (centroid("ABa") - centre) @ ap < 0:
        ap = -ap
    lr = centroid("ABpr") - centroid("ABpl")
    lr -= (lr @ ap) * ap
    lr /= np.linalg.norm(lr)
    return centre, ap, lr, np.cross(ap, lr)


def position_after(lT, chain, dt=2):
    """Return the position of a cell a few time points after its birth.

    Parameters
    ----------
    lT : LineageTree
        The lineage, with positions.
    chain : list of int
        The nodes of the cell, in time order.
    dt : int, default=2
        Number of time points after the birth; the last node of the chain is
        used if the cell lives shorter than that.

    Returns
    -------
    numpy.ndarray
        The position of the node.
    """
    return np.asarray(lT.pos[chain[min(dt, len(chain) - 1)]])


def surface_depth(lT, spacing):
    """Measure how deep each cell is below the surface of the embryo.

    The surface of a time point is the convex hull of its cells.

    Parameters
    ----------
    lT : LineageTree
        The lineage, with positions.
    spacing : dict of {int: float}
        The spacing between cells at each time point, from
        ``cell_spacing``.

    Returns
    -------
    dict of {int: float}
        Maps each node of the time points with at least 5 cells to its
        distance to the closest face of the hull, in cell spacings: 0 at the
        surface, positive inside.
    """
    depth = {}
    for t, nodes_t in lT.time_nodes.items():
        nodes_t = list(nodes_t)
        if len(nodes_t) < 5:
            continue
        xyz = np.array([lT.pos[n] for n in nodes_t])
        hull = ConvexHull(xyz)
        inside = -(xyz @ hull.equations[:, :3].T + hull.equations[:, 3]).max(
            axis=1
        )
        depth.update(zip(nodes_t, inside / spacing[t], strict=True))
    return depth


def stage_times(lT):
    """Group the time points by their number of cells.

    Parameters
    ----------
    lT : LineageTree
        The lineage.

    Returns
    -------
    dict of {int: list of int}
        Maps each number of cells to the time points, sorted, with that
        number of cells: the 4-cell stage is ``stage_times(lT)[4]``.
    """
    stages = defaultdict(list)
    for t in sorted(lT.time_nodes):
        stages[len(lT.time_nodes[t])].append(t)
    return stages


def contact_frequency(lT, gabriel, times):
    """Measure how long each pair of cells is in contact.

    Parameters
    ----------
    lT : LineageTree
        The lineage, with cell names as labels.
    gabriel : dict of {int: set of int}
        The Gabriel graph, from ``lT.gabriel_graph``.
    times : list of int
        The time points, e.g. those of a stage.

    Returns
    -------
    dict of {tuple of (str, str): float}
        Maps each pair of cell names, sorted, to the fraction of the time
        points during which the two cells are neighbours in the Gabriel
        graph.
    """
    count = defaultdict(int)
    for t in times:
        pairs = {
            tuple(sorted((lT.labels[a], lT.labels[b])))
            for a in lT.time_nodes[t]
            for b in gabriel.get(a, ())
        }
        for pair in pairs:
            count[pair] += 1
    return {pair: c / len(times) for pair, c in count.items()}


def stage_layout(lT, times):
    """Place the cells of a stage in 2D, for drawing.

    Each cell is put at its mean position during the stage, projected on the
    two main axes of these positions, with ABa on the left.

    Parameters
    ----------
    lT : LineageTree
        The lineage, with positions and cell names as labels.
    times : list of int
        The time points of the stage; it must contain ABa.

    Returns
    -------
    dict of {str: numpy.ndarray}
        Maps the name of each cell to its 2D position.
    """
    position = defaultdict(list)
    for t in times:
        for n in lT.time_nodes[t]:
            position[lT.labels[n]].append(lT.pos[n])
    names = sorted(position)
    xyz = np.array([np.mean(position[name], axis=0) for name in names])
    xy = (xyz - xyz.mean(axis=0)) @ np.linalg.svd(xyz - xyz.mean(axis=0))[2][
        :2
    ].T
    if xy[names.index("ABa"), 0] > 0:
        xy[:, 0] *= -1
    return dict(zip(names, xy, strict=True))


def reflection(lT, node, axes):
    """Reflect a node through the sagittal plane of its time point.

    The plane goes through the centre of the embryo and is perpendicular to
    the left-right axis of ``body_axes``, so it contains the long axis.

    Parameters
    ----------
    lT : LineageTree
        The lineage, with positions.
    node : int
        A node.
    axes : dict of {int: tuple}
        The body axes of each time point, from ``body_axes``.

    Returns
    -------
    numpy.ndarray
        The reflected position.
    """
    centre, _, lr, _ = axes[lT.time[node]]
    p = np.asarray(lT.pos[node])
    return p - 2 * ((p - centre) @ lr) * lr


def side_view(p, axes):
    """Project a position on the side view of the embryo.

    Parameters
    ----------
    p : numpy.ndarray
        A 3D position.
    axes : tuple of numpy.ndarray
        The centre and the body axes of the time point, from ``body_axes``.

    Returns
    -------
    tuple of float
        The coordinates along the anterior-posterior axis (anterior to the
        left) and the dorsal-ventral axis, relative to the centre.
    """
    centre, ap, _, dv = axes
    return -(p - centre) @ ap, (p - centre) @ dv


def mirror(points, normal, offset):
    """Reflect points through a plane.

    Parameters
    ----------
    points : numpy.ndarray
        Array of shape (n, 3).
    normal : numpy.ndarray
        Unit normal vector of the plane.
    offset : float
        Position of the plane: it contains the points ``x`` with
        ``x @ normal == offset``.

    Returns
    -------
    numpy.ndarray
        The reflected points, of shape (n, 3).
    """
    return points - 2 * ((points @ normal) - offset)[:, None] * normal


def best_mirror_plane(left, right, start):
    """Find the mirror plane that best maps right cells onto left ones.

    The plane minimises the sum of the squared distances between each left
    cell and the reflection of its right homologue. For a given normal, the
    best offset is the mean of ``(left + right) @ normal / 2``; the normal
    is optimised with the Nelder-Mead method from ``start`` and from 20
    random directions, and the best result is kept.

    Parameters
    ----------
    left : numpy.ndarray
        Positions of the left cells, of shape (n, 3).
    right : numpy.ndarray
        Positions of their right homologues, in the same order.
    start : numpy.ndarray
        A first guess for the normal of the plane.

    Returns
    -------
    numpy.ndarray
        Unit normal of the best plane.
    float
        Offset of the best plane.
    """

    def cost(v):
        """Measure how well a plane maps the right cells onto the left ones.

        Parameters
        ----------
        v : numpy.ndarray
            Normal of the plane, of any length.

        Returns
        -------
        float
            The sum of the squared distances between the left cells and the
            reflections of their homologues, through the plane of normal
            ``v`` and its best offset.
        """
        normal = v / np.linalg.norm(v)
        offset = np.mean((left + right) @ normal) / 2
        return np.sum((left - mirror(right, normal, offset)) ** 2)

    starts = [start, *np.random.default_rng(0).normal(size=(20, 3))]
    best = min(
        (minimize(cost, x0, method="Nelder-Mead") for x0 in starts),
        key=lambda r: r.fun,
    )
    normal = best.x / np.linalg.norm(best.x)
    return normal, np.mean((left + right) @ normal) / 2


def mirrored_position(lT, node, planes, axes):
    """Reflect a node through the best mirror plane of its time point.

    Before the first fitted plane, the sagittal plane is used.

    Parameters
    ----------
    lT : LineageTree
        The lineage, with positions.
    node : int
        A node.
    planes : dict of {int: tuple of (numpy.ndarray, float)}
        The normal and offset of the best plane of each time point, from
        ``best_mirror_plane``.
    axes : dict of {int: tuple}
        The body axes of each time point, from ``body_axes``.

    Returns
    -------
    numpy.ndarray
        The reflected position.
    """
    if lT.time[node] not in planes:
        return reflection(lT, node, axes)
    normal, offset = planes[lT.time[node]]
    return mirror(np.asarray(lT.pos[node])[None], normal, offset)[0]


def ancestry(lT, node):
    """List the ancestors of a cell, from the root.

    Parameters
    ----------
    lT : LineageTree
        The lineage.
    node : int
        Any node of the cell.

    Returns
    -------
    list of int
        The first nodes of the cells from the root of the tree to the cell
        of ``node``, included.
    """
    cells = [lT.get_chain_of_node(node)[0]]
    while lT.predecessor[cells[-1]]:
        cells.append(lT.get_chain_of_node(lT.predecessor[cells[-1]][0])[0])
    return cells[::-1]


def lineage_distance(ancestry_1, ancestry_2):
    """Count the divisions back to the last common ancestor of two cells.

    Parameters
    ----------
    ancestry_1, ancestry_2 : list of int
        The ancestries of the two cells, from ``ancestry``.

    Returns
    -------
    int
        The number of divisions between the last common ancestor and the
        deeper of the two cells: 1 for sisters, 2 for first cousins, etc.
    """
    common = sum(
        1 for x, y in zip(ancestry_1, ancestry_2, strict=False) if x == y
    )
    return max(len(ancestry_1), len(ancestry_2)) - common


# --------------------------------------------------------------------------
# Reporters
# --------------------------------------------------------------------------


def mean_per_cell(lT, values):
    """Average a node property over the life of each cell.

    Parameters
    ----------
    lT : LineageTree
        The lineage.
    values : dict of {int: float}
        The property, NaN where it was not measured.

    Returns
    -------
    dict of {int: float}
        Maps the first node of each chain to the mean of the property over
        the nodes of the chain, NaN if none of them was measured.
    """
    out = {}
    for chain in lT.all_chains:
        measured = [
            values[n] for n in chain if n in values and not np.isnan(values[n])
        ]
        out[chain[0]] = np.mean(measured) if measured else np.nan
    return out


def founder_contrast(lT, values):
    """Measure how much a node property differs between founder lineages.

    Parameters
    ----------
    lT : LineageTree
        The lineage, with cell names as labels.
    values : dict of {int: float}
        The property, NaN where it was not measured.

    Returns
    -------
    float
        The range of the mean values of the founder lineages (those with
        more than 10 measured cells), in units of the standard deviation of
        the cell values; 0 if fewer than three founders were measured.
    """
    per_cell = mean_per_cell(lT, values)
    by_founder = {}
    for c, v in per_cell.items():
        if not np.isnan(v):
            by_founder.setdefault(founder_of(lT, c), []).append(v)
    means = [
        np.mean(v)
        for f, v in by_founder.items()
        if f in FOUNDERS and len(v) > 10
    ]
    return (
        np.ptp(means) / (np.nanstd(list(per_cell.values())) or 1)
        if len(means) > 2
        else 0
    )


def zscore(values, nodes):
    """Z-score a node property over a set of nodes.

    Parameters
    ----------
    values : dict of {int: float}
        The property, NaN where it was not measured.
    nodes : list of int
        The nodes, e.g. those of a time window.

    Returns
    -------
    dict of {int: float}
        Maps each measured node of ``nodes`` to its value, minus the mean and
        divided by the standard deviation of the measured values of
        ``nodes``.
    """
    x = np.array([values.get(n, np.nan) for n in nodes])
    mean, std = np.nanmean(x), np.nanstd(x)
    return {
        n: (values[n] - mean) / std
        for n in nodes
        if not np.isnan(values.get(n, np.nan))
    }


def median_score(zscored, nodes, conditions):
    """Score some nodes for a set of conditions.

    Parameters
    ----------
    zscored : dict of {str: dict of {int: float}}
        The z-scores of each condition, from ``zscore``.
    nodes : list of int
        The nodes, usually those of one cell in a time window.
    conditions : list of str
        The conditions, usually those of one reporter.

    Returns
    -------
    float
        The median, over the conditions that measured at least one of the
        nodes, of their mean z-score on these nodes; NaN if none did.
    """
    means = [
        np.mean(z)
        for c in conditions
        if (z := [zscored[c][n] for n in nodes if n in zscored[c]])
    ]
    return np.median(means) if means else np.nan


def call_fate(scores, threshold=0.5):
    """Call the fate of a cell from its reporter scores.

    Parameters
    ----------
    scores : dict of {str: float}
        The score of the cell for each tissue, NaN where unknown.
    threshold : float, default=0.5
        Minimum score for a fate to be called.

    Returns
    -------
    str
        The tissue with the highest score, if that score is above
        ``threshold``; ``"none"`` otherwise.
    """
    known = {t: s for t, s in scores.items() if not np.isnan(s)}
    best = max(known, key=known.get, default=None)
    return best if best and known[best] > threshold else "none"


def same_fate_fraction(fates, edges):
    """Return the fraction of contacts between cells of the same fate.

    Parameters
    ----------
    fates : dict of {int: str}
        Maps nodes to their fates.
    edges : set of tuple of (int, int)
        Pairs of nodes in contact.

    Returns
    -------
    float
        The fraction of the pairs whose two nodes have the same fate.
    """
    return np.mean([fates[a] == fates[b] for a, b in edges])


def sublineage_profile(zscored, sublineage_of, sublineages):
    """Average the z-scores of a condition over each sublineage.

    Parameters
    ----------
    zscored : dict of {int: float}
        The z-scores of the condition, by node, from ``zscore``.
    sublineage_of : dict of {int: str}
        The sublineage of each node.
    sublineages : list of str
        The sublineages to report, in order.

    Returns
    -------
    numpy.ndarray
        For each sublineage, the mean z-score of its measured nodes; NaN
        where none was measured.
    """
    per = defaultdict(list)
    for n, z in zscored.items():
        per[sublineage_of[n]].append(z)
    return np.array(
        [np.mean(per[s]) if per[s] else np.nan for s in sublineages]
    )


def report_change(name, change, sublineages, threshold=0.5):
    """Print the sublineages where a knockdown changes a reporter the most.

    Parameters
    ----------
    name : str
        Name of the knockdown, e.g. ``"GSK-3"``.
    change : numpy.ndarray
        Its change in each sublineage, against the typical profile.
    sublineages : list of str
        The sublineages, in the order of ``change``.
    threshold : float, default=0.5
        Smallest change printed.
    """
    lost = [
        f"{sublineages[i]} {change[i]:+.1f}"
        for i in np.argsort(change)[:2]
        if change[i] < -threshold
    ]
    gained = [
        f"{sublineages[i]} {change[i]:+.1f}"
        for i in np.argsort(change)[::-1][:3]
        if change[i] > threshold
    ]
    print(
        f"{name:7} lost in {', '.join(lost) or '-':24} "
        f"gained in {', '.join(gained) or '-'}"
    )


def reporter_profile(lT, sublineages, last=30):
    """Average the late reporter values of an embryo over each sublineage.

    Parameters
    ----------
    lT : LineageTree
        The lineage of the embryo, with its reporter in ``lT.expression``.
    sublineages : list of str
        The sublineages to report, in order.
    last : int, default=30
        Number of time points, at the end of the recording, to average
        over.

    Returns
    -------
    numpy.ndarray
        For each sublineage, the mean of the reporter values of its nodes in
        these time points, z-scored over all the nodes of these time points;
        NaN for a sublineage that was not followed.
    """
    nodes_late = [n for n in lT.nodes if lT.time[n] > lT.t_e - last]
    values = np.array([lT.expression[n] for n in nodes_late], dtype=float)
    z = (values - values.mean()) / values.std()
    per = defaultdict(list)
    for n, v in zip(nodes_late, z, strict=True):
        per[sublineage(lT.labels[n])].append(v)
    return np.array(
        [np.mean(per[s]) if per[s] else np.nan for s in sublineages]
    )


# --------------------------------------------------------------------------
# Tree distances
# --------------------------------------------------------------------------


class SimpleWithReporters(simple_tree):
    """The `simple` tree style, with reporter values added to the costs.

    As in `simple_tree`, each cell (chain) of a lineage becomes one node,
    weighted by its length. Each node also carries the mean, over the nodes
    of the cell, of the property ``property_name`` of the lineage tree. For
    two cells of lengths ``l1`` and ``l2`` and mean values ``v1`` and ``v2``,
    the costs are:

    - match: ``|l1 - l2| + weight * min(l1, l2) * sum(|v1 - v2|)``;
    - insertion or deletion: ``l1 * (1 + weight * sum(|v1|))``.

    Values that are missing cost nothing. The norm of a tree, used to
    normalise the distance, is the cost of deleting all of its cells.

    Attributes
    ----------
    property_name : str
        Name of the property of the lineage tree holding the values: a
        dictionary mapping nodes to a float or to an array (one value per
        reporter). Nodes without a value are ignored.
    weight : float
        Cost of a difference of one unit of value during one time point,
        relative to the cost of one time point of difference in length.
    """

    property_name = "reporters"
    weight = 1.0

    def get_tree(self):
        """Build the tree of `simple_tree`, with the mean value of each cell.

        Returns
        -------
        dict of {int: list of int}
            Adjacency dictionary, keyed by the first node of each chain.
        dict of {int: tuple}
            Maps the first node of each chain to its length (times
            ``time_scale``) and to the mean of its values, an array, or None
            if none of its nodes has a value.
        """
        tree, lengths = super().get_tree()
        values = getattr(self.lT, self.property_name)
        self.times = {}
        for first, length in lengths.items():
            chain = [
                n
                for n in self.lT.get_successors(first)
                if self.lT.time[n] <= self.end_time
            ]
            measured = [np.atleast_1d(values[n]) for n in chain if n in values]
            self.times[first] = (
                length,
                np.mean(measured, axis=0) if measured else None,
            )
        return tree, self.times

    @staticmethod
    def value_cost(v1, v2):
        """Return the difference between two values.

        Parameters
        ----------
        v1, v2 : numpy.ndarray, float or None
            The values; None for a cell without value.

        Returns
        -------
        float
            The sum of the absolute differences, 0 if a value is missing.
        """
        if v1 is None or v2 is None:
            return 0.0
        return float(np.sum(np.abs(np.asarray(v1) - v2)))

    def delta(self, x, y, corres1, corres2, times1, times2):
        """Return the cost of matching two cells, or of adding or removing one.

        Parameters
        ----------
        x : int or None
            Cell of the first tree, in the numbering of edist; None when
            ``y`` is inserted.
        y : int or None
            Cell of the second tree; None when ``x`` is deleted.
        corres1, corres2 : dict
            Map the edist numbering of each tree to the first nodes of its
            cells.
        times1, times2 : dict
            The ``(length, value)`` of each cell of each tree, from
            ``get_tree``.

        Returns
        -------
        float
            The cost of the edit.
        """
        if x is None and y is None:
            return 0
        if x is None or y is None:
            length, value = (
                times2[corres2[y]] if x is None else times1[corres1[x]]
            )
            return length * (1 + self.weight * self.value_cost(value, 0))
        (l1, v1), (l2, v2) = times1[corres1[x]], times2[corres2[y]]
        return abs(l1 - l2) + self.weight * min(l1, l2) * self.value_cost(
            v1, v2
        )

    def get_norm(self, root):
        """Return the cost of deleting the whole tree.

        Parameters
        ----------
        root : int
            The root of the tree (unused: the tree is the one of this object).

        Returns
        -------
        float
            The sum of the deletion costs of all the cells.
        """
        return sum(
            length * (1 + self.weight * self.value_cost(value, 0))
            for length, value in self.times.values()
        )


def sister_divergence(lT, style, window, last_time):
    """Compare the lineages of the two daughters of every division.

    Parameters
    ----------
    lT : LineageTree
        The lineage.
    style : str or type
        The tree style of the comparison, e.g. ``"simple"`` or
        ``SimpleWithReporters``.
    window : int
        Number of time points after the birth of the daughters over which
        their lineages are compared.
    last_time : int
        Only the divisions whose window ends by this time point are scored.

    Returns
    -------
    dict of {int: float}
        Maps the first node of each mother cell to the unordered tree edit
        distance between the lineages of its daughters.
    """
    out = {}
    for chain in lT.all_chains:
        if not divides(lT, chain):
            continue
        first, second = lT.successor[chain[-1]]
        born = lT.time[first]
        if born + window <= last_time:
            out[chain[0]] = float(
                lT.unordered_tree_edit_distance(
                    first, second, end_time=born + window, style=style
                )
            )
    lT.clear_comparisons()
    return out


def homologue_rank(lT, cells, gen, style, end_time):
    """Rank the homologue of each ABpl lineage among the ABpr lineages.

    Parameters
    ----------
    lT : LineageTree
        The lineage, with cell names as labels.
    cells : dict of {str: int}
        The first node of each cell, from ``first_nodes``.
    gen : int
        Generation of the cells compared (4 for ABplaa, ABplap, ...).
    style : str or type
        The tree style of the comparison.
    end_time : int
        Last time point of the comparison.

    Returns
    -------
    list of int
        For each ABpl cell of the generation, the rank of the lineage of its
        homologue among the lineages of all the ABpr cells of the
        generation, sorted by distance to the ABpl lineage (1: the closest).
    """
    lefts = sorted(
        name
        for name in cells
        if name.startswith("ABpl") and generation(name) == gen
    )
    rights = sorted(
        name
        for name in cells
        if name.startswith("ABpr") and generation(name) == gen
    )
    ranks = []
    for name in lefts:
        distance = {
            r: lT.unordered_tree_edit_distance(
                cells[name], cells[r], end_time=end_time, style=style
            )
            for r in rights
        }
        ranks.append(
            sorted(rights, key=distance.get).index(homologue(name)) + 1
        )
        lT.clear_comparisons()
    return ranks


def annotated_copy(lT, nodes, values, name):
    """Copy a lineage with one set of reporter values.

    Parameters
    ----------
    lT : LineageTree
        The lineage to copy.
    nodes : list of int
        The nodes annotated.
    values : numpy.ndarray
        One value per node of ``nodes``, NaN where unknown.
    name : str
        Name of the new tree.

    Returns
    -------
    LineageTree
        A tree with the nodes, successors, times and time resolution of
        ``lT``, whose property ``reporters`` maps the nodes of ``nodes`` to
        their finite values.
    """
    copy = LineageTree(
        successor={n: list(s) for n, s in lT.successor.items()},
        time=dict(lT.time),
        name=name,
    )
    copy.reporters = {
        n: v for n, v in zip(nodes, values, strict=True) if np.isfinite(v)
    }
    copy.time_resolution = lT.time_resolution
    return copy


def cell_by_cell_distance(lT, nodes, reference, other, root, end_time):
    """Compare two annotations of a lineage, with the cells matched by name.

    The costs are those of ``SimpleWithReporters``; the two annotations
    share the cells and their lengths, so only the values count.

    Parameters
    ----------
    lT : LineageTree
        The lineage.
    nodes : list of int
        The annotated nodes.
    reference, other : numpy.ndarray
        Two sets of values, one per node of ``nodes``, NaN where unknown.
    root : int
        The first node of the lineage compared.
    end_time : int
        Last time point of the comparison.

    Returns
    -------
    float
        The cost of matching every cell with itself, divided by the larger
        of the costs of deleting the two annotated lineages.
    """
    weight, value_cost = (
        SimpleWithReporters.weight,
        SimpleWithReporters.value_cost,
    )
    index = {n: i for i, n in enumerate(nodes)}
    cost = norm_reference = norm_other = 0.0
    for chain in lT.get_all_chains_of_subtree(root, end_time=end_time):
        rows = [index[n] for n in chain]
        means = []
        for values in (reference, other):
            measured = values[rows][np.isfinite(values[rows])]
            means.append(measured.mean() if measured.size else None)
        cost += weight * len(chain) * value_cost(*means)
        norm_reference += len(chain) * (1 + weight * value_cost(means[0], 0))
        norm_other += len(chain) * (1 + weight * value_cost(means[1], 0))
    return cost / max(norm_reference, norm_other)


def founder_distance(
    manager, first_node, a, b, lineage, duration, style="simple"
):
    """Compare a founder lineage of two embryos of a manager.

    Parameters
    ----------
    manager : LineageTreeManager
        The manager holding the embryos.
    first_node : dict of {tuple of (str, str): int}
        Maps ``(embryo, lineage)`` to the first node of the founder cell.
    a, b : str
        Names of the two embryos.
    lineage : str
        The founder lineage, e.g. ``"E"``.
    duration : int
        Number of time points after the birth of the founder that are
        compared.
    style : str or type, default="simple"
        The tree style of the comparison.

    Returns
    -------
    float
        The normalised unordered tree edit distance between the two
        lineages.
    """
    ra, rb = first_node[a, lineage], first_node[b, lineage]
    return float(
        manager.cross_lineage_edit_distance(
            ra,
            a,
            rb,
            b,
            manager[a].time[ra] + duration,
            manager[b].time[rb] + duration,
            style=style,
        )
    )


def typical_embryo(embryo_score, gene_of, gene, lineage):
    """Pick the knockdown embryo of a gene with the median score in a lineage.

    Parameters
    ----------
    embryo_score : dict of {tuple of (str, str): float}
        The score of each ``(embryo, lineage)``.
    gene_of : dict of {str: str}
        The gene knocked down in each embryo.
    gene : str
        The gene.
    lineage : str
        The founder lineage.

    Returns
    -------
    str
        Name of the embryo whose score in the lineage is the median one
        among the embryos of the gene (the higher of the two middle ones for
        an even number).
    """
    embryos = sorted(
        (s, name)
        for (name, lin), s in embryo_score.items()
        if lin == lineage and gene_of[name] == gene
    )
    return embryos[len(embryos) // 2][1]


# --------------------------------------------------------------------------
# Comparison of two embryos
# --------------------------------------------------------------------------

SCALE = 100
"""Multiplier of the time resolutions, so that a pace ratio is kept to 0.1%."""


class LineageComparison:
    """Simple UTED between the lineages of a cell in two embryos, cell by cell.

    Matching two cells costs the difference of their lifetimes, and a cell
    without a partner costs its lifetime, in frames of embryo 2.

    Parameters
    ----------
    manager : LineageTreeManager
        The manager holding both embryos.
    name_1, name_2 : str
        The two embryos. The UTED is split over the cells of embryo 1.
    cell : str, default="P0"
        The cell whose lineages are compared.
    match_speed : bool, default=True
        Rescale embryo 1 so that its cell cycles match those of embryo 2 on
        average. Times are then in frames of embryo 2.

    Attributes
    ----------
    speed : float
        Median ratio of the cycle lengths of embryo 1 to those of embryo 2.
    resolution_1, resolution_2 : float
        Time resolutions given to the two trees, ``SCALE`` times the length
        of a frame of each embryo in frames of embryo 2.
    end_1, end_2 : int
        Last frame of each embryo in the comparison: both lineages are
        compared over the same duration, while they are fully tracked.
    partner : dict of {int: int}
        Maps each matched cell of embryo 1 to its partner in embryo 2.
    own : dict of {int: float}
        Cost of each cell of embryo 1: the difference of the lifetimes with
        its partner, or its lifetime if it has none.
    inserted : dict of {int: list of int}
        Maps a cell of embryo 1 to the cells of embryo 2 without a partner
        found below its partner.
    width : dict of {int: float}
        UTED carried by the subtree of each cell of embryo 1.
    n_cells : dict of {int: int}
        Number of cells of the subtree of each cell of embryo 1.
    """

    def __init__(self, manager, name_1, name_2, cell="P0", match_speed=True):
        """Align the lineages and split their UTED over the cells of embryo 1.

        The parameters are described in the docstring of the class.
        """
        self.manager, self.name_1, self.name_2 = manager, name_1, name_2
        self.cell = cell
        self.lT1, self.lT2 = manager[name_1], manager[name_2]

        nodes_1, nodes_2 = first_nodes(self.lT1), first_nodes(self.lT2)
        self.root, self.root_2 = nodes_1[cell], nodes_2[cell]

        cycles_1, cycles_2 = division_cycles(self.lT1), division_cycles(
            self.lT2
        )
        self.speed = np.median(
            [cycles_1[c] / cycles_2[c] for c in cycles_1 if c in cycles_2]
        )
        self.resolution_1 = (
            round(SCALE / self.speed, 1) if match_speed else SCALE
        )
        self.resolution_2 = SCALE

        # same duration, in frames of embryo 2, while both are fully tracked
        duration = min(
            (tracked_until(self.lT1, self.root) - self.lT1.time[self.root])
            * self.resolution_1,
            (tracked_until(self.lT2, self.root_2) - self.lT2.time[self.root_2])
            * self.resolution_2,
        )
        self.end_1 = self.lT1.time[self.root] + int(
            duration / self.resolution_1
        )
        self.end_2 = self.lT2.time[self.root_2] + int(
            duration / self.resolution_2
        )

        self._set_resolutions()
        mapping = manager.labelled_mappings(
            self.root,
            name_1,
            self.root_2,
            name_2,
            self.end_1,
            self.end_2,
            style="simple",
        )
        self.partner = {nodes_1[a]: nodes_2[b] for a, b in mapping["matched"]}
        self._split()

    # time ------------------------------------------------------------------

    def _set_resolutions(self):
        """Give the two trees the time resolutions of the comparison.

        The manager reads the time resolutions from the trees, so they are
        set again before each comparison.
        """
        self.lT1.time_resolution = self.resolution_1
        self.lT2.time_resolution = self.resolution_2

    def time_1(self, node):
        """Return the time of a node of embryo 1, in frames of embryo 2.

        Parameters
        ----------
        node : int
            A node of embryo 1.

        Returns
        -------
        float
            Its time, rescaled to the frames of embryo 2.
        """
        return self.lT1.time[node] * self.resolution_1 / SCALE

    def time_2(self, node):
        """Return the time of a node of embryo 2.

        Parameters
        ----------
        node : int
            A node of embryo 2.

        Returns
        -------
        float
            Its time, in frames of embryo 2.
        """
        return self.lT2.time[node] * self.resolution_2 / SCALE

    def lifetime_1(self, cell):
        """Return how long a cell of embryo 1 lives within the compared window.

        Parameters
        ----------
        cell : int
            A node of a cell of embryo 1, usually its first one.

        Returns
        -------
        float
            The number of its frames up to ``end_1``, in frames of embryo 2.
        """
        chain = self.lT1.get_chain_of_node(cell)
        n = sum(self.lT1.time[node] <= self.end_1 for node in chain)
        return n * self.resolution_1 / SCALE

    def lifetime_2(self, cell):
        """Return how long a cell of embryo 2 lives within the compared window.

        Parameters
        ----------
        cell : int
            A node of a cell of embryo 2, usually its first one.

        Returns
        -------
        float
            The number of its frames up to ``end_2``.
        """
        chain = self.lT2.get_chain_of_node(cell)
        n = sum(self.lT2.time[node] <= self.end_2 for node in chain)
        return n * self.resolution_2 / SCALE

    def daughters_1(self, cell):
        """Return the daughters of a cell of embryo 1.

        Parameters
        ----------
        cell : int
            A node of a cell of embryo 1.

        Returns
        -------
        tuple of int
            The first nodes of its daughters, empty if the cell does not
            divide before ``end_1``.
        """
        last = self.lT1.get_chain_of_node(cell)[-1]
        return (
            self.lT1.successor[last]
            if self.lT1.time[last] < self.end_1
            else ()
        )

    # costs -----------------------------------------------------------------

    def match_cost(self, cell_1, cell_2):
        """Return the cost of matching two cells.

        Parameters
        ----------
        cell_1 : int
            First node of a cell of embryo 1.
        cell_2 : int
            First node of a cell of embryo 2.

        Returns
        -------
        float
            The difference of their lifetimes, in frames of embryo 2.
        """
        return abs(self.lifetime_1(cell_1) - self.lifetime_2(cell_2))

    def uted(self, cell_1, cell_2):
        """Compute the UTED between two subtrees with the manager.

        Parameters
        ----------
        cell_1 : int
            First node of a cell of embryo 1.
        cell_2 : int
            First node of a cell of embryo 2.

        Returns
        -------
        float
            The unnormalised UTED (``simple`` style) between the subtrees of
            the two cells, up to ``end_1`` and ``end_2``, in frames of
            embryo 2.
        """
        self._set_resolutions()
        cost = self.manager.cross_lineage_edit_distance(
            cell_1,
            self.name_1,
            cell_2,
            self.name_2,
            self.end_1,
            self.end_2,
            style="simple",
            norm=None,
        )
        # the manager counts durations in tenths of `time_resolution`
        return cost / 10 / SCALE

    def _split(self):
        """Split the UTED over the cells of embryo 1, and check it.

        Fills ``own``, ``inserted``, ``width`` and ``n_cells``. The width of
        the subtree of a cell is its own cost, plus the lifetimes of the
        cells of embryo 2 inserted below its partner, plus the widths of its
        daughters.

        Raises
        ------
        AssertionError
            If the width of a matched cell differs from the UTED computed by
            the manager between its subtree and its partner's.
        """
        lT1, lT2 = self.lT1, self.lT2
        matched_2 = {c2: c1 for c1, c2 in self.partner.items()}
        chains_1 = [
            c[0] for c in lT1.get_all_chains_of_subtree(self.root, self.end_1)
        ]
        chains_2 = [
            c[0]
            for c in lT2.get_all_chains_of_subtree(self.root_2, self.end_2)
        ]

        self.own = {
            c1: (
                self.match_cost(c1, self.partner[c1])
                if c1 in self.partner
                else self.lifetime_1(c1)
            )
            for c1 in chains_1
        }
        # cell of embryo 1 -> unmatched cells of embryo 2 below its partner
        self.inserted = {}
        for c2 in chains_2:
            if c2 not in matched_2:
                parent = lT2.get_chain_of_node(lT2.predecessor[c2][0])[0]
                while parent not in matched_2:
                    parent = lT2.get_chain_of_node(lT2.predecessor[parent][0])[
                        0
                    ]
                self.inserted.setdefault(matched_2[parent], []).append(c2)

        self.width, self.n_cells = {}, {}

        def fill(c1):
            """Compute the width and the number of cells of a subtree.

            Parameters
            ----------
            c1 : int
                First node of a cell of embryo 1.

            Returns
            -------
            float
                The width of the subtree of ``c1``, also stored in ``width``.
            """
            kids = self.daughters_1(c1)
            self.width[c1] = (
                self.own[c1]
                + sum(self.lifetime_2(c2) for c2 in self.inserted.get(c1, []))
                + sum(fill(d) for d in kids)
            )
            self.n_cells[c1] = 1 + sum(self.n_cells[d] for d in kids)
            return self.width[c1]

        fill(self.root)
        for c1, c2 in self.partner.items():
            assert abs(self.width[c1] - self.uted(c1, c2)) < 1e-6, lT1.labels[
                c1
            ]

    # summaries -------------------------------------------------------------

    @property
    def total(self):
        """Return the UTED between the two lineages.

        Returns
        -------
        float
            The width of the root, in frames of embryo 2.
        """
        return self.width[self.root]

    @property
    def cells(self):
        """Return the number of cells of the compared lineage of embryo 1.

        Returns
        -------
        int
            The number of cells of the subtree of the root, up to ``end_1``.
        """
        return self.n_cells[self.root]

    def name(self, cell):
        """Name a cell of embryo 1 together with its partner.

        Parameters
        ----------
        cell : int
            First node of a cell of embryo 1.

        Returns
        -------
        str
            ``"name ↔ partner"``, or ``"name (no match)"``.
        """
        if cell in self.partner:
            partner = self.lT2.labels[self.partner[cell]]
            return f"{self.lT1.labels[cell]} ↔ {partner}"
        return f"{self.lT1.labels[cell]} (no match)"

    def most_loaded(self):
        """Find the subtree carrying the most distance compared with its size.

        Returns
        -------
        int
            The matched cell of embryo 1, other than the root, whose share of
            the UTED exceeds its share of the cells the most.
        float
            Its share of the UTED.
        float
            Its share of the cells.
        """
        best = max(
            (c for c in self.partner if c != self.root),
            key=lambda c: self.width[c] / self.total
            - self.n_cells[c] / self.cells,
        )
        return (
            best,
            self.width[best] / self.total,
            self.n_cells[best] / self.cells,
        )

    def enrichment(self, cell):
        """Compare a subtree's share of the UTED with its share of the cells.

        Parameters
        ----------
        cell : int
            First node of a cell of embryo 1.

        Returns
        -------
        float
            The share of the UTED carried by the subtree of ``cell``, divided
            by its share of the cells: 1 when the subtree carries what its
            size predicts.
        """
        return (self.width[cell] / self.total) / (
            self.n_cells[cell] / self.cells
        )

    def drill_down(self):
        """Print the path that carries the most distance, from the root.

        At each division, the daughter with the highest ``enrichment`` is
        followed, until a leaf, a cell without partner, or a subtree with
        less than 5% of the cells.
        """
        print(
            f"UTED between the {self.cell} lineages: "
            f"{self.total:.0f} frames, {self.cells} cells"
        )
        cell = self.root
        while True:
            kids = self.daughters_1(cell)
            if not kids:
                break
            followed = max(kids, key=self.enrichment)
            print(f"{self.lT1.labels[cell]} divides into:")
            for d in sorted(kids, key=self.enrichment, reverse=True):
                print(
                    f"  {self.name(d):18} "
                    f"{self.width[d] / self.total:4.0%} of the UTED, "
                    f"{self.n_cells[d] / self.cells:4.0%} of the cells "
                    f"({self.enrichment(d):.1f}×)"  # noqa: RUF001
                    + ("   <-" if d == followed else "")
                )
            cell = followed
            if (
                self.n_cells[cell] / self.cells < 0.05
                or cell not in self.partner
            ):
                break


class ComparisonFigure:
    """Draw a `LineageComparison`: one bar per cell of embryo 1.

    The width of a bar is the UTED of its subtree with its partner's
    subtree, its height the lifetime of the cell in embryo 1. Daughters are
    drawn below their mother, centred, the root cost being the gap left on
    each side. `share`, `enrichment` and `mother` draw the figure with
    different colours.

    Parameters
    ----------
    comparison : LineageComparison
        The comparison to draw.

    Attributes
    ----------
    c : LineageComparison
        The comparison.
    rects : list of dict
        The rectangles to draw, with their position, size, colour values
        and hover text.
    """

    def __init__(self, comparison):
        """Lay out the rectangles of the comparison.

        The parameters are described in the docstring of the class.
        """
        self.c = comparison
        self.rects = []
        self._layout(comparison.root, 0, None)

    def _layout(self, cell, x0, mother_width):
        """Add the rectangle of a cell, then those of its descendants.

        The rectangles of the cells of embryo 2 without partner below the
        partner of the cell come first, then the daughters, side by side.

        Parameters
        ----------
        cell : int
            First node of a cell of embryo 1.
        x0 : float
            Left side of the rectangle.
        mother_width : float or None
            Width of the mother's rectangle, None for the root.
        """
        c = self.c
        top, life = c.time_1(cell), c.lifetime_1(cell)
        share = c.width[cell] / c.total
        cell_share = c.n_cells[cell] / c.cells
        rect = {
            "x0": x0,
            "width": c.width[cell],
            "top": top,
            "height": life,
            "share": share,
            # 1 when the subtree carries the distance its size predicts
            "enrichment": share / cell_share,
            # the root has no mother: neutral
            "share_of_mother": (
                c.width[cell] / mother_width if mother_width else 0.5
            ),
            "root": mother_width is None,
        }
        hover = (
            f"<b>{c.name(cell)}</b><br>"
            f"subtree UTED: {c.width[cell]:.1f} frames<br>"
            f"share of the total: {100 * share:.1f}% "
            f"for {100 * cell_share:.1f}% of the cells "
            f"({rect['enrichment']:.2f}× expected)<br>"  # noqa: RUF001
        )
        if mother_width:
            hover += (
                "share of the mother: "
                f"{100 * rect['share_of_mother']:.0f}%<br>"
            )
        if cell in c.partner:
            partner = c.partner[cell]
            hover += (
                f"root cost: {c.own[cell]:.1f}<br>"
                f"lifetime: {life:.1f} vs "
                f"{c.lifetime_2(partner):.1f} frames<br>"
                f"born at: {top:.1f} vs {c.time_2(partner):.1f}<br>"
                f"cells in the subtree: {c.n_cells[cell]}"
            )
            kind = "matched"
        else:
            hover += f"born at: {top:.1f}"
            kind = "only in embryo 1"
        rect |= {"kind": kind, "name": c.name(cell), "hover": hover}
        self.rects.append(rect)

        x = x0 + c.own[cell] / 2
        for cell_2 in c.inserted.get(cell, []):
            cost = c.lifetime_2(cell_2)
            self.rects.append(
                {
                    "x0": x,
                    "width": cost,
                    "top": top + life,
                    "height": cost,
                    "kind": "only in embryo 2",
                    "name": c.lT2.labels[cell_2],
                    "hover": (
                        f"<b>{c.lT2.labels[cell_2]}</b>: "
                        "no match in embryo 1<br>"
                        f"cost: {cost:.1f} frames (its lifetime)<br>"
                        f"born at: {c.time_2(cell_2):.1f} in embryo 2"
                    ),
                }
            )
            x += cost
        for d in sorted(c.daughters_1(cell), key=c.lT1.labels.get):
            self._layout(d, x, c.width[cell])
            x += c.width[d]

    def _bars(self, kind, label=None, **marker):
        """Make one bar trace with the rectangles of a kind.

        Parameters
        ----------
        kind : str
            ``"matched"``, ``"only in embryo 1"`` or ``"only in embryo 2"``.
        label : Callable, optional
            Maps a rectangle to the text written in it, after its name; only
            wide rectangles are labelled.
        **marker
            Passed to the marker of the trace (colour, colour scale...).

        Returns
        -------
        plotly.graph_objects.Bar
            The trace.
        """
        rs = [r for r in self.rects if r["kind"] == kind]
        label_width = 0.06 * self.c.total  # label only the wide bars
        return go.Bar(
            x=[r["x0"] + r["width"] / 2 for r in rs],
            width=[r["width"] for r in rs],
            base=[r["top"] for r in rs],
            y=[r["height"] for r in rs],
            text=[
                (
                    f"{r['name']}<br>{label(r)}"
                    if label and r["width"] > label_width
                    else ""
                )
                for r in rs
            ],
            textposition="inside",
            insidetextanchor="middle",
            hovertext=[r["hover"] for r in rs],
            hoverinfo="text",
            name=kind,
            marker=dict(line=dict(width=0.5, color="white"), **marker),
        )

    def plot(self, colour, label, subtitle, **colour_scale):
        """Draw the figure.

        Parameters
        ----------
        colour : Callable
            Maps a rectangle of a matched cell to its colour value.
        label : Callable
            Maps a rectangle to the text written in it.
        subtitle : str
            Explains the colours, below the title.
        **colour_scale
            Colour scale of the matched cells, passed to their marker.

        Returns
        -------
        plotly.graph_objects.Figure
            The figure.
        """
        c = self.c
        matched = [r for r in self.rects if r["kind"] == "matched"]
        fig = go.Figure(
            [
                self._bars(
                    "matched",
                    label,
                    color=[colour(r) for r in matched],
                    **colour_scale,
                ),
                self._bars(
                    "only in embryo 1",
                    color="lightgrey",
                    pattern=dict(shape="/"),
                ),
                self._bars(
                    "only in embryo 2",
                    color="lightgrey",
                    pattern=dict(shape="x"),
                ),
            ]
        )
        fig.update_layout(
            title=(
                f"{c.cell}: {c.name_1} (embryo 1) vs {c.name_2} (embryo 2)<br>"
                f"<sup>UTED {c.total:.0f} frames. {subtitle}</sup>"
            ),
            xaxis=dict(title="UTED (frames)", range=[0, c.total]),
            yaxis=dict(
                title="time in embryo 1 (frames of embryo 2)",
                autorange="reversed",
            ),
            barmode="overlay",
            bargap=0,
            uniformtext=dict(minsize=6, mode="hide"),
            legend=dict(orientation="h", y=-0.12),
            height=750,
            plot_bgcolor="white",
        )
        return fig

    def share(self):
        """Draw the figure coloured by each subtree's share of the UTED.

        Returns
        -------
        plotly.graph_objects.Figure
            The figure.
        """
        return self.plot(
            colour=lambda r: 100 * r["share"],
            label=lambda r: f"{100 * r['share']:.0f}%",
            subtitle="colour: share of the total UTED carried by the subtree",
            colorscale="Magma",
            reversescale=True,
            cmin=0,
            cmax=100,
            colorbar=dict(title="share of the<br>total UTED", ticksuffix="%"),
        )

    def enrichment(self):
        """Draw the figure coloured by UTED share over cell share.

        Returns
        -------
        plotly.graph_objects.Figure
            The figure.
        """
        return self.plot(
            colour=lambda r: float(
                np.clip(np.log2(max(r["enrichment"], 1e-3)), -3, 3)
            ),
            label=lambda r: f"{r['enrichment']:.1f}×",  # noqa: RUF001
            subtitle=(
                "colour: share of the total UTED ÷ share of the cells "
                "(1 = expected)"
            ),
            colorscale="RdBu",
            reversescale=True,
            cmin=-3,
            cmax=3,
            colorbar=dict(
                title="UTED share ÷<br>cell share",
                tickvals=[-3, -2, -1, 0, 1, 2, 3],
                ticktext=["1/8", "1/4", "1/2", "1", "2", "4", "8"],
            ),
        )

    def mother(self):
        """Draw the figure coloured by the share of the mother's UTED.

        Returns
        -------
        plotly.graph_objects.Figure
            The figure.
        """
        return self.plot(
            colour=lambda r: 100 * r["share_of_mother"],
            label=lambda r: (
                "–"  # noqa: RUF001
                if r["root"]
                else f"{100 * r['share_of_mother']:.0f}%"
            ),
            subtitle=(
                "colour: share of the mother's UTED carried by the subtree"
            ),
            colorscale="RdBu",
            reversescale=True,
            cmin=0,
            cmax=100,
            cmid=50,
            colorbar=dict(
                title="share of the<br>mother's UTED", ticksuffix="%"
            ),
        )
