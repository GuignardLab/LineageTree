"""Helpers of the notebook ``phallusia_variability.ipynb``.

The notebook compares nine embryos of the ascidian *Phallusia mammillata*,
segmented and tracked with ASTEC (Guignard et al., Science 369, eaar5663,
2020) and named with the nomenclature of Conklin (1905). This module holds
the code that the notebook would otherwise repeat:

- the cell names: parsing, mothers, daughters, sisters and bilateral
  homologues;
- loading the ASTEC files into a ``LineageTreeManager``;
- one table per embryo with one row per named cell;
- the linear alignment of the developmental time of the embryos;
- measures of variability: of timing, volume, shape, position, contacts,
  and of the lineage trees with the unordered tree edit distance;
- two recap figures, sized for slides;
- simple figures for slides, each showing one kind of variability with
  one example;
- for ``phallusia_tissue_time_variability.ipynb``: tables of pairwise
  differences for each measure, confidence intervals from resampled
  embryos, a model that separates the effects of tissue, time and cell
  size, and the figures of that notebook.
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lineagetree import LineageTreeManager, read_from_ASTEC
from lineagetree.tree_approximation import simple_tree
from scipy.optimize import minimize

DATA_DIR = Path(__file__).resolve().parent / "data" / "Pmmamm"
SOURCE = (
    "https://figshare.com/projects/"
    "Phallusia_mammillata_embryonic_development/64301"
)
EMBRYOS = {
    "Phmamm-1-v2": "Phmamm-1-v2_properties.xml",
    "Phmamm-3-v2": "Phmamm-3-v2_properties.xml",
    "Phmamm-4-v2": "Phmamm-4-v2_properties.xml",
    "Phmamm-5-v2": "Phmamm-5-v2_properties.xml",
    "Phmamm-7-v2": "Phmamm-7-v2_properties.xml",
    "Phmamm-8-v2": "Phmamm-8-v2_properties.xml",
    "Phmamm-9-v2": "Phmamm-9-v2_properties.xml",
    "Astec-Pm6": "Astec-Pm6_properties.xml",
    "Astec-Pm10": "Astec-pm10_properties.xml",
}
REFERENCE = "Phmamm-8-v2"
FOUNDERS = {
    "a": "a-line (anterior animal)",
    "b": "b-line (posterior animal)",
    "A": "A-line (anterior vegetal)",
    "B": "B-line (posterior vegetal)",
}
TISSUE_GROUPS = {
    "Head Epidermis": "epidermis",
    "Lateral Tail Epidermis": "epidermis",
    "Midline Tail Epidermis": "epidermis",
    "Medio-Lateral Tail Epidermis": "epidermis",
    "Tail Epidermis": "epidermis",
    "Anterior Ventral Neural Plate": "neural",
    "Posterior Ventral Neural Plate": "neural",
    "Anterior Dorsal Neural Plate": "neural",
    "Posterior Dorsal Neural Plate": "neural",
    "Posterior Lateral Neural Plate": "neural",
    "1st Lineage, Notochord": "notochord",
    "2nd Lineage, Notochord": "notochord",
    "1st Lineage, Tail Muscle": "muscle",
    "2nd Lineage, Tail Muscle": "muscle",
    "Mesenchyme": "mesenchyme",
    "Trunk Lateral Cells": "trunk mesoderm",
    "Trunk Ventral Cells": "trunk mesoderm",
    "Anterior Head Endoderm": "endoderm",
    "Posterior Head Endoderm": "endoderm",
    "Endodermal Strand": "endoderm",
    "Germ Line": "germ line",
    "Tail-tip Cells": "tail tip",
}
TISSUE_COLORS = {
    "epidermis": "#9e9e9e",
    "neural": "#1f77b4",
    "notochord": "#9467bd",
    "muscle": "#d62728",
    "mesenchyme": "#ff7f0e",
    "trunk mesoderm": "#8c564b",
    "endoderm": "#2ca02c",
    "germ line": "#e377c2",
    "tail tip": "#bcbd22",
    "mixed": "#17becf",
}

# --------------------------------------------------------------------------
# Cell names
# --------------------------------------------------------------------------

NAME = re.compile(r"^([aAbB])(\d+)\.(\d{4})([_*])$")


def parse_name(name):
    """Split a Conklin name into its parts.

    ASTEC writes the names as ``a7.0009_``: a letter, ``a`` for the
    anterior and ``b`` for the posterior half of the embryo, the generation
    (7 at the 64-cell stage), the index of the cell in its generation, and
    the side, ``_`` or ``*`` for the two bilateral halves. The letter is
    always lowercase: the vegetal (``A``, ``B``) and animal (``a``, ``b``)
    founder lines are told apart by the index (see ``founder``).

    Parameters
    ----------
    name : str
        A cell name.

    Returns
    -------
    tuple of (str, int, int, str) or None
        The letter, the generation, the index and the side, or None if
        ``name`` is not a valid name.
    """
    match = NAME.match(str(name))
    if match is None:
        return None
    line, gen, index, side = match.groups()
    return line, int(gen), int(index), side


def founder(name):
    """Founder line of a cell: ``A``, ``B``, ``a`` or ``b``.

    The four founder lines are the descendants of the four cells of each
    side at the 8-cell stage: A4.1 (anterior vegetal), B4.1 (posterior
    vegetal), a4.2 (anterior animal) and b4.2 (posterior animal). At
    generation ``g``, the vegetal cells have the indices 1 to ``2**(g-4)``
    and the animal cells the next ones.

    Parameters
    ----------
    name : str
        A cell name.

    Returns
    -------
    str
        The founder line.
    """
    letter, gen, index, _ = parse_name(name)
    return letter.upper() if index <= 2 ** (gen - 4) else letter.lower()


def conklin(name):
    """Name of a cell as written in the literature, e.g. ``B7.6*``.

    Parameters
    ----------
    name : str
        A cell name, as written by ASTEC (``b7.0006*``).

    Returns
    -------
    str
        The name with its founder line, without the zeros of the index, and
        with ``*`` for the cells of the ``*`` side only.
    """
    _, gen, index, side = parse_name(name)
    return f"{founder(name)}{gen}.{index}{'*' if side == '*' else ''}"


def format_name(line, gen, index, side):
    """Write a cell name from its parts.

    Parameters
    ----------
    line : str
        The founder line, ``a``, ``b``, ``A`` or ``B``.
    gen : int
        The generation.
    index : int
        The index of the cell in its generation.
    side : str
        ``_`` or ``*``.

    Returns
    -------
    str
        The name, e.g. ``a7.0009_``.
    """
    return f"{line}{gen}.{index:04d}{side}"


def mother(name):
    """Name of the mother of a cell.

    Parameters
    ----------
    name : str
        A cell name.

    Returns
    -------
    str
        The name of its mother: the daughters of F(g).p are F(g+1).(2p-1)
        and F(g+1).(2p).
    """
    line, gen, index, side = parse_name(name)
    return format_name(line, gen - 1, (index + 1) // 2, side)


def daughters(name):
    """Names of the two daughters of a cell.

    Parameters
    ----------
    name : str
        A cell name.

    Returns
    -------
    tuple of (str, str)
        The names of its daughters, the odd index first.
    """
    line, gen, index, side = parse_name(name)
    return (
        format_name(line, gen + 1, 2 * index - 1, side),
        format_name(line, gen + 1, 2 * index, side),
    )


def sister(name):
    """Name of the sister of a cell.

    Parameters
    ----------
    name : str
        A cell name.

    Returns
    -------
    str
        The name of the other daughter of its mother.
    """
    line, gen, index, side = parse_name(name)
    other = index + 1 if index % 2 else index - 1
    return format_name(line, gen, other, side)


def homologue(name):
    """Name of the bilateral homologue of a cell.

    Parameters
    ----------
    name : str
        A cell name.

    Returns
    -------
    str
        The same name on the other side of the embryo.
    """
    return name[:-1] + ("*" if name.endswith("_") else "_")


def ancestor(name, gen):
    """Name of the ancestor of a cell at a given generation.

    Parameters
    ----------
    name : str
        A cell name.
    gen : int
        A generation, at most that of the cell.

    Returns
    -------
    str
        The name of its ancestor at generation ``gen`` (the cell itself if
        ``gen`` is its generation).
    """
    line, g, index, side = parse_name(name)
    return format_name(line, gen, (index - 1) // 2 ** (g - gen) + 1, side)


def is_ancestor(name, other):
    """Whether a cell is an ancestor of another one, or the same cell.

    Parameters
    ----------
    name : str
        A cell name.
    other : str
        Another cell name.

    Returns
    -------
    bool
        True if ``name`` is ``other`` or one of its ancestors.
    """
    p, q = parse_name(name), parse_name(other)
    if p[0] != q[0] or p[3] != q[3] or p[1] > q[1]:
        return False
    return ancestor(other, p[1]) == name


def tissue_group(fates):
    """Coarse tissue of a cell from the tissues of its descendants.

    Parameters
    ----------
    fates : list of str
        The tissues its descendants contribute to (``cell_tissue_fate`` of
        the ASTEC files).

    Returns
    -------
    str
        One of the groups of ``TISSUE_GROUPS``, or ``"mixed"`` if the
        descendants contribute to several groups.
    """
    groups = {TISSUE_GROUPS.get(f, f) for f in fates or []}
    if len(groups) == 1:
        return groups.pop()
    return "mixed" if groups else "unknown"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_embryos(folder=DATA_DIR, embryos=None):
    """Read the ASTEC files of the embryos into a lineage tree manager.

    Each file is read with ``read_from_ASTEC``. The cell names, stored by
    the reader in the ``label`` property, are made the labels of the tree,
    and each tree gets the attribute ``version``: ``"v2"`` for the 2023
    releases, curated and named with the reference atlas of Biasuz et al.
    (bioRxiv, 2025), and ``"v1"`` for the 2020 releases.

    Parameters
    ----------
    folder : pathlib.Path, optional
        The folder of the ``*_properties.xml`` files.
    embryos : dict of {str: str}, optional
        Maps the name of each embryo to its file; ``EMBRYOS`` by default.

    Returns
    -------
    LineageTreeManager
        The embryos, under the names of ``embryos``.

    Raises
    ------
    FileNotFoundError
        If a file is missing; the files are distributed on figshare.
    """
    manager = LineageTreeManager()
    for name, file in (embryos or EMBRYOS).items():
        path = Path(folder) / file
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; see {SOURCE}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lT = read_from_ASTEC(str(path), name=name)
            lT.change_labels("label")
        lT.version = "v2" if name.endswith("-v2") else "v1"
        manager.add(lT, name)
    return manager


def embryo_volume(lT):
    """Total volume of the segmented cells at each time point.

    Parameters
    ----------
    lT : LineageTree
        An embryo.

    Returns
    -------
    dict of {int: float}
        Maps each time point to the sum of the volumes of its cells, in
        voxels.
    """
    return {
        t: float(sum(lT.volume[n] for n in nodes))
        for t, nodes in lT.time_nodes.items()
    }


def cell_counts(lT):
    """Number of cells at each time point.

    Parameters
    ----------
    lT : LineageTree
        An embryo.

    Returns
    -------
    pandas.Series
        The number of cells, indexed by time point.
    """
    return pd.Series(
        {t: len(nodes) for t, nodes in sorted(lT.time_nodes.items())}
    )


def sphericity(volume, surface):
    """Sphericity of a cell: 1 for a sphere, less for any other shape.

    Parameters
    ----------
    volume : float or numpy.ndarray
        Volume, in voxels.
    surface : float or numpy.ndarray
        Surface area, in voxel faces.

    Returns
    -------
    float or numpy.ndarray
        The surface of the sphere of the same volume divided by the surface
        of the cell.
    """
    return np.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / surface


def cell_table(lT):
    """One row per named cell of an embryo.

    A cell is a chain of the tree: the nodes between two divisions. Cells
    with no valid name, with names that change along the chain, or whose
    name is used by two cells, are left out.

    Parameters
    ----------
    lT : LineageTree
        An embryo, as returned by ``load_embryos``.

    Returns
    -------
    pandas.DataFrame
        Indexed by cell name, with the columns:

        - ``line``: the founder line (see ``founder``), and ``gen``,
          ``index``, ``side``: the other parts of the name;
        - ``first``, ``last``: the first and last nodes of the cell;
        - ``start``, ``end``: their time points;
        - ``birth``: the time point of the first node, NaN if the cell is
          already there at the first time point;
        - ``division``: the time point of the last node, NaN if the cell
          does not divide before the end of the movie;
        - ``frames``: the number of time points of the cell;
        - ``volume``: the median volume of the cell over its life, as a
          fraction of the volume of the embryo;
        - ``sphericity``, ``exterior``: the median sphericity, and the
          median fraction of the surface of the cell in contact with the
          outside of the embryo (v2 embryos only, NaN otherwise);
        - ``elongation``: the median ratio of the longest to the shortest
          axis of the cell, from the principal values of its shape;
        - ``tissue``: the coarse tissue of the cell (see ``tissue_group``).
    """
    total = embryo_volume(lT)
    has_shape = hasattr(lT, "cell_surface")
    rows, seen = {}, set()
    for chain in lT.all_chains:
        names = {lT.label.get(n, "") for n in chain}
        if len(names) != 1:
            continue
        name = names.pop()
        parts = parse_name(name)
        if parts is None:
            continue
        if name in rows:
            seen.add(name)
            continue
        first, last = chain[0], chain[-1]
        volume = np.array([lT.volume[n] / total[lT.time[n]] for n in chain])
        row = {
            "line": founder(name),
            "gen": parts[1],
            "index": parts[2],
            "side": parts[3],
            "first": first,
            "last": last,
            "start": lT.time[first],
            "end": lT.time[last],
            "birth": lT.time[first] if lT.predecessor.get(first) else np.nan,
            "division": (
                lT.time[last] if len(lT.successor[last]) == 2 else np.nan
            ),
            "frames": len(chain),
            "volume": float(np.median(volume)),
        }
        if has_shape:
            surface = np.array([lT.cell_surface[n] for n in chain])
            raw_volume = np.array([lT.volume[n] for n in chain])
            outside = [lT.contact[n].get(-1, 0.0) for n in chain]
            values = np.array([lT.cell_principal_values[n] for n in chain])
            row["sphericity"] = float(
                np.median(sphericity(raw_volume, surface))
            )
            row["exterior"] = float(np.median(np.array(outside) / surface))
            row["elongation"] = float(
                np.median(np.sqrt(values[:, 0] / values[:, 2]))
            )
            row["tissue"] = tissue_group(lT.cell_tissue_fate.get(first))
        else:
            row.update(sphericity=np.nan, exterior=np.nan, elongation=np.nan)
            row["tissue"] = "unknown"
        rows[name] = row
    for name in seen:
        rows.pop(name)
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "name"
    return table.sort_values(["start", "line", "gen", "index", "side"])


def tissue_of(tables):
    """Tissue of each cell name, taken from the embryos that annotate it.

    Parameters
    ----------
    tables : dict of {str: pandas.DataFrame}
        The cell tables of the embryos, by embryo name.

    Returns
    -------
    pandas.Series
        The most frequent known tissue of each cell name.
    """
    tissues = pd.concat(
        [t.tissue[t.tissue != "unknown"] for t in tables.values()]
    )
    return tissues.groupby(level=0).agg(lambda s: s.mode().iloc[0])


def fates_of(manager, tables):
    """Tissues that the descendants of each named cell contribute to.

    Parameters
    ----------
    manager : LineageTreeManager
        The embryos, with the ``cell_tissue_fate`` property.
    tables : dict of {str: pandas.DataFrame}
        The cell tables of the embryos to use, by embryo name.

    Returns
    -------
    pandas.Series
        The most frequent set of tissues of each cell name, as a frozenset
        of the tissue names of the ASTEC files.
    """
    found = {}
    for e, table in tables.items():
        lT = manager[e]
        for name, first in table["first"].items():
            fates = lT.cell_tissue_fate.get(first)
            if fates:
                found.setdefault(name, []).append(frozenset(fates))
    return pd.Series(
        {n: max(set(v), key=v.count) for n, v in found.items()}, dtype=object
    )


def restricts_fate(fates, name, coarse=True):
    """Whether the two daughters of a cell contribute to different tissues.

    Parameters
    ----------
    fates : pandas.Series
        The set of tissues of each cell name (see ``fates_of``).
    name : str
        The name of the mother.
    coarse : bool, default=True
        Compare the tissues grouped as in ``TISSUE_GROUPS`` (epidermis,
        neural, notochord...). Otherwise, the territories of the ASTEC
        files are compared, so that, for instance, daughters making the
        lateral and the medio-lateral tail epidermis differ.

    Returns
    -------
    bool or None
        True if the sets of tissues of the two daughters differ, None if
        one of them is unknown.
    """
    odd, even = daughters(name)
    if odd not in fates.index or even not in fates.index:
        return None
    first, second = fates[odd], fates[even]
    if coarse:
        first = {TISSUE_GROUPS.get(f, f) for f in first}
        second = {TISSUE_GROUPS.get(f, f) for f in second}
    return first != second


# --------------------------------------------------------------------------
# Time alignment
# --------------------------------------------------------------------------


def _robust_line(x, y, cutoff=4.0):
    """Fit a line, then fit it again without the outliers.

    Parameters
    ----------
    x, y : numpy.ndarray
        The points.
    cutoff : float, default=4.0
        Points further than ``cutoff`` robust standard deviations (1.4826
        times the median absolute deviation) from the first line are left
        out of the second fit.

    Returns
    -------
    tuple of (float, float)
        The slope and the intercept.
    """
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (slope * x + intercept)
    spread = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
    keep = np.abs(residuals) <= cutoff * max(spread, 1e-9)
    return tuple(np.polyfit(x[keep], y[keep], 1))


class TimeAlignment:
    """Linear alignment of the developmental time of several embryos.

    Each embryo ``e`` gets a slope ``a_e`` and an offset ``b_e``: its time
    point ``t`` corresponds to time point ``a_e * t + b_e`` of the reference
    embryo. The coefficients are fitted on the birth times of the named
    cells. Each embryo is aligned on the median birth time of each of its
    cells in the other embryos; the medians are updated and the fits
    repeated until they no longer change. The median of a cell only uses
    the embryos whose movie covers its birth with ``margin`` time points to
    spare, so that the end of a movie does not bias it: near the end of a
    movie, only the cells born early in that embryo are seen.

    Parameters
    ----------
    tables : dict of {str: pandas.DataFrame}
        The cell tables of the embryos (see ``cell_table``), by embryo name.
    reference : str, default=REFERENCE
        The embryo whose time points are the common time axis.
    margin : float, default=10
        Margin, in time points of the reference, between an event and the
        ends of a movie for the event to count in that embryo.
    references : dict of {str: object}, optional
        The ``LineageTree`` of the reference, under its name, to read the
        time post fertilization of its first time point and its time
        interval (``embryo_time_post_fertilization`` and
        ``embryo_time_interval``). Without it, 240 and 2 minutes are used.

    Attributes
    ----------
    slope, offset : dict of {str: float}
        The coefficients of each embryo.
    coverage : dict of {str: tuple of (float, float)}
        The first and last time points of each movie, in reference time.
    """

    def __init__(
        self, tables, reference=REFERENCE, margin=10, references=None
    ):
        """Fit the alignment; see the class docstring for the parameters."""
        self.tables = tables
        self.reference = reference
        self.margin = margin
        self.embryos = list(tables)
        self.frames = {
            e: (t.start.min(), t.end.max()) for e, t in tables.items()
        }
        self.start_minutes, self.interval = 240.0, 2.0
        if references and reference in references:
            lT = references[reference]
            self.start_minutes = float(
                next(iter(lT.embryo_time_post_fertilization.values()))
            )
            self.interval = float(next(iter(lT.embryo_time_interval.values())))
        self.slope = {e: 1.0 for e in self.embryos}
        self.offset = {e: 0.0 for e in self.embryos}
        self._fit()

    def _set(self, coefficients):
        """Store coefficients, re-expressed on the time axis of the reference.

        Parameters
        ----------
        coefficients : dict of {str: tuple of (float, float)}
            Slope and offset of each embryo, on any common time axis.
        """
        a_ref, b_ref = coefficients[self.reference]
        for e, (a, b) in coefficients.items():
            self.slope[e] = a / a_ref
            self.offset[e] = (b - b_ref) / a_ref
        self.coverage = {
            e: tuple(self.to_reference(e, np.array(self.frames[e])))
            for e in self.embryos
        }

    def _fit(self, iterations=100, tolerance=1e-7):
        """Fit the coefficients on the birth times of the named cells.

        Parameters
        ----------
        iterations : int, default=100
            Maximum number of rounds of fits.
        tolerance : float, default=1e-7
            The fits stop when no slope changes by more than this.
        """
        births = self.event_frames("birth")
        self._set({e: (1.0, 0.0) for e in self.embryos})
        for _ in range(iterations):
            new = {}
            for e in self.embryos:
                target = self._others_median(births, e)
                ok = births[e].notna() & target.notna()
                new[e] = _robust_line(births[e][ok].values, target[ok].values)
            old = dict(self.slope)
            self._set(new)
            if max(abs(self.slope[e] - old[e]) for e in self.embryos) < (
                tolerance
            ):
                break

    def add_embryo(self, embryo, slope, offset, frames):
        """Add an embryo aligned by other means, e.g. on its cell counts.

        The embryo can then be converted to reference time, but it takes
        no part in the consensus times.

        Parameters
        ----------
        embryo : str
            The embryo.
        slope, offset : float
            Its coefficients: its time point ``t`` matches reference time
            point ``slope * t + offset``.
        frames : tuple of (int, int)
            Its first and last time points.
        """
        self.slope[embryo], self.offset[embryo] = slope, offset
        self.frames[embryo] = frames
        self.coverage[embryo] = tuple(
            self.to_reference(embryo, np.array(frames, dtype=float))
        )

    def event_frames(self, event):
        """Time points of an event of each cell, in each embryo.

        Parameters
        ----------
        event : {"birth", "division"}
            The event.

        Returns
        -------
        pandas.DataFrame
            Time points, in the frames of each embryo, indexed by cell name,
            with one column per embryo.
        """
        return pd.DataFrame({e: t[event] for e, t in self.tables.items()})

    def to_reference(self, embryo, t):
        """Convert time points of an embryo to reference time points.

        Parameters
        ----------
        embryo : str
            The embryo.
        t : float or numpy.ndarray or pandas.Series
            Its time points.

        Returns
        -------
        float or numpy.ndarray or pandas.Series
            The corresponding time points of the reference.
        """
        return self.slope[embryo] * t + self.offset[embryo]

    def minutes(self, embryo, t):
        """Convert time points of an embryo to minutes post fertilization.

        The minutes are those of the reference embryo: its time point ``t``
        is at ``start_minutes + interval * t`` minutes, the convention of
        the ``embryo_time_post_fertilization`` property of the ASTEC files
        (the time of time point 0).

        Parameters
        ----------
        embryo : str
            The embryo.
        t : float or numpy.ndarray or pandas.Series
            Its time points.

        Returns
        -------
        float or numpy.ndarray or pandas.Series
            Minutes post fertilization, in reference time.
        """
        reference = self.to_reference(embryo, t)
        return self.start_minutes + self.interval * reference

    def frame_minutes(self, embryo):
        """Duration of one time point of an embryo, in reference minutes.

        Parameters
        ----------
        embryo : str
            The embryo.

        Returns
        -------
        float
            ``interval`` times its slope.
        """
        return self.interval * self.slope[embryo]

    def aligned(self, event):
        """Reference time of an event of each cell, in each embryo.

        Parameters
        ----------
        event : {"birth", "division"}
            The event.

        Returns
        -------
        pandas.DataFrame
            Indexed by cell name, one column per embryo.
        """
        frames = self.event_frames(event)
        return pd.DataFrame(
            {e: self.to_reference(e, frames[e]) for e in self.embryos}
        )

    def covers(self, embryo, start, end=None):
        """Whether the movie of an embryo covers a time window with margin.

        Parameters
        ----------
        embryo : str
            The embryo.
        start : float or numpy.ndarray or pandas.Series
            Start of the window, in reference time.
        end : float or numpy.ndarray or pandas.Series, optional
            End of the window; ``start`` by default.

        Returns
        -------
        bool or numpy.ndarray or pandas.Series
            True where ``[start - margin, end + margin]`` is in the movie.
        """
        end = start if end is None else end
        low, high = self.coverage[embryo]
        return (start - self.margin >= low) & (end + self.margin <= high)

    def _others_median(self, frames, embryo):
        """Median reference time of each event in the other embryos.

        Parameters
        ----------
        frames : pandas.DataFrame
            Time points of the event in each embryo (``event_frames``).
        embryo : str
            The embryo left out; only the embryos whose movie covers the
            time of the event in ``embryo`` count.

        Returns
        -------
        pandas.Series
            The median, indexed by cell name.
        """
        own = self.to_reference(embryo, frames[embryo])
        values = []
        for other in self.embryos:
            if other == embryo:
                continue
            times = self.to_reference(other, frames[other])
            values.append(times.where(self.covers(other, own)))
        return pd.concat(values, axis=1).median(axis=1)

    def consensus(self, event="birth"):
        """Consensus reference time of an event of each cell.

        Parameters
        ----------
        event : {"birth", "division"}, default="birth"
            The event.

        Returns
        -------
        pandas.Series
            The median over the embryos whose movie covers the event with
            margin, indexed by cell name.
        """
        times = self.aligned(event)
        first = times.median(axis=1)
        kept = pd.DataFrame(
            {e: times[e].where(self.covers(e, first)) for e in self.embryos}
        )
        return kept.median(axis=1)

    def deviations(self, event="birth"):
        """Advance or delay of each event in each embryo, in minutes.

        Parameters
        ----------
        event : {"birth", "division"}, default="birth"
            The event.

        Returns
        -------
        pandas.DataFrame
            For each cell and embryo, the reference time of the event minus
            its median in the other embryos that cover it, in reference
            minutes: positive when the embryo is late. Events whose
            consensus time is not covered by the movie of the embryo, with
            margin, are NaN, since only the early ones would be seen.
        """
        frames = self.event_frames(event)
        consensus = self.consensus(event)
        out = {}
        for e in self.embryos:
            own = self.to_reference(e, frames[e])
            deviation = (own - self._others_median(frames, e)) * self.interval
            out[e] = deviation.where(self.covers(e, consensus))
        return pd.DataFrame(out)

    def summary(self):
        """Coefficients of each embryo.

        Returns
        -------
        pandas.DataFrame
            One row per embryo: slope, offset, the duration of a time point
            in reference minutes, the minutes post fertilization of the
            first and last time points, and the median absolute deviation
            of the birth times from the other embryos.
        """
        deviations = self.deviations("birth")
        rows = {}
        for e in self.embryos:
            first, last = self.frames[e]
            rows[e] = {
                "slope": self.slope[e],
                "offset": self.offset[e],
                "minutes per time point": self.frame_minutes(e),
                "start (min)": self.minutes(e, first),
                "end (min)": self.minutes(e, last),
                "births": int(deviations[e].notna().sum()),
                "median |deviation| (min)": float(
                    deviations[e].abs().median()
                ),
            }
        return pd.DataFrame.from_dict(rows, orient="index")


def count_curve(counts, alignment, embryos, grid=None):
    """Median number of cells over time, from aligned embryos.

    Parameters
    ----------
    counts : dict of {str: pandas.Series}
        The number of cells at each time point of each embryo
        (``cell_counts``).
    alignment : TimeAlignment
        The alignment of the embryos.
    embryos : list of str
        The embryos to use.
    grid : numpy.ndarray, optional
        Reference time points at which to compute the curve; every 0.5 time
        point over the aligned movies by default.

    Returns
    -------
    pandas.Series
        The median of the cell counts, indexed by reference time, where at
        least two embryos cover the time point.
    """
    if grid is None:
        low = min(alignment.coverage[e][0] for e in embryos)
        high = max(alignment.coverage[e][1] for e in embryos)
        grid = np.arange(np.floor(low), np.ceil(high) + 0.5, 0.5)
    curves = []
    for e in embryos:
        x = alignment.to_reference(e, counts[e].index.values.astype(float))
        y = np.interp(grid, x, counts[e].values, left=np.nan, right=np.nan)
        curves.append(y)
    curves = np.array(curves)
    enough = (~np.isnan(curves)).sum(axis=0) >= 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(curves, axis=0)
    return pd.Series(median[enough], index=grid[enough])


def fit_to_count_curve(counts, curve, start=(1.0, 0.0)):
    """Align an embryo on a reference curve of cell counts.

    The slope and offset minimise the mean squared difference between the
    logarithm of the number of cells of the embryo and that of the curve,
    over the time points where they overlap.

    Parameters
    ----------
    counts : pandas.Series
        The number of cells at each time point of the embryo.
    curve : pandas.Series
        The reference curve, indexed by reference time (``count_curve``).
    start : tuple of (float, float), default=(1.0, 0.0)
        Initial slope and offset. A coarse grid around it is searched first.

    Returns
    -------
    tuple of (float, float)
        The slope and the offset.
    """
    t = counts.index.values.astype(float)
    log_counts = np.log(counts.values)
    log_curve = np.log(curve.values)
    grid = curve.index.values

    def cost(params):
        """Mean squared difference of the logarithms of the cell numbers.

        Parameters
        ----------
        params : sequence of float
            The slope and the offset.

        Returns
        -------
        float
            The mean squared difference over the time points where the
            embryo and the curve overlap, or 1000 if they overlap on fewer
            than 10 time points.
        """
        x = params[0] * t + params[1]
        inside = (x >= grid[0]) & (x <= grid[-1])
        if inside.sum() < 10:
            return 1e3
        reference = np.interp(x[inside], grid, log_curve)
        return float(np.mean((log_counts[inside] - reference) ** 2))

    candidates = [
        (start[0] * s, start[1] + o)
        for s in np.linspace(0.8, 1.25, 19)
        for o in np.linspace(-30, 30, 61)
    ]
    best = min(candidates, key=cost)
    result = minimize(cost, best, method="Nelder-Mead")
    return float(result.x[0]), float(result.x[1])


# --------------------------------------------------------------------------
# Snapshots at matched developmental times
# --------------------------------------------------------------------------


def frame_at(alignment, embryo, reference_time):
    """Time point of an embryo that matches a reference time point.

    Parameters
    ----------
    alignment : TimeAlignment
        The alignment of the embryos.
    embryo : str
        The embryo.
    reference_time : float
        A reference time point.

    Returns
    -------
    int or None
        The nearest time point of the embryo, or None if it is outside its
        movie.
    """
    t = round(
        (reference_time - alignment.offset[embryo]) / alignment.slope[embryo]
    )
    first, last = alignment.frames[embryo]
    return int(t) if first <= t <= last else None


def named_nodes(lT, t, min_age=0):
    """Named cells of an embryo at a time point.

    Parameters
    ----------
    lT : LineageTree
        An embryo.
    t : int
        A time point.
    min_age : int, default=0
        Cells born less than ``min_age`` time points before ``t`` are left
        out; their contacts are still changing after the division.

    Returns
    -------
    dict of {str: int}
        Maps each valid and unique name to its node.
    """
    out, seen = {}, set()
    for node in lT.time_nodes.get(t, ()):
        name = lT.label.get(node, "")
        if parse_name(name) is None:
            continue
        if name in out:
            seen.add(name)
        start = lT.time[lT.get_chain_of_node(node)[0]]
        if t - start < min_age and lT.predecessor.get(
            lT.get_chain_of_node(node)[0]
        ):
            continue
        out[name] = node
    for name in seen:
        out.pop(name, None)
    return out


def nearest_name(lT, node):
    """Name of a node, or of its closest named ancestor.

    Parameters
    ----------
    lT : LineageTree
        An embryo.
    node : int
        A node.

    Returns
    -------
    str or None
        The first valid name found going back in time from ``node``, or
        None.
    """
    while node is not None:
        name = lT.label.get(node, "")
        if parse_name(name):
            return name
        before = lT.predecessor.get(node)
        node = before[0] if before else None
    return None


def neighbourhood(lT, node):
    """Contacts of a cell, as fractions of its surface, by neighbour name.

    Parameters
    ----------
    lT : LineageTree
        An embryo with contact surfaces (``contact`` property).
    node : int
        A node.

    Returns
    -------
    dict of {str: float}
        Maps the name of each neighbour, ``"outside"`` for the exterior of
        the embryo and ``"unnamed"`` for cells without a valid name, to the
        fraction of the surface of the cell in contact with it. The values
        sum to 1.
    """
    contacts = lT.contact.get(node, {})
    total = sum(contacts.values())
    out = {}
    for other, area in contacts.items():
        if other == -1:
            key = "outside"
        else:
            key = lT.label.get(other, "")
            key = key if parse_name(key) else "unnamed"
        out[key] = out.get(key, 0.0) + area / total
    return out


def _merge_to_ancestors(keys):
    """Map each cell name to its oldest ancestor among a set of names.

    Parameters
    ----------
    keys : set of str
        Cell names, and possibly ``"outside"`` or ``"unnamed"``.

    Returns
    -------
    dict of {str: str}
        Maps each key to the oldest of its ancestors in ``keys`` (itself if
        it has none there); other keys map to themselves.
    """
    named = {k for k in keys if parse_name(k)}
    out = {}
    for key in keys:
        if key not in named:
            out[key] = key
            continue
        _, gen, _, _ = parse_name(key)
        target = key
        for g in range(gen - 1, 0, -1):
            try:
                candidate = ancestor(key, g)
            except (TypeError, ValueError):
                break
            if candidate in named:
                target = candidate
        out[key] = target
    return out


def common_form(first, second):
    """Express two neighbourhoods with the same neighbour names.

    A neighbour that has divided in one embryo and not in the other appears
    as its daughters in the first and as itself in the second. The
    daughters are merged into their mother, in both neighbourhoods: this is
    the virtual common neighbourhood of Biasuz et al. (2025).

    Parameters
    ----------
    first, second : dict of {str: float}
        Two neighbourhoods (see ``neighbourhood``).

    Returns
    -------
    tuple of (dict, dict)
        The two neighbourhoods, with merged names.
    """
    mapping = _merge_to_ancestors(set(first) | set(second))
    merged = []
    for values in (first, second):
        out = {}
        for key, value in values.items():
            out[mapping[key]] = out.get(mapping[key], 0.0) + value
        merged.append(out)
    return tuple(merged)


def neighbourhood_distance(first, second, ignore=()):
    """Fraction of the surface of a cell in contact with different cells.

    Parameters
    ----------
    first, second : dict of {str: float}
        The neighbourhoods of two cells (see ``neighbourhood``), whose
        neighbours are named in the same frame (mirror the names first to
        compare the two sides of an embryo, see ``mirror_neighbourhood``).
    ignore : collection of str, optional
        Neighbours left out of both, e.g. the two sisters when comparing
        sisters. The neighbourhoods are normalised again without them.

    Returns
    -------
    float
        Half the sum of the absolute differences of the two neighbourhoods
        after ``common_form``: 0 when the two cells touch the same
        neighbours with the same fractions of their surface, 1 when they
        share no neighbour.
    """
    first = {k: v for k, v in first.items() if k not in ignore}
    second = {k: v for k, v in second.items() if k not in ignore}
    first, second = common_form(first, second)
    total_1, total_2 = sum(first.values()), sum(second.values())
    keys = set(first) | set(second)
    return 0.5 * sum(
        abs(first.get(k, 0.0) / total_1 - second.get(k, 0.0) / total_2)
        for k in keys
    )


def mirror_neighbourhood(values):
    """Neighbourhood with the names of the other side of the embryo.

    Parameters
    ----------
    values : dict of {str: float}
        A neighbourhood (see ``neighbourhood``).

    Returns
    -------
    dict of {str: float}
        The same values, with each neighbour renamed to its bilateral
        homologue, so that it can be compared with the homologue of the
        cell.
    """
    return {
        (homologue(k) if parse_name(k) else k): v for k, v in values.items()
    }


def life_neighbourhood(lT, first, last, settle=0):
    """Neighbourhood of a cell averaged over its life.

    Parameters
    ----------
    lT : LineageTree
        An embryo with contact surfaces.
    first, last : int
        The first and last nodes of the cell.
    settle : int, default=0
        Number of time points left out after the birth of the cell, while
        its contacts settle.

    Returns
    -------
    dict of {str: float}
        The mean over the time points of the cell of its neighbourhood
        (see ``neighbourhood``). Neighbours that divide during the life of
        the cell appear under their own name and under the names of their
        daughters; ``common_form`` merges them when two cells are compared.
    """
    chain = lT.get_chain_of_node(first)
    chain = chain[min(settle, len(chain) - 1) :]
    total = {}
    for node in chain:
        for key, value in neighbourhood(lT, node).items():
            total[key] = total.get(key, 0.0) + value / len(chain)
    return total


def count_neighbours(values, threshold=0.05):
    """Number of neighbours of a cell.

    Parameters
    ----------
    values : dict of {str: float}
        A neighbourhood (see ``neighbourhood``).
    threshold : float, default=0.05
        A cell is a neighbour if it covers more than this fraction of the
        surface, as in Guignard et al. (2020).

    Returns
    -------
    int
        The number of neighbouring cells, the outside excluded.
    """
    return sum(
        1 for k, v in values.items() if k != "outside" and v > threshold
    )


def positions(lT, t):
    """Positions and volumes of the named cells at a time point.

    Parameters
    ----------
    lT : LineageTree
        An embryo.
    t : int
        A time point.

    Returns
    -------
    dict of {str: tuple of (numpy.ndarray, float)}
        Maps each name to the barycentre and the volume of the cell.
    """
    return {
        name: (np.asarray(lT.pos[node], dtype=float), float(lT.volume[node]))
        for name, node in named_nodes(lT, t).items()
    }


def common_cells(first, second):
    """Express two sets of cells with the same names.

    A cell that has divided in one embryo and not in the other is replaced
    by its mother, at the barycentre of its daughters weighted by their
    volumes.

    Parameters
    ----------
    first, second : dict of {str: tuple of (numpy.ndarray, float)}
        Positions and volumes by name (see ``positions``).

    Returns
    -------
    list of str
        The names present in both, after merging.
    numpy.ndarray
        Their positions in ``first``, one row per name.
    numpy.ndarray
        Their positions in ``second``.
    """
    mapping = _merge_to_ancestors(set(first) | set(second))
    merged = []
    for cells in (first, second):
        sums = {}
        for name, (position, volume) in cells.items():
            key = mapping[name]
            total, weight = sums.get(key, (0.0, 0.0))
            sums[key] = (total + position * volume, weight + volume)
        merged.append({k: s / w for k, (s, w) in sums.items()})
    names = sorted(set(merged[0]) & set(merged[1]))
    return (
        names,
        np.array([merged[0][n] for n in names]),
        np.array([merged[1][n] for n in names]),
    )


def similarity_fit(source, target, reflection=False):
    """Rotation, scaling and translation that best map points onto others.

    The least-squares solution of Umeyama (1991).

    Parameters
    ----------
    source, target : numpy.ndarray
        Matched points, one per row.
    reflection : bool, default=False
        Allow a reflection instead of a rotation, to compare the two sides
        of an embryo.

    Returns
    -------
    numpy.ndarray
        ``source`` mapped onto ``target``.
    float
        The scaling factor.
    numpy.ndarray
        The rotation (or reflection) matrix: a point ``p`` of the source
        maps to ``scale * rotation @ (p - source.mean(0)) + target.mean(0)``.
    """
    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    s, t = source - mu_s, target - mu_t
    u, d, vt = np.linalg.svd(t.T @ s / len(s))
    sign = np.ones(3)
    if (np.linalg.det(u @ vt) < 0) != reflection:
        sign[-1] = -1
    rotation = u @ np.diag(sign) @ vt
    scale = (d * sign).sum() / s.var(axis=0).sum()
    return scale * s @ rotation.T + mu_t, float(scale), rotation


def embryo_radius(lT, t):
    """Radius of the sphere of the volume of an embryo.

    Parameters
    ----------
    lT : LineageTree
        An embryo.
    t : int
        A time point.

    Returns
    -------
    float
        The radius, in voxels.
    """
    volume = sum(lT.volume[n] for n in lT.time_nodes[t])
    return (3 * volume / (4 * np.pi)) ** (1 / 3)


# --------------------------------------------------------------------------
# Timing and volumes
# --------------------------------------------------------------------------


def cycle_lengths(alignment):
    """Length of the cell cycle of each named cell, in reference minutes.

    Parameters
    ----------
    alignment : TimeAlignment
        The alignment of the embryos, with their cell tables.

    Returns
    -------
    pandas.DataFrame
        Indexed by cell name, one column per embryo. A value is given only
        for cells born and divided during the movie, and only if the movie
        covers, with margin, the consensus birth and division of the cell:
        otherwise only the short cycles would be seen near the end.
    """
    births = alignment.consensus("birth")
    divisions = alignment.consensus("division")
    out = {}
    for e, table in alignment.tables.items():
        complete = table.birth.notna() & table.division.notna()
        lengths = (table.frames * alignment.frame_minutes(e)).where(complete)
        seen = alignment.covers(
            e, births.reindex(table.index), divisions.reindex(table.index)
        )
        out[e] = lengths.where(seen)
    return pd.DataFrame(out)


def relative_differences(values):
    """Relative differences of a measure between embryos and between sides.

    Parameters
    ----------
    values : pandas.DataFrame
        A measure, indexed by cell name, one column per embryo.

    Returns
    -------
    pandas.DataFrame
        One row per comparison, with the columns ``comparison``
        (``"between embryos"``: the same cell in two embryos, or
        ``"left/right"``: a cell and its bilateral homologue in the same
        embryo), ``name`` and ``difference``, the absolute difference
        divided by the mean of the two values.
    """
    rows = []
    for name, row in values.iterrows():
        observed = row.dropna()
        for (_, x), (_, y) in _pairs(observed.items()):
            rows.append(("between embryos", name, abs(x - y) / ((x + y) / 2)))
        other = homologue(name)
        if name.endswith("_") and other in values.index:
            both = pd.concat([row, values.loc[other]], axis=1).dropna()
            for x, y in both.values:
                rows.append(("left/right", name, abs(x - y) / ((x + y) / 2)))
    return pd.DataFrame(rows, columns=["comparison", "name", "difference"])


def _pairs(items):
    """All unordered pairs of a sequence.

    Parameters
    ----------
    items : iterable
        The elements.

    Returns
    -------
    list of tuple
        Each pair once.
    """
    items = list(items)
    return [
        (items[i], items[j])
        for i in range(len(items))
        for j in range(i + 1, len(items))
    ]


def side_correlation(values):
    """Correlation of the deviations of the two sides of the same embryos.

    For each cell, the deviation of a measure in an embryo is its value
    minus the mean over the embryos. If the two sides of an embryo deviate
    together, part of the variability is shared by the whole embryo; if
    not, each cell varies on its own.

    Parameters
    ----------
    values : pandas.DataFrame
        A measure, indexed by cell name, one column per embryo.

    Returns
    -------
    float
        The Pearson correlation between the deviations of the left cells
        and of their right homologues, in the same embryos.
    int
        The number of pairs.
    """
    deviations = values.sub(values.mean(axis=1), axis=0)
    pairs = []
    for name in deviations.index:
        other = homologue(name)
        if name.endswith("_") and other in deviations.index:
            both = pd.concat(
                [deviations.loc[name], deviations.loc[other]], axis=1
            ).dropna()
            pairs.extend(both.values.tolist())
    pairs = np.array(pairs)
    return float(np.corrcoef(pairs.T)[0, 1]), len(pairs)


def stage_composition(manager, embryos, sizes):
    """Fraction of the cells shared by the embryos at a given cell number.

    At the N-cell stage of each embryo (its first time point with the
    number of cells closest to N), the names of its cells are compared
    with those of the other embryos at their N-cell stage.

    Parameters
    ----------
    manager : LineageTreeManager
        The embryos.
    embryos : list of str
        The embryos to compare.
    sizes : iterable of int
        The numbers of cells.

    Returns
    -------
    pandas.DataFrame
        One row per number of cells N, with the number of embryos that reach
        it (``embryos``), the median number of cells of their N-cell stage
        (``cells``), and the mean fraction of the cells of an embryo whose
        name is found in all the other embryos (``shared``).
    """
    counts = {e: cell_counts(manager[e]) for e in embryos}
    rows = []
    for size in sizes:
        sets = []
        for e in embryos:
            c = counts[e]
            if c.min() > size or c.max() < size:
                continue
            t = int((c - size).abs().idxmin())
            lT = manager[e]
            sets.append({lT.label.get(n, "") for n in lT.time_nodes[t]})
        if len(sets) < 2:
            continue
        common = set.intersection(*sets)
        rows.append(
            {
                "cells": size,
                "embryos": len(sets),
                "shared": float(np.mean([len(common) / len(x) for x in sets])),
            }
        )
    return pd.DataFrame(rows).set_index("cells")


def sister_asynchrony(alignment):
    """Difference between the division times of sister cells.

    Parameters
    ----------
    alignment : TimeAlignment
        The alignment of the embryos, with their cell tables.

    Returns
    -------
    pandas.DataFrame
        One row per pair of sisters that both divide in an embryo, with the
        columns ``mother`` (the name of the mother without its side),
        ``side``, ``embryo`` and ``difference``: the division time of the
        daughter with the odd index minus that of its sister, in reference
        minutes. Pairs whose consensus division times are not covered by
        the movie are left out.
    """
    divisions = alignment.consensus("division")
    rows = []
    for e, table in alignment.tables.items():
        for name in table.index:
            if parse_name(name)[2] % 2 == 0 or sister(name) not in table.index:
                continue
            other = sister(name)
            a, b = table.at[name, "division"], table.at[other, "division"]
            late = max(
                divisions.get(name, np.nan), divisions.get(other, np.nan)
            )
            if np.isnan(a) or np.isnan(b) or not alignment.covers(e, late):
                continue
            rows.append(
                {
                    "mother": mother(name)[:-1],
                    "side": name[-1],
                    "embryo": e,
                    "difference": (a - b) * alignment.frame_minutes(e),
                }
            )
    return pd.DataFrame(rows)


def sister_volume_ratios(tables):
    """Volume ratio of the two daughters of each division.

    Parameters
    ----------
    tables : dict of {str: pandas.DataFrame}
        The cell tables of the embryos, by embryo name.

    Returns
    -------
    pandas.DataFrame
        One row per pair of sisters in an embryo, with the columns
        ``mother`` (without its side), ``side``, ``embryo`` and
        ``log2_ratio``: the base-2 logarithm of the median volume of the
        daughter with the odd index divided by that of its sister.
    """
    rows = []
    for e, table in tables.items():
        for name in table.index:
            if parse_name(name)[2] % 2 == 0 or sister(name) not in table.index:
                continue
            ratio = table.at[name, "volume"] / table.at[sister(name), "volume"]
            rows.append(
                {
                    "mother": mother(name)[:-1],
                    "side": name[-1],
                    "embryo": e,
                    "log2_ratio": float(np.log2(ratio)),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Divisions, positions and contacts
# --------------------------------------------------------------------------


def node_before(lT, chain_last, frames):
    """Node of a cell a number of time points before its last one.

    Parameters
    ----------
    lT : LineageTree
        An embryo.
    chain_last : int
        The last node of the cell.
    frames : int
        How many time points to go back; the first node of the cell is
        returned if it is shorter.

    Returns
    -------
    int
        The node.
    """
    chain = lT.get_chain_of_node(chain_last)
    return chain[max(0, len(chain) - 1 - frames)]


def division_table(lT, table, minutes_per_frame, after=2.0, before=6.0):
    """Orientation of the divisions of the named cells of an embryo.

    Parameters
    ----------
    lT : LineageTree
        An embryo with principal axes (``cell_principal_vectors``).
    table : pandas.DataFrame
        Its cell table (see ``cell_table``).
    minutes_per_frame : float
        The duration of a time point, in reference minutes.
    after : float, default=2.0
        The daughters are located this many minutes after the division
        (at least one time point after it).
    before : float, default=6.0
        The shape of the mother is taken this many minutes before its
        division, before it rounds up for mitosis.

    Returns
    -------
    pandas.DataFrame
        Indexed by the name of the mother, with the time point of the
        division (``frame``), the unit vector from the daughter with the
        even index to the daughter with the odd index (``x``, ``y``, ``z``),
        and ``hertwig``: the absolute cosine between this vector and the
        longest axis of the mother. Hertwig's rule predicts values close
        to 1.
    """
    rows = {}
    k_after = max(1, round(after / minutes_per_frame))
    k_before = max(1, round(before / minutes_per_frame))
    for name, row in table.iterrows():
        if np.isnan(row.division):
            continue
        odd, even = daughters(name)
        if odd not in table.index or even not in table.index:
            continue
        points = []
        for d in (odd, even):
            chain = lT.get_chain_of_node(table.at[d, "first"])
            points.append(
                np.asarray(lT.pos[chain[min(k_after, len(chain) - 1)]])
            )
        vector = points[0] - points[1]
        vector = vector / np.linalg.norm(vector)
        shape = node_before(lT, row["last"], k_before)
        axis = np.asarray(lT.cell_principal_vectors[shape])[0]
        rows[name] = {
            "frame": int(row.division),
            "x": vector[0],
            "y": vector[1],
            "z": vector[2],
            "hertwig": float(abs(vector @ axis / np.linalg.norm(axis))),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def rotation_to(manager, embryo, t, reference, t_reference):
    """Rotation that brings an embryo onto a reference embryo.

    The two embryos are registered on their common named cells (see
    ``common_cells`` and ``similarity_fit``).

    Parameters
    ----------
    manager : LineageTreeManager
        The embryos.
    embryo : str
        The embryo to rotate.
    t : int
        Its time point.
    reference : str
        The reference embryo.
    t_reference : int
        The matching time point of the reference.

    Returns
    -------
    numpy.ndarray
        The 3 x 3 rotation matrix.
    """
    _, target, source = common_cells(
        positions(manager[reference], t_reference),
        positions(manager[embryo], t),
    )
    return similarity_fit(source, target)[2]


def position_deviations(manager, alignment, reference_time, embryos):
    """Distances between the positions of matched cells after registration.

    At a reference time, each pair of embryos is registered on their common
    cells by rotation, scaling and translation, and each embryo is
    registered with its own mirror image, with the names of the two sides
    swapped. The distances are in typical cell radii: the radius of the
    embryo divided by the cube root of its number of cells.

    Parameters
    ----------
    manager : LineageTreeManager
        The embryos.
    alignment : TimeAlignment
        Their alignment.
    reference_time : float
        The reference time point.
    embryos : list of str
        The embryos to compare.

    Returns
    -------
    pandas.DataFrame
        One row per cell and comparison, with the columns ``comparison``
        (``"between embryos"`` or ``"left/right"``), ``name``, ``embryo``,
        ``other`` and ``distance``.
    """
    frames = {e: frame_at(alignment, e, reference_time) for e in embryos}
    frames = {e: t for e, t in frames.items() if t is not None}
    cells = {e: positions(manager[e], t) for e, t in frames.items()}
    unit = {
        e: embryo_radius(manager[e], t)
        / len(manager[e].time_nodes[t]) ** (1 / 3)
        for e, t in frames.items()
    }
    rows = []
    for e, f in _pairs(frames):
        names, x, y = common_cells(cells[e], cells[f])
        mapped = similarity_fit(y, x)[0]
        distance = np.linalg.norm(mapped - x, axis=1) / unit[e]
        rows += [
            ("between embryos", n, e, f, d)
            for n, d in zip(names, distance, strict=True)
        ]
    for e in frames:
        mirror = {homologue(n): v for n, v in cells[e].items()}
        names, x, y = common_cells(cells[e], mirror)
        mapped = similarity_fit(y, x, reflection=True)[0]
        distance = np.linalg.norm(mapped - x, axis=1) / unit[e]
        rows += [
            ("left/right", n, e, e, d)
            for n, d in zip(names, distance, strict=True)
            if n.endswith("_")
        ]
    return pd.DataFrame(
        rows, columns=["comparison", "name", "embryo", "other", "distance"]
    )


def neighbourhoods_at(manager, alignment, reference_time, embryos, settle=8.0):
    """Neighbourhoods of the named cells of several embryos at a matched time.

    Parameters
    ----------
    manager : LineageTreeManager
        The embryos.
    alignment : TimeAlignment
        Their alignment.
    reference_time : float
        The reference time point.
    embryos : list of str
        The embryos.
    settle : float, default=8.0
        Cells born less than this many reference minutes before are left
        out: their contacts are still settling after the division (about
        four time points, Biasuz et al. 2025).

    Returns
    -------
    dict of {str: dict of {str: dict}}
        For each embryo whose movie covers the time, the neighbourhood (see
        ``neighbourhood``) of each of its named cells.
    """
    out = {}
    for e in embryos:
        t = frame_at(alignment, e, reference_time)
        if t is None:
            continue
        lT = manager[e]
        age = int(np.ceil(settle / alignment.frame_minutes(e)))
        out[e] = {
            name: neighbourhood(lT, node)
            for name, node in named_nodes(lT, t, min_age=age).items()
        }
    return out


def neighbourhood_comparisons(neighbourhoods):
    """Neighbourhood distances between embryos, sides and sisters.

    Parameters
    ----------
    neighbourhoods : dict of {str: dict of {str: dict}}
        The neighbourhoods of the cells of each embryo, from
        ``neighbourhoods_at``.

    Returns
    -------
    pandas.DataFrame
        One row per comparison, with the columns ``comparison``:
        ``"between embryos"`` (the same cell in two embryos),
        ``"left/right"`` (a cell and its homologue, neighbours renamed to
        the other side) or ``"sisters"`` (two sisters, their mutual contact
        left out); ``name``, ``embryo``, ``other`` and ``distance`` (see
        ``neighbourhood_distance``).
    """
    rows = []
    for e, f in _pairs(neighbourhoods):
        for name in set(neighbourhoods[e]) & set(neighbourhoods[f]):
            d = neighbourhood_distance(
                neighbourhoods[e][name], neighbourhoods[f][name]
            )
            rows.append(("between embryos", name, e, f, d))
    for e, cells in neighbourhoods.items():
        for name, values in cells.items():
            other = homologue(name)
            if name.endswith("_") and other in cells:
                d = neighbourhood_distance(
                    mirror_neighbourhood(values), cells[other]
                )
                rows.append(("left/right", name, e, e, d))
            pair = sister(name)
            if parse_name(name)[2] % 2 and pair in cells:
                d = neighbourhood_distance(
                    values, cells[pair], ignore={name, pair}
                )
                rows.append(("sisters", name, e, e, d))
    return pd.DataFrame(
        rows, columns=["comparison", "name", "embryo", "other", "distance"]
    )


def contact_variability(neighbourhoods, min_embryos=4):
    """Variability of each contact between embryos.

    The neighbourhoods of a cell in all the embryos are first written with
    the same neighbour names (daughters merged into their mother when the
    embryos differ, see ``common_form``).

    Parameters
    ----------
    neighbourhoods : dict of {str: dict of {str: dict}}
        The neighbourhoods of the cells of each embryo, from
        ``neighbourhoods_at``.
    min_embryos : int, default=4
        Cells found in fewer embryos are left out.

    Returns
    -------
    pandas.DataFrame
        One row per cell and neighbour (the outside included), with the
        mean fraction of the surface of the cell in contact with the
        neighbour (``mean``), its standard deviation (``sd``), and the
        fraction of the embryos in which the contact exists (``present``).
    """
    rows = []
    names = set().union(*[set(c) for c in neighbourhoods.values()])
    for name in names:
        found = [c[name] for c in neighbourhoods.values() if name in c]
        if len(found) < min_embryos:
            continue
        mapping = _merge_to_ancestors(set().union(*[set(v) for v in found]))
        merged = []
        for values in found:
            out = {}
            for key, value in values.items():
                out[mapping[key]] = out.get(mapping[key], 0.0) + value
            merged.append(out)
        for key in set().union(*[set(m) for m in merged]) - {"unnamed"}:
            x = np.array([m.get(key, 0.0) for m in merged])
            rows.append(
                {
                    "name": name,
                    "neighbour": key,
                    "mean": x.mean(),
                    "sd": x.std(ddof=1),
                    "present": float((x > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def shared_neighbours(first, second, threshold=0.05):
    """Neighbours shared by two cells, and the change of their contacts.

    The two measures of Guignard et al. (2020): the fraction of shared
    neighbours, and the mean relative change of the contact areas with the
    shared neighbours.

    Parameters
    ----------
    first, second : dict of {str: float}
        The neighbourhoods of the two cells, with neighbour names in the
        same frame (see ``neighbourhood_distance``).
    threshold : float, default=0.05
        A cell is a neighbour if it covers more than this fraction of the
        surface.

    Returns
    -------
    float
        The fraction of shared neighbours: twice the number of neighbours of
        both cells divided by the sum of their numbers of neighbours.
    float
        The mean, over the shared neighbours, of the absolute difference of
        the two contact fractions divided by their mean.
    """
    first, second = common_form(first, second)
    n1 = {k for k, v in first.items() if v > threshold and k != "outside"}
    n2 = {k for k, v in second.items() if v > threshold and k != "outside"}
    shared = n1 & n2
    if not n1 or not n2:
        return np.nan, np.nan
    change = [
        abs(first[k] - second[k]) / ((first[k] + second[k]) / 2)
        for k in shared
    ]
    return 2 * len(shared) / (len(n1) + len(n2)), (
        float(np.mean(change)) if change else np.nan
    )


def life_comparisons(manager, alignment, embryos, settle=8.0):
    """Neighbourhoods of the same cells over their whole lives, two by two.

    Parameters
    ----------
    manager : LineageTreeManager
        The embryos.
    alignment : TimeAlignment
        Their alignment, with their cell tables.
    embryos : list of str
        The embryos to compare.
    settle : float, default=8.0
        Reference minutes left out after the birth of each cell.

    Returns
    -------
    pandas.DataFrame
        One row per cell born and divided in two embryos, with the columns
        ``name``, ``embryo``, ``other``, ``distance`` (see
        ``neighbourhood_distance``), ``shared`` and ``change`` (see
        ``shared_neighbours``).
    """
    lives = {}
    for e in embryos:
        table = alignment.tables[e]
        complete = table[table.birth.notna() & table.division.notna()]
        skip = int(np.ceil(settle / alignment.frame_minutes(e)))
        lives[e] = {
            name: life_neighbourhood(
                manager[e], row["first"], row["last"], skip
            )
            for name, row in complete.iterrows()
        }
    rows = []
    for e, f in _pairs(embryos):
        for name in sorted(set(lives[e]) & set(lives[f])):
            shared, change = shared_neighbours(lives[e][name], lives[f][name])
            distance = neighbourhood_distance(lives[e][name], lives[f][name])
            rows.append((name, e, f, distance, shared, change))
    return pd.DataFrame(
        rows,
        columns=["name", "embryo", "other", "distance", "shared", "change"],
    )


def exterior_by_tissue(manager, alignment, tissues, embryos):
    """Fraction of the surface of each tissue facing the outside of the embryo.

    Parameters
    ----------
    manager : LineageTreeManager
        The embryos, with contact surfaces.
    alignment : TimeAlignment
        Their alignment.
    tissues : pandas.Series
        The tissue of each cell name (see ``tissue_of``).
    embryos : list of str
        The embryos.

    Returns
    -------
    pandas.DataFrame
        One row per embryo, time point and tissue, with the minutes post
        fertilization (``minutes``) and the fraction of the total surface of
        the cells of the tissue in contact with the outside (``exterior``).
    """
    rows = []
    for e in embryos:
        lT = manager[e]
        for t, nodes in sorted(lT.time_nodes.items()):
            sums = {}
            for node in nodes:
                tissue = tissues.get(lT.label.get(node, ""), None)
                if tissue is None:
                    continue
                outside, surface = sums.get(tissue, (0.0, 0.0))
                sums[tissue] = (
                    outside + lT.contact[node].get(-1, 0.0),
                    surface + sum(lT.contact[node].values()),
                )
            for tissue, (outside, surface) in sums.items():
                rows.append(
                    {
                        "embryo": e,
                        "t": t,
                        "minutes": alignment.minutes(e, t),
                        "tissue": tissue,
                        "exterior": outside / surface,
                    }
                )
    return pd.DataFrame(rows)


def curve_shifts(curves, shifts=None):
    """Time shift of each curve that best matches the median curve.

    Parameters
    ----------
    curves : dict of {str: pandas.Series}
        One curve per embryo, indexed by minutes post fertilization.
    shifts : numpy.ndarray, optional
        The shifts to try, in minutes; -30 to 30 by steps of 0.5 by
        default.

    Returns
    -------
    pandas.Series
        For each embryo, the shift, in minutes, that minimises the mean
        squared difference between its shifted curve and the median of the
        curves (positive: the embryo is late).
    """
    shifts = np.arange(-30, 30.5, 0.5) if shifts is None else shifts
    low = max(c.index.min() for c in curves.values())
    high = min(c.index.max() for c in curves.values())
    grid = np.arange(np.ceil(low), np.floor(high) + 1, 1.0)
    reference = np.median(
        [np.interp(grid, c.index, c.values) for c in curves.values()], axis=0
    )
    out = {}
    for name, c in curves.items():
        costs = []
        for shift in shifts:
            x = grid + shift
            inside = (x >= c.index.min()) & (x <= c.index.max())
            if inside.sum() < 10:
                costs.append(np.inf)
                continue
            y = np.interp(x[inside], c.index, c.values)
            costs.append(np.mean((y - reference[inside]) ** 2))
        out[name] = float(shifts[int(np.argmin(costs))])
    return pd.Series(out)


# --------------------------------------------------------------------------
# Lineage trees
# --------------------------------------------------------------------------

SCALE = 100


def set_time_resolutions(manager, alignment, embryos, scale=SCALE):
    """Give each embryo the duration of its time points, from the alignment.

    ``LineageTreeManager`` multiplies the lengths of the cells by the time
    resolution of their tree before comparing them, which is how the
    linear alignment enters the tree edit distance. The resolution is
    stored as a whole number of tenths, so the minutes per time point are
    multiplied by ``scale`` to keep their precision; the scale cancels out
    in normalised distances.

    Parameters
    ----------
    manager : LineageTreeManager
        The embryos.
    alignment : TimeAlignment
        Their alignment.
    embryos : list of str
        The embryos to update.
    scale : float, default=SCALE
        Factor applied to the minutes per time point.
    """
    for e in embryos:
        manager[e].time_resolution = alignment.frame_minutes(e) * scale


def lineage_distances(
    manager,
    alignment,
    start,
    end,
    embryos,
    style="simple",
    sides=True,
    frames_only=False,
):
    """Tree edit distances between the lineages of the same cells.

    The lineage of each cell alive at reference time ``start`` is followed
    until reference time ``end`` in each embryo, and compared with the
    lineage of the same cell in the other embryos and, within each embryo,
    with the lineage of its bilateral homologue.

    Parameters
    ----------
    manager : LineageTreeManager
        The embryos, with time resolutions (see ``set_time_resolutions``).
    alignment : TimeAlignment
        Their alignment.
    start, end : float
        The reference time points that bound the lineages.
    embryos : list of str
        The embryos; their movies must cover ``start`` and ``end``.
    style : str or type, default="simple"
        The tree style of ``cross_lineage_edit_distance``.
    sides : bool, default=True
        Also compare the two sides of each embryo.
    frames_only : bool, default=False
        Ignore the different paces of the embryos: each lineage is followed
        for as many time points as the reference takes from ``start`` to
        ``end``. Use it with equal time resolutions for all the embryos, to
        see what the alignment changes.

    Returns
    -------
    pandas.DataFrame
        One row per comparison, with the columns ``comparison``
        (``"between embryos"`` or ``"left/right"``), ``name``, ``embryo``,
        ``other`` and ``distance``, the distance normalised by the larger
        of the two trees.
    """
    roots, ends = {}, {}
    for e in embryos:
        t0, t1 = frame_at(alignment, e, start), frame_at(alignment, e, end)
        if frames_only:
            t1 = t0 + round(end - start)
        roots[e], ends[e] = named_nodes(manager[e], t0), t1
    rows = []
    for e, f in _pairs(embryos):
        for name in sorted(set(roots[e]) & set(roots[f])):
            d = manager.cross_lineage_edit_distance(
                roots[e][name],
                e,
                roots[f][name],
                f,
                ends[e],
                ends[f],
                style=style,
            )
            rows.append(("between embryos", name, e, f, d))
        manager.clear_comparisons()
    if sides:
        for e in embryos:
            for name in sorted(roots[e]):
                other = homologue(name)
                if name.endswith("_") and other in roots[e]:
                    d = manager.cross_lineage_edit_distance(
                        roots[e][name],
                        e,
                        roots[e][other],
                        e,
                        ends[e],
                        ends[e],
                        style=style,
                    )
                    rows.append(("left/right", name, e, e, d))
            manager.clear_comparisons()
    return pd.DataFrame(
        rows, columns=["comparison", "name", "embryo", "other", "distance"]
    )


def sister_divergence(
    manager, alignment, embryo, duration=None, style="simple"
):
    """Tree edit distance between the lineages of the two daughters of cells.

    As in Guignard et al. (2020), a division whose two daughters make
    different lineages is asymmetric.

    Parameters
    ----------
    manager : LineageTreeManager
        The embryos, with time resolutions (see ``set_time_resolutions``).
    alignment : TimeAlignment
        Their alignment.
    embryo : str
        The embryo.
    duration : float, optional
        The lineages are followed for this many reference minutes after the
        division; divisions too late for that are left out. By default,
        they are followed until the end of the movie.
    style : str or type, default="simple"
        The tree style of ``cross_lineage_edit_distance``.

    Returns
    -------
    pandas.DataFrame
        Indexed by the name of the mother, with the columns ``embryo``,
        ``gen`` (generation of the mother), ``minutes`` (time of the
        division, in minutes post fertilization) and ``distance``.
    """
    table = alignment.tables[embryo]
    last = alignment.frames[embryo][1]
    rows = {}
    for name in table.index:
        if parse_name(name)[2] % 2 == 0 or sister(name) not in table.index:
            continue
        birth = table.at[name, "birth"]
        if np.isnan(birth):
            continue
        if duration is None:
            end = last
        else:
            end = birth + duration / alignment.frame_minutes(embryo)
            if end > last:
                continue
        d = manager.cross_lineage_edit_distance(
            table.at[name, "first"],
            embryo,
            table.at[sister(name), "first"],
            embryo,
            end,
            end,
            style=style,
        )
        rows[mother(name)] = {
            "embryo": embryo,
            "gen": parse_name(name)[1] - 1,
            "minutes": alignment.minutes(embryo, birth),
            "distance": d,
        }
    manager.clear_comparisons()
    return pd.DataFrame.from_dict(rows, orient="index")


def add_log_volume(lT):
    """Store the relative volume of each node, for ``SimpleWithVolume``.

    Parameters
    ----------
    lT : LineageTree
        An embryo. The property ``log_volume`` is added: the base-2
        logarithm of the volume of each node divided by the volume of the
        embryo at its time point.
    """
    total = embryo_volume(lT)
    lT.log_volume = {
        n: float(np.log2(lT.volume[n] / total[lT.time[n]])) for n in lT.nodes
    }


class SimpleWithVolume(simple_tree):
    """The ``simple`` tree style, with the volumes of the cells in the costs.

    Each cell of a lineage becomes one node, weighted by its length, as in
    ``simple_tree``, and carries the median of the ``log_volume`` property
    over its nodes (see ``add_log_volume``). For two cells of lengths
    ``l1`` and ``l2`` and log volumes ``v1`` and ``v2``:

    - matching them costs ``|l1 - l2| + weight * min(l1, l2) * |v1 - v2|``:
      a two-fold difference of volume during the whole life of the cells
      costs ``weight`` times their length;
    - inserting or deleting a cell costs its length.

    The norm of a tree, which normalises the distance, is the sum of the
    lengths of its cells.

    Attributes
    ----------
    weight : float
        Cost of a two-fold difference of volume during one time point,
        relative to that of one time point of difference in length.
    """

    weight = 1.0

    def get_tree(self):
        """Build the tree of ``simple_tree``, with the volume of each cell.

        Returns
        -------
        dict of {int: list of int}
            Adjacency dictionary, keyed by the first node of each chain.
        dict of {int: tuple of (float, float)}
            Maps the first node of each chain to its length (times
            ``time_scale``) and to its median log volume. ``simple_tree``
            also lists, with a length of 0, the daughters of a cell that
            divides just before ``end_time``; their volume is set to 0,
            and costs nothing since their length is 0.
        """
        tree, lengths = super().get_tree()
        self.times = {}
        for first, length in lengths.items():
            volumes = [
                self.lT.log_volume[n]
                for n in self.lT.get_successors(first)
                if self.lT.time[n] <= self.end_time
            ]
            volume = float(np.median(volumes)) if volumes else 0.0
            self.times[first] = (length, volume)
        return tree, self.times

    def delta(self, x, y, corres1, corres2, times1, times2):
        """Cost of matching two cells, or of inserting or deleting one.

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
            The ``(length, log volume)`` of each cell, from ``get_tree``.

        Returns
        -------
        float
            The cost of the edit.
        """
        if x is None and y is None:
            return 0
        if x is None:
            return times2[corres2[y]][0]
        if y is None:
            return times1[corres1[x]][0]
        (l1, v1), (l2, v2) = times1[corres1[x]], times2[corres2[y]]
        return abs(l1 - l2) + self.weight * min(l1, l2) * abs(v1 - v2)

    def get_norm(self, root):
        """Sum of the lengths of the cells of the tree.

        Parameters
        ----------
        root : int
            The root of the tree (unused: the tree is the one of this
            object).

        Returns
        -------
        float
            The cost of deleting the whole tree.
        """
        return sum(length for length, _ in self.times.values())


# --------------------------------------------------------------------------
# Quality of the names
# --------------------------------------------------------------------------


def naming_report(lT):
    """Checks of the names of an embryo.

    Parameters
    ----------
    lT : LineageTree
        An embryo.

    Returns
    -------
    dict
        ``named``: the fraction of the nodes with a valid name;
        ``fully named until``: the last time point up to which all the
        cells are named; ``several names``: the number of cells whose name
        changes along their life; ``duplicates``: the number of names
        carried by two cells at the same time; ``bad daughters``: the
        number of daughters whose name does not follow from that of their
        mother.
    """
    valid = {n for n in lT.nodes if parse_name(lT.label.get(n, ""))}
    fully = None
    for t in sorted(lT.time_nodes):
        if all(n in valid for n in lT.time_nodes[t]):
            fully = t
        else:
            break
    several = sum(
        1 for c in lT.all_chains if len({lT.label.get(n, "") for n in c}) > 1
    )
    duplicates = 0
    for nodes in lT.time_nodes.values():
        names = [lT.label[n] for n in nodes if n in valid]
        duplicates += len(names) - len(set(names))
    bad = 0
    for chain in lT.all_chains:
        first = chain[0]
        before = lT.predecessor.get(first)
        if before and first in valid and before[0] in valid:
            bad += mother(lT.label[first]) != lT.label[before[0]]
    return {
        "named": len(valid) / len(lT.nodes),
        "fully named until": fully,
        "several names": several,
        "duplicates": duplicates,
        "bad daughters": bad,
    }


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def embryo_snapshots(lT, times, tissues, title=""):
    """3D views of an embryo at several time points, coloured by tissue.

    Parameters
    ----------
    lT : LineageTree
        An embryo.
    times : list of int
        The time points to show.
    tissues : pandas.Series
        The tissue of each cell name (see ``tissue_of``).
    title : str, optional
        Title of the figure.

    Returns
    -------
    plotly.graph_objects.Figure
        One 3D panel per time point; the size of each cell follows its
        volume. Unnamed cells take the tissue of their closest named
        ancestor.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=1,
        cols=len(times),
        specs=[[{"type": "scene"}] * len(times)],
        subplot_titles=[f"{len(lT.time_nodes[t])} cells" for t in times],
    )
    shown = set()
    for column, t in enumerate(times, 1):
        nodes = list(lT.time_nodes[t])
        xyz = np.array([lT.pos[n] for n in nodes])
        xyz = xyz - xyz.mean(axis=0)
        names = [nearest_name(lT, n) for n in nodes]
        group = [tissues.get(name, "unknown") for name in names]
        size = np.array([lT.volume[n] for n in nodes]) ** (1 / 3)
        for tissue in sorted(set(group)):
            keep = [i for i, g in enumerate(group) if g == tissue]
            figure.add_trace(
                go.Scatter3d(
                    x=xyz[keep, 0],
                    y=xyz[keep, 1],
                    z=xyz[keep, 2],
                    mode="markers",
                    name=tissue,
                    legendgroup=tissue,
                    showlegend=tissue not in shown,
                    text=[conklin(names[i]) if names[i] else "" for i in keep],
                    marker=dict(
                        size=size[keep] / size.max() * 14,
                        color=TISSUE_COLORS.get(tissue, "#cccccc"),
                        line=dict(width=0),
                        opacity=0.9,
                    ),
                ),
                row=1,
                col=column,
            )
            shown.add(tissue)
    axis = dict(visible=False)
    figure.update_scenes(xaxis=axis, yaxis=axis, zaxis=axis, aspectmode="data")
    figure.update_layout(
        title=title, height=420, width=1100, margin=dict(l=0, r=0, t=60, b=0)
    )
    return figure


# --------------------------------------------------------------------------
# Recap figures
# --------------------------------------------------------------------------

_TIMES = "\N{MULTIPLICATION SIGN}"
_DASH = "\N{EN DASH}"
RECAP_COLORS = {
    "between embryos": "#2a78d6",
    "left/right": "#eb6834",
    "reference": "#a8a7a1",
}
_INK = {
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "empty": "#f0efec",
}


def _recap_style():
    """Matplotlib settings of the recap figures.

    Returns
    -------
    dict
        Settings for ``matplotlib.pyplot.rc_context``: the first sans-serif
        font found among Arial, Helvetica and DejaVu Sans, recessive axes
        and a horizontal hairline grid.
    """
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    font = next(
        (f for f in ("Arial", "Helvetica", "DejaVu Sans") if f in available),
        "sans-serif",
    )
    return {
        "font.family": [font],
        "font.size": 11,
        "axes.edgecolor": _INK["axis"],
        "axes.labelcolor": _INK["secondary"],
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": _INK["grid"],
        "grid.linewidth": 0.8,
        "xtick.color": _INK["axis"],
        "ytick.color": _INK["axis"],
        "xtick.labelcolor": _INK["muted"],
        "ytick.labelcolor": _INK["muted"],
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }


def _panel_title(ax, title, subtitle):
    """Write a title and a subtitle above a panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The panel.
    title : str
        What the panel shows.
    subtitle : str
        The numbers to remember from it, on one or two lines.
    """
    lines = subtitle.count("\n") + 1
    ax.annotate(
        subtitle,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 8),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=10.5,
        linespacing=1.3,
        color=_INK["secondary"],
    )
    ax.annotate(
        title,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 12 + 14 * lines),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=_INK["primary"],
    )


def _cumulative(ax, samples, xmax, percent=True):
    """Draw cumulative distributions, with a dot at each median.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The panel.
    samples : dict
        Colour of each distribution -> its values. Drawn in this order.
    xmax : float
        Right end of the horizontal axis; larger values leave the panel.
    percent : bool, optional
        Whether to write the horizontal ticks as percentages.
    """
    from matplotlib.ticker import FuncFormatter

    for color, values in samples.items():
        values = np.sort(np.asarray(values, dtype=float))
        values = values[~np.isnan(values)]
        share = np.arange(1, len(values) + 1) / len(values)
        ax.step(np.r_[0, values], np.r_[0, share], where="post", color=color)
        ax.plot(
            np.median(values),
            0.5,
            "o",
            ms=7,
            color=color,
            mec="white",
            mew=1.5,
            zorder=4,
        )
    ax.set(xlim=(0, xmax), ylim=(0, 1.02))
    ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
    if percent:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))


def _random_orientation_angle(embryos=7, draws=4000, seed=0):
    """Median angle of random directions to their mean direction.

    Parameters
    ----------
    embryos : int, optional
        Number of directions averaged, as for one division seen in that
        many embryos.
    draws : int, optional
        Number of random sets.
    seed : int, optional
        Seed of the random generator.

    Returns
    -------
    float
        Median, over the sets, of the median angle (degrees) between each
        random unit vector and the mean direction of its set.
    """
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(draws, embryos, 3))
    vectors /= np.linalg.norm(vectors, axis=2, keepdims=True)
    mean = vectors.mean(axis=1, keepdims=True)
    mean /= np.linalg.norm(mean, axis=2, keepdims=True)
    cosines = np.clip((vectors * mean).sum(axis=2), -1, 1)
    return float(np.median(np.median(np.degrees(np.arccos(cosines)), axis=1)))


def _figure_heading(figure, title, keys=(), deck=""):
    """Write the title of a figure, with a legend or a line under it.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure.
    title : str
        Its message.
    keys : sequence of (str, str), optional
        Label and colour of each series shared by the panels.
    deck : str, optional
        A sentence under the title, when there are no keys.
    """
    import matplotlib.pyplot as plt

    figure.text(
        0.05,
        0.965,
        title,
        fontsize=18,
        fontweight="bold",
        color=_INK["primary"],
        va="top",
    )
    if deck:
        figure.text(
            0.05,
            0.905,
            deck,
            fontsize=12,
            color=_INK["secondary"],
            va="center",
        )
    renderer = figure.canvas.get_renderer()
    x = 0.05
    for label, color in keys:
        figure.add_artist(
            plt.Line2D(
                [x, x + 0.02],
                [0.905, 0.905],
                transform=figure.transFigure,
                color=color,
                lw=3,
            )
        )
        text = figure.text(
            x + 0.026,
            0.905,
            label,
            va="center",
            fontsize=12,
            color=_INK["secondary"],
        )
        box = text.get_window_extent(renderer)
        x = figure.transFigure.inverted().transform((box.x1, 0))[0] + 0.03


def recap_variability(
    counts,
    alignment,
    embryos,
    cycles,
    volumes,
    orientation,
    positions,
    contacts,
    gastrulation,
    lineages,
    naive,
    until=400,
):
    """One slide summarising how much the same cell varies between embryos.

    Eight panels, one per property. Where it can be measured, the same cell
    in two embryos is compared with the same cell on the two sides of one
    embryo.

    Parameters
    ----------
    counts : dict
        Embryo name -> number of cells at each time point (``cell_counts``).
    alignment : TimeAlignment
        The time alignment of the embryos.
    embryos : list of str
        The embryos compared by name.
    cycles, volumes : pandas.DataFrame
        Relative differences of cycle length and of volume
        (``relative_differences``), with columns ``comparison`` and
        ``difference``.
    orientation : pandas.DataFrame
        One row per division with its ``median angle`` (degrees) to the mean
        orientation over the embryos and the number of ``embryos``.
    positions, contacts : pandas.DataFrame
        Distances between matched cells after registration
        (``position_deviations``) and neighbourhood distances
        (``neighbourhood_comparisons``), with columns ``comparison``,
        ``minutes`` and ``distance``.
    gastrulation : pandas.DataFrame
        Shift (minutes) of the internalisation curve of each tissue in each
        embryo (``curve_shifts``), embryos as rows and tissues as columns.
    lineages, naive : pandas.DataFrame
        Normalised UTED between the lineages of the same cells
        (``lineage_distances``), with and without the time alignment.
    until : float, optional
        Last reference minute shown for positions and contacts, while at
        least two embryos are still filmed.

    Returns
    -------
    matplotlib.figure.Figure
        A 16:9 figure (13.33 x 7.5 inches) for a slide.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    blue = RECAP_COLORS["between embryos"]
    orange = RECAP_COLORS["left/right"]
    gray = RECAP_COLORS["reference"]
    with plt.rc_context(_recap_style()):
        figure, axes = plt.subplots(2, 4, figsize=(13.33, 7.5))
        figure.subplots_adjust(
            left=0.05,
            right=0.98,
            bottom=0.08,
            top=0.78,
            wspace=0.38,
            hspace=0.68,
        )

        # the clock: one linear change of time per embryo
        ax = axes[0, 0]
        for e in embryos:
            c = counts[e]
            frames = c.index.values
            ax.plot(
                alignment.start_minutes + alignment.interval * frames,
                c.values,
                color=gray,
                lw=1.2,
            )
            ax.plot(
                alignment.minutes(e, frames),
                c.values,
                color=_INK["primary"],
                lw=1.2,
            )
        births = alignment.deviations("birth")[embryos].abs().median()
        slopes = [alignment.slope[e] for e in embryos]
        ax.set(
            yscale="log",
            xlim=(240, until + 20),
            ylim=(60, 800),
            xlabel="minutes post fertilization",
            ylabel="cells",
        )
        ax.set_yticks(
            [64, 100, 200, 400, 800], ["64", "100", "200", "400", "800"]
        )
        ax.minorticks_off()
        ax.text(
            0.97,
            0.05,
            "not aligned",
            transform=ax.transAxes,
            ha="right",
            color=_INK["muted"],
            fontsize=10,
        )
        ax.text(
            0.05,
            0.9,
            "aligned",
            transform=ax.transAxes,
            color=_INK["primary"],
            fontsize=10,
        )
        _panel_title(
            ax,
            "Developmental speed",
            f"speed {min(slopes) / max(slopes):.2f}{_DASH}1{_TIMES}; births "
            f"±{births.min():.0f}{_DASH}{births.max():.0f} min once aligned",
        )

        # cycle length and volume of the same cell
        for ax, values, title, xmax in (
            (axes[0, 1], cycles, "Cell cycle length", 0.25),
            (axes[0, 2], volumes, "Cell volume", 0.5),
        ):
            samples = {
                blue: values[
                    values.comparison == "between embryos"
                ].difference,
                orange: values[values.comparison == "left/right"].difference,
            }
            _cumulative(ax, samples, xmax)
            ax.set(xlabel="difference", ylabel="cells (cumulative)")
            m = [np.median(s) for s in samples.values()]
            _panel_title(
                ax, title, f"median {m[0]:.1%} between, {m[1]:.1%} left/right"
            )

        # division orientation
        ax = axes[0, 3]
        angles = orientation["median angle"]
        random = _random_orientation_angle(int(orientation.embryos.median()))
        ax.hist(angles, bins=np.linspace(0, 90, 19), color=blue, alpha=0.1)
        ax.hist(
            angles,
            bins=np.linspace(0, 90, 19),
            color=blue,
            histtype="step",
            lw=2,
        )
        ax.axvline(random, color=gray, lw=2)
        ax.text(
            random + 2,
            ax.get_ylim()[1] * 0.95,
            "random\ndirections",
            color=_INK["muted"],
            fontsize=10,
            va="top",
        )
        ax.set(
            xlabel="angle to the mean direction (°)",
            xlim=(0, 90),
            ylabel="divisions",
        )
        ax.set_xticks([0, 30, 60, 90])
        _panel_title(
            ax,
            "Division orientation",
            f"median {angles.median():.0f}° (random: {random:.0f}°)",
        )

        # positions and contacts over time
        for ax, values in ((axes[1, 0], positions), (axes[1, 1], contacts)):
            table = values[values.minutes <= until]
            median = (
                table.groupby(["minutes", "comparison"])
                .distance.median()
                .unstack()
            )
            series = [("between embryos", blue), ("left/right", orange)]
            if "sisters" in median:
                series.insert(0, ("sisters", gray))
            for comparison, color in series:
                s = median[comparison].dropna()
                ax.plot(
                    s.index,
                    s.values,
                    color=color,
                    marker="o",
                    ms=6,
                    mec="white",
                    mew=1.5,
                )
            ax.set(
                xlabel="minutes post fertilization",
                ylim=(0, None),
                xlim=(250, until + 10),
            )
            first, last = median["between embryos"].dropna().iloc[[0, -1]]
            if "sisters" in median:
                ax.yaxis.set_major_formatter(
                    FuncFormatter(lambda v, _: f"{v:.0%}")
                )
                ax.set(
                    ylim=(0, 0.75), ylabel="surface with different neighbours"
                )
                ax.text(
                    255,
                    median.sisters.max() + 0.03,
                    "sisters",
                    color=_INK["muted"],
                    fontsize=10,
                )
                _panel_title(
                    ax,
                    "Cell contacts",
                    f"{first:.0%} → {last:.0%} between; sisters "
                    f"{median.sisters.median():.0%}",
                )
            else:
                ax.set_ylabel("distance (cell radii)")
                _panel_title(
                    ax,
                    "Cell position",
                    f"{first:.1f} → {last:.1f} cell radii between",
                )

        # gastrulation
        ax = axes[1, 2]
        sd = gastrulation.std().sort_values()
        for i, tissue in enumerate(sd.index):
            values = gastrulation[tissue].dropna()
            ax.scatter(
                values,
                [i] * len(values),
                s=40,
                color=blue,
                edgecolor="white",
                lw=1.5,
                zorder=3,
            )
        ax.set_yticks(range(len(sd)), sd.index)
        ax.tick_params(axis="y", length=0, labelcolor=_INK["secondary"])
        ax.grid(False)
        ax.grid(True, axis="x")
        ax.axvline(0, color=_INK["axis"], lw=1)
        ax.set(xlabel="shift of each embryo (min)", ylim=(-0.6, len(sd) - 0.4))
        rest = sd.iloc[:-1]
        _panel_title(
            ax,
            "Gastrulation timing",
            f"SD {rest.min():.1f}{_DASH}{rest.max():.1f} min "
            f"({sd.index[-1]} {sd.iloc[-1]:.1f})",
        )

        # lineage trees
        ax = axes[1, 3]
        samples = {
            gray: naive.distance,
            blue: lineages[lineages.comparison == "between embryos"].distance,
            orange: lineages[lineages.comparison == "left/right"].distance,
        }
        _cumulative(ax, samples, 0.5)
        ax.set(xlabel="normalised UTED", ylabel="lineages (cumulative)")
        m = {c: np.median(s) for c, s in samples.items()}
        ax.text(
            m[gray] + 0.02,
            0.45,
            "not aligned",
            color=_INK["muted"],
            fontsize=10,
            va="top",
        )
        _panel_title(
            ax,
            "Lineage trees",
            f"{m[blue]:.0%} between, {m[orange]:.0%} left/right",
        )

        _figure_heading(
            figure,
            "Two embryos differ only slightly more than the two sides of "
            "one embryo",
            keys=[
                ("same cell, two embryos", blue),
                ("same cell, two sides of one embryo", orange),
            ],
        )
    return figure


def recap_structure(
    asynchrony,
    pairs,
    programmed,
    divisions,
    unequal_cycles,
    contact_sizes,
    sisters,
    tissue_profile,
    b77,
):
    """One slide on where the variability lies and what is reproducible.

    Parameters
    ----------
    asynchrony : pandas.DataFrame
        Division time of the odd daughter minus that of its sister, one row
        per observation (``sister_asynchrony``).
    pairs : pandas.DataFrame
        ``mean`` and ``std`` of that difference for each mother seen often
        enough, indexed by the name of the mother without its side.
    programmed : pandas.DataFrame
        The rows of ``pairs`` where the same sister always divides first.
    divisions : pandas.DataFrame
        ``mean`` and ``std`` over the embryos of the log2 volume ratio of
        the daughters of each division.
    unequal_cycles : list of float
        For each unequal division, the cycle of the smaller daughter minus
        that of the larger one (minutes).
    contact_sizes : pandas.DataFrame
        Contacts binned by their size, indexed by intervals of the fraction
        of the cell surface, with the fraction ``present_in_all`` the
        embryos.
    sisters : pandas.DataFrame
        Normalised UTED ``distance`` between the lineages of sister cells,
        and whether their ``tissues differ``.
    tissue_profile : pandas.DataFrame
        Median variability of the cells of each tissue (rows) for each
        measure (columns).
    b77 : pandas.DataFrame
        For each side of each embryo, when B7.7 divides (``B7.7 divides
        (min)``) and the ``volume B8.14 / B8.13`` of its daughters.

    Returns
    -------
    matplotlib.figure.Figure
        A 16:9 figure (13.33 x 7.5 inches) for a slide.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    blue = RECAP_COLORS["between embryos"]
    gray = RECAP_COLORS["reference"]
    ratios = [1, 2, 4, 8, 16]
    with plt.rc_context(_recap_style()):
        figure, axes = plt.subplots(2, 3, figsize=(13.33, 7.5))
        figure.subplots_adjust(
            left=0.08,
            right=0.98,
            bottom=0.08,
            top=0.76,
            wspace=0.45,
            hspace=0.72,
        )

        # sister asynchrony
        ax = axes[0, 0]
        kept = asynchrony[asynchrony.mother.isin(pairs.index)]
        share = 1 - (
            kept.groupby("mother").difference.var().mean()
            / kept.difference.var()
        )
        delays = programmed["mean"].abs().sort_values()
        others = pairs.drop(programmed.index)
        rng = np.random.default_rng(0)
        ax.scatter(
            others["mean"].abs(),
            rng.uniform(-0.25, 0.25, len(others)),
            s=30,
            color=gray,
            edgecolor="white",
            lw=1,
            zorder=3,
        )
        for i, (mother, delay) in enumerate(delays.items(), 1):
            ax.errorbar(
                delay,
                i,
                xerr=programmed.at[mother, "std"],
                fmt="o",
                ms=7,
                color=blue,
                mec="white",
                mew=1.5,
                lw=1.5,
                zorder=3,
            )
        ax.set_yticks(
            range(len(delays) + 1),
            [f"{len(others)} other pairs"]
            + [conklin(m + "_") for m in delays.index],
        )
        ax.tick_params(axis="y", length=0, labelcolor=_INK["secondary"])
        ax.grid(False)
        ax.grid(True, axis="x")
        ax.set(xlabel="delay between the sisters (min)", xlim=(0, None))
        _panel_title(
            ax,
            "Sister asynchrony",
            f"{len(programmed)} mothers: one daughter always divides first;\n"
            f"the pair explains {share:.0%} of the variance",
        )

        # unequal divisions
        ax = axes[0, 1]
        order = divisions["mean"].abs().sort_values(ascending=False)
        ratio = 2**order
        unequal = order > 0.5
        low = 2 ** (order - divisions["std"].reindex(order.index))
        high = 2 ** (order + divisions["std"].reindex(order.index))
        x = np.arange(len(order))
        ax.vlines(
            x[unequal],
            low[unequal].clip(lower=1),
            high[unequal],
            color=blue,
            lw=1,
            alpha=0.5,
        )
        for mask, color in ((~unequal, gray), (unequal, blue)):
            ax.scatter(x[mask], ratio[mask], s=14, color=color, zorder=3)
        ax.axhline(2**0.5, color=_INK["axis"], lw=1)
        ax.text(
            len(order),
            2**0.5,
            f"1.4{_TIMES}",
            ha="right",
            va="bottom",
            fontsize=10,
            color=_INK["muted"],
        )
        ax.set_yscale("log", base=2)
        ax.set_yticks(ratios, [f"{r}{_TIMES}" for r in ratios])
        ax.minorticks_off()
        ax.set(
            xlabel="divisions, most unequal first",
            xticks=[],
            ylabel="larger / smaller daughter",
            ylim=(0.9, 24),
        )
        later = np.mean(np.asarray(unequal_cycles) > 0)
        _panel_title(
            ax,
            "Unequal divisions",
            f"{unequal.sum()} of {len(order)} divisions are unequal "
            f"(over 1.4{_TIMES});\n"
            f"the smaller daughter divides later in {later:.0%}",
        )

        # B7.7
        ax = axes[0, 2]
        time = b77["B7.7 divides (min)"].astype(float)
        volume = b77["volume B8.14 / B8.13"].astype(float)
        late = time > time.median() + 20
        ax.scatter(
            time, volume, s=60, color=blue, edgecolor="white", lw=1.5, zorder=3
        )
        ax.set_yscale("log", base=2)
        ax.set_yticks(ratios, [f"{r}{_TIMES}" for r in ratios])
        ax.minorticks_off()
        ax.set(
            xlabel="B7.7 divides (min post fertilization)",
            ylabel="B8.14 / B8.13 volume",
            ylim=(0.8, 32),
            xlim=(time.min() - 15, time.max() + 15),
        )
        ax.annotate(
            f"{(~late).sum()} sides: early, unequal",
            (time[~late].min(), volume[~late].max()),
            xytext=(0, 10),
            textcoords="offset points",
            ha="left",
            fontsize=10,
            color=_INK["secondary"],
        )
        ax.annotate(
            f"{late.sum()} sides: late, equal",
            (time[late].max(), volume[late].max()),
            xytext=(0, 12),
            textcoords="offset points",
            ha="right",
            fontsize=10,
            color=_INK["secondary"],
        )
        gap = time[late].median() - time[~late].median()
        where = sorted(
            {index.split()[0].removesuffix("-v2") for index in b77.index[late]}
        )
        _panel_title(
            ax,
            "B7.7 differs between embryos",
            f"a tiny B8.13 that never divides, or an equal\n"
            f"division {gap:.0f} min later ({', '.join(where)})",
        )

        # contacts by size
        ax = axes[1, 0]
        labels = [
            (
                f"<{iv.right:.0%}"
                if i == 0
                else f"{iv.left * 100:.0f}{_DASH}{iv.right:.0%}"
            )
            for i, iv in enumerate(contact_sizes.index)
        ]
        present = contact_sizes["present_in_all"].values
        bars = ax.bar(range(len(present)), present, width=0.55, color=blue)
        for bar, value in zip(bars, present, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.0%}",
                ha="center",
                va="bottom",
                fontsize=10,
                color=_INK["secondary"],
            )
        ax.set_xticks(range(len(present)), labels)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set(
            ylim=(0, 1.12),
            xlabel="contact size (share of the surface)",
            ylabel="found in every embryo",
        )
        ax.set_yticks([0, 0.5, 1])
        _panel_title(
            ax,
            "Contacts",
            "small contacts come and go;\n"
            "large ones are found in every embryo",
        )

        # sister lineages
        ax = axes[1, 1]
        known = sisters.dropna(subset=["tissues differ"])
        differ = known["tissues differ"].astype(bool)
        samples = {gray: known.distance[~differ], blue: known.distance[differ]}
        _cumulative(ax, samples, 1)
        auc = (
            known.distance[differ].values[:, None]
            > known.distance[~differ].values[None, :]
        ).mean() + 0.5 * (
            known.distance[differ].values[:, None]
            == known.distance[~differ].values[None, :]
        ).mean()
        ax.set(
            xlabel="normalised UTED between the sister lineages",
            ylabel="divisions (cumulative)",
        )
        ax.legend(
            [plt.Line2D([], [], color=blue), plt.Line2D([], [], color=gray)],
            [
                f"different tissues ({differ.sum()})",
                f"same tissues ({(~differ).sum()})",
            ],
            loc="lower right",
            frameon=False,
            fontsize=10,
            labelcolor=_INK["secondary"],
        )
        _panel_title(
            ax,
            "Sister lineages",
            f"diverge more when their tissues differ\n"
            f"(AUC {auc:.2f}; {known.embryo.iloc[0]})",
        )

        # tissues
        ax = axes[1, 2]
        profile = tissue_profile.dropna(how="all")
        ranks = profile.rank(pct=True)
        colours = plt.get_cmap("Blues")(0.15 + 0.7 * ranks.fillna(0).values)
        colours[ranks.isna().values] = plt.matplotlib.colors.to_rgba(
            _INK["empty"]
        )
        ax.imshow(colours, aspect="auto")
        formats = ["{:.0%}", "{:.0f}", "{:.0%}", "{:.1f}", "{:.0%}", "{:.0%}"]
        for i in range(profile.shape[0]):
            for j in range(profile.shape[1]):
                value = profile.values[i, j]
                ax.text(
                    j,
                    i,
                    (
                        _DASH
                        if np.isnan(value)
                        else formats[j % len(formats)].format(value)
                    ),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=(
                        "white"
                        if ranks.values[i, j] > 0.6
                        else _INK["primary"]
                    ),
                )
        ax.set_xticks(
            range(profile.shape[1]),
            [
                "cycle\n(CV)",
                "division\n(SD, min)",
                "volume\n(diff.)",
                "position\n(radii)",
                "contacts\n(dist.)",
                "lineage\n(UTED)",
            ][: profile.shape[1]],
            fontsize=9,
        )
        ax.set_yticks(range(profile.shape[0]), profile.index)
        ax.tick_params(length=0, labelcolor=_INK["secondary"])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        _panel_title(
            ax,
            "Tissues",
            "median over the cells of each tissue;\n"
            "darker: more variable than the other tissues",
        )

        _figure_heading(
            figure,
            "Where the variability lies",
            deck="Many differences between cells are the same in every "
            "embryo; small contacts and B7.7 are not",
        )
    return figure


# --------------------------------------------------------------------------
# Simple examples
# --------------------------------------------------------------------------

EXAMPLE_COLOR = RECAP_COLORS["between embryos"]
EXAMPLE_SIZE = (6.4, 4.2)
_DEGREE = "\N{DEGREE SIGN}"
_LEFT_ARROW = "\N{LEFTWARDS ARROW}"


def _short(embryo):
    """Short label of an embryo, e.g. ``"embryo 1"`` for ``Phmamm-1-v2``.

    Parameters
    ----------
    embryo : str
        The name of the embryo.

    Returns
    -------
    str
        The label.
    """
    return "embryo " + embryo.split("-")[1]


def _example_style():
    """Matplotlib settings of the example figures.

    Returns
    -------
    dict
        The settings of the recap figures (see ``_recap_style``), with
        larger text for slides.
    """
    return {
        **_recap_style(),
        "font.size": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    }


def _example_figure(title, subtitle, grid="y"):
    """Create the figure of one example, with its title.

    Parameters
    ----------
    title : str
        The message of the figure.
    subtitle : str
        What is plotted, on one line.
    grid : {"x", "y", "none"}, default="y"
        Direction of the grid lines: ``"x"`` for vertical lines at the
        ticks of the horizontal axis.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    matplotlib.axes.Axes
        Its only panel.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=EXAMPLE_SIZE, layout="constrained")
    fig.get_layout_engine().set(rect=(0.01, 0.01, 0.98, 0.82))
    ax.grid(False)
    if grid != "none":
        ax.grid(True, axis=grid)
    fig.text(
        0.025,
        0.965,
        title,
        fontsize=16,
        fontweight="bold",
        color=_INK["primary"],
        va="top",
    )
    fig.text(
        0.025, 0.88, subtitle, fontsize=12, color=_INK["secondary"], va="top"
    )
    return fig, ax


def _bare(ax):
    """Remove the ticks and the frame of a panel that shows a picture.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The panel.
    """
    ax.set_aspect("equal")
    ax.set(xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _side_legend(ax):
    """Legend of the two homologues, to the right of a panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The panel.
    """
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        frameon=False,
        handletextpad=0.3,
        labelcolor=_INK["secondary"],
    )


def pace_example(counts, embryos, start=76, stop=218, interval=2.0):
    """Example figure: the embryos develop at different paces.

    Parameters
    ----------
    counts : dict of {str: pandas.Series}
        Number of cells of each embryo at each time point of its movie (see
        ``cell_counts``).
    embryos : list of str
        The embryos shown.
    start, stop : int, default=76, 218
        The two stages compared, as numbers of cells: the clock of each
        embryo starts when it first has ``start`` cells, and its curve ends
        with a dot when it first has ``stop`` cells.
    interval : float, default=2.0
        Minutes between two time points of the movies.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    import matplotlib.pyplot as plt

    with plt.rc_context(_example_style()):
        fig, ax = _example_figure(
            "Same development, different pace",
            f"Number of cells in {len(embryos)} embryos, from the {start}-cell"
            f" to the {stop}-cell stage",
        )
        durations = []
        for e in embryos:
            c = counts[e]
            first = c[c >= start].index[0]
            last = c[c >= stop].index[0] if (c >= stop).any() else c.index[-1]
            shown = c[(c.index >= first) & (c.index <= last)]
            x = (shown.index - first) * interval
            ax.plot(x, shown.values, color=EXAMPLE_COLOR, lw=1.8, alpha=0.8)
            if shown.iloc[-1] >= stop:
                durations.append(x[-1])
                ax.plot(
                    x[-1],
                    shown.iloc[-1],
                    "o",
                    ms=9,
                    color=EXAMPLE_COLOR,
                    mec="white",
                    mew=1.5,
                    zorder=4,
                )
        fast, slow = min(durations), max(durations)
        y = stop * 1.1
        ax.annotate(
            "",
            xy=(slow, y),
            xytext=(fast, y),
            arrowprops={
                "arrowstyle": "<->",
                "color": _INK["secondary"],
                "lw": 1.5,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
        ax.text(
            fast - 3,
            y,
            f"{fast:.0f} to {slow:.0f} min",
            ha="right",
            va="center",
            color=_INK["primary"],
        )
        ax.set(
            xlabel=f"minutes of movie after the {start}-cell stage",
            ylabel="number of cells",
            ylim=(start * 0.9, stop * 1.18),
            xlim=(0, slow * 1.06),
        )
    return fig


def cell_in_embryos(values, embryos, name):
    """Values of a cell and of its homologue in several embryos.

    Parameters
    ----------
    values : pandas.DataFrame
        One row per cell name, one column per embryo.
    embryos : list of str
        The embryos.
    name : str
        A cell name, either side.

    Returns
    -------
    pandas.DataFrame
        One row per embryo, two columns: the cell of the ``_`` side and its
        homologue, named as in the literature (see ``conklin``).
    """
    pair = sorted([name, homologue(name)], key=lambda n: n[-1] != "_")
    return pd.DataFrame(
        {conklin(n): values.loc[n, embryos] for n in pair}
    ).astype(float)


def dot_example(table, title, subtitle, xlabel, unit="", ratio=False):
    """Example figure: one value per embryo and side, as a dot plot.

    The two sides of an embryo are drawn slightly above and below its row,
    joined by a line, so that equal values stay visible.

    Parameters
    ----------
    table : pandas.DataFrame
        One row per embryo, two columns: the cell and its homologue (see
        ``cell_in_embryos``).
    title : str
        The message of the figure.
    subtitle : str
        What is plotted.
    xlabel : str
        Label of the horizontal axis.
    unit : str, default=""
        Unit of the range written on the figure, e.g. ``" min"``.
    ratio : bool, default=False
        Write the ratio of the largest to the smallest value instead of the
        range.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    import matplotlib.pyplot as plt

    with plt.rc_context(_example_style()):
        fig, ax = _example_figure(title, subtitle, grid="x")
        rows = np.arange(len(table))[::-1]
        shifts = (0.13, -0.13)
        low, high = np.nanmin(table.values), np.nanmax(table.values)
        for row, values in zip(rows, table.values, strict=True):
            ax.plot(
                values[:2],
                row + np.array(shifts),
                color=_INK["axis"],
                lw=1.5,
                zorder=1,
            )
        for (side, values), filled, shift in zip(
            table.items(), (True, False), shifts, strict=False
        ):
            ax.plot(
                values.values,
                rows + shift,
                "o",
                ms=10,
                mew=2,
                color=EXAMPLE_COLOR,
                mfc=EXAMPLE_COLOR if filled else "white",
                label=side,
                ls="none",
                zorder=3,
            )
        ax.set_yticks(rows, [_short(e) for e in table.index])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel(xlabel)
        span = high - low
        ax.set_xlim(low - 0.2 * span, high + 0.2 * span)
        ax.set_ylim(-0.7, len(table) - 0.3)
        _side_legend(ax)
        if ratio:
            note = f"largest:\n{high / low:.2f}{_TIMES} the\nsmallest"
        else:
            note = f"range:\n{span:.2g}{unit}"
        ax.text(
            1.04,
            0.45,
            note,
            transform=ax.transAxes,
            ha="left",
            va="center",
            color=_INK["primary"],
        )
    return fig


def registered_cell(manager, alignment, reference_time, embryos, name):
    """Positions of a cell in several embryos, registered on the reference.

    Parameters
    ----------
    manager : LineageTreeManager
        The embryos.
    alignment : TimeAlignment
        Their alignment.
    reference_time : float
        The reference time point.
    embryos : list of str
        The embryos.
    name : str
        The cell.

    Returns
    -------
    pandas.DataFrame
        The coordinates ``x``, ``y`` and ``z`` of the cell in each embryo
        whose movie covers the time, after registration on the reference
        embryo by rotation, scaling and translation (see ``common_cells``
        and ``similarity_fit``), one row per embryo.
    """
    reference = alignment.reference
    target_cells = positions(
        manager[reference], frame_at(alignment, reference, reference_time)
    )
    rows = {}
    for e in embryos:
        t = frame_at(alignment, e, reference_time)
        if t is None:
            continue
        cells = positions(manager[e], t)
        if name not in cells:
            continue
        _, target, source = common_cells(target_cells, cells)
        _, scale, rotation = similarity_fit(source, target)
        offset = cells[name][0] - source.mean(axis=0)
        rows[e] = scale * rotation @ offset + target.mean(axis=0)
    return pd.DataFrame(rows, index=["x", "y", "z"]).T


def position_example(manager, alignment, reference_time, embryos, name):
    """Example figure: a cell sits at nearly the same place in every embryo.

    The cells of the reference embryo are drawn as discs of their own
    radius, seen from the side of the embryo where the cell is, with the
    anterior to the left. The dots are the registered positions of the cell
    and of its homologue in each embryo (see ``registered_cell``).

    Parameters
    ----------
    manager : LineageTreeManager
        The embryos.
    alignment : TimeAlignment
        Their alignment.
    reference_time : float
        The reference time point.
    embryos : list of str
        The embryos.
    name : str
        The cell.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    lT = manager[alignment.reference]
    cells = positions(
        lT, frame_at(alignment, alignment.reference, reference_time)
    )
    names = list(cells)
    xyz = np.array([cells[n][0] for n in names])
    radius = np.array(
        [(3 * cells[n][1] / (4 * np.pi)) ** (1 / 3) for n in names]
    )
    centre = xyz.mean(axis=0)

    def centroid(keep):
        """Mean position of the reference cells whose name passes a test.

        Parameters
        ----------
        keep : callable
            Test on the name of a cell.

        Returns
        -------
        numpy.ndarray
            The mean position.
        """
        return xyz[[keep(n) for n in names]].mean(axis=0)

    anterior = centroid(lambda n: n[0] == "a") - centroid(
        lambda n: n[0] == "b"
    )
    anterior /= np.linalg.norm(anterior)
    side = centroid(lambda n: n[-1] == "_") - centroid(lambda n: n[-1] == "*")
    side -= (side @ anterior) * anterior
    side /= np.linalg.norm(side)
    depth = np.cross(anterior, side)
    if (cells[name][0] - centre) @ depth < 0:  # look from the other side
        side, depth = -side, -depth
    view = np.stack([-anterior, side, depth])  # anterior to the left
    flat = (xyz - centre) @ view.T
    minutes = alignment.minutes(alignment.reference, reference_time)
    pair = (name, homologue(name))
    with plt.rc_context(_example_style()):
        fig, ax = _example_figure(
            "Same cell, nearly the same place",
            f"{conklin(name)} and its homologue in {len(embryos)} registered"
            f" embryos, {minutes:.0f} min",
            grid="none",
        )
        for i in np.argsort(flat[:, 2]):  # from the back to the front
            ax.add_patch(
                Circle(
                    flat[i, :2],
                    radius[i],
                    facecolor="#dceaf9" if names[i] in pair else _INK["empty"],
                    edgecolor="white",
                    lw=1.2,
                )
            )
        for n, filled in zip(pair, (True, False), strict=True):
            points = registered_cell(
                manager, alignment, reference_time, embryos, n
            )
            ax.plot(
                *((points.values - centre) @ view[:2].T).T,
                "o",
                ms=7,
                mew=1.6,
                color=EXAMPLE_COLOR,
                mfc=EXAMPLE_COLOR if filled else "white",
                label=conklin(n),
                ls="none",
                zorder=3,
            )
        low = (flat[:, :2] - radius[:, None]).min(axis=0)
        high = (flat[:, :2] + radius[:, None]).max(axis=0)
        margin = 0.08 * (high[1] - low[1])
        ax.set(xlim=(low[0], high[0]), ylim=(low[1] - margin, high[1]))
        _bare(ax)
        ax.text(
            low[0],
            low[1] - margin / 2,
            f"{_LEFT_ARROW} anterior",
            va="center",
            fontsize=11,
            color=_INK["muted"],
        )
        _side_legend(ax)
    return fig


def axis_example(oriented, mother, reference=True):
    """Example figure: a cell divides along similar axes in every embryo.

    Each division axis is drawn at its true angle to the average axis of
    the embryos, on the side given by its component along the main
    direction of the spread.

    Parameters
    ----------
    oriented : pandas.DataFrame
        Columns ``mother``, ``embryo`` and ``v``: the division axes,
        registered on the reference embryo.
    mother : str
        The dividing cell.
    reference : bool, default=True
        Draw, as grey dashed lines, the median angle of as many random axes
        to their average (see ``_random_orientation_angle``).

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    import matplotlib.pyplot as plt

    vectors = np.stack(oriented[oriented.mother == mother].v.values)
    vectors = vectors * np.sign(vectors @ vectors[0])[:, None]
    mean = vectors.mean(axis=0)
    mean /= np.linalg.norm(mean)
    across = vectors - np.outer(vectors @ mean, mean)
    spread = np.linalg.svd(across, full_matrices=False)[2][0]
    angles = np.degrees(np.arccos(np.clip(vectors @ mean, -1, 1)))
    signs = np.where(across @ spread >= 0, 1, -1)

    def line(ax, angle, **style):
        """Draw a line through the centre at an angle from the vertical.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The panel.
        angle : float
            The angle, in degrees.
        **style
            Passed to ``Axes.plot``.
        """
        a = np.radians(angle)
        ax.plot([-np.sin(a), np.sin(a)], [-np.cos(a), np.cos(a)], **style)

    with plt.rc_context(_example_style()):
        fig, ax = _example_figure(
            "Same cell, similar division axes",
            f"Axis of the division of {conklin(mother)} in {len(vectors)}"
            " embryos",
            grid="none",
        )
        if reference:
            random_angle = _random_orientation_angle(len(vectors))
            for s in (1, -1):
                line(ax, s * random_angle, color=_INK["axis"], ls="--", lw=1.5)
            a = np.radians(random_angle)
            ax.text(
                np.sin(a) + 0.05,
                np.cos(a),
                f"random axes: median\n{random_angle:.0f}{_DEGREE} away",
                color=_INK["muted"],
                va="center",
                fontsize=11,
            )
        for angle, s in zip(angles, signs, strict=True):
            line(ax, s * angle, color=EXAMPLE_COLOR, lw=2.5, alpha=0.8)
        ax.plot(0, 0, "o", ms=9, color=EXAMPLE_COLOR, mec="white", mew=1.5)
        ax.text(
            0.0,
            -1.1,
            f"{angles.min():.0f} to {angles.max():.0f}{_DEGREE} from their"
            " average",
            ha="center",
            va="top",
            color=_INK["primary"],
        )
        ax.set(xlim=(-1.7, 1.7), ylim=(-1.3, 1.1))
        _bare(ax)
    return fig


def cell_contacts(neighbourhoods, name, embryos):
    """Contacts of a cell in several embryos, with common neighbour names.

    Neighbours that have divided in some embryos only are replaced by their
    mother, as in ``common_cells``.

    Parameters
    ----------
    neighbourhoods : dict of {str: dict of {str: dict}}
        Neighbourhoods of the cells of each embryo at a matched time (see
        ``neighbourhoods_at``).
    name : str
        The cell.
    embryos : list of str
        The embryos.

    Returns
    -------
    pandas.DataFrame
        One row per neighbour (named as in the literature, ``"outside"``
        for the exterior of the embryo), one column per embryo: the
        fraction of the surface of the cell in contact, 0 where they do not
        touch. Sorted by mean contact.
    """
    found = {
        e: neighbourhoods[e][name]
        for e in embryos
        if name in neighbourhoods.get(e, {})
    }
    mapping = _merge_to_ancestors(set().union(*found.values()))
    rows = {}
    for e, contacts in found.items():
        rows[e] = {}
        for k, v in contacts.items():
            rows[e][mapping[k]] = rows[e].get(mapping[k], 0.0) + v
    table = pd.DataFrame(rows).fillna(0.0)
    table = table.drop(index="unnamed", errors="ignore")
    table.index = [k if k == "outside" else conklin(k) for k in table.index]
    return table.loc[table.mean(axis=1).sort_values(ascending=False).index]


def contact_example(contacts, name, minutes, smallest=0.005):
    """Example figure: a cell touches mostly the same neighbours.

    Parameters
    ----------
    contacts : pandas.DataFrame
        Fraction of the surface of the cell in contact with each neighbour,
        one column per embryo (see ``cell_contacts``).
    name : str
        The name of the cell, for the subtitle.
    minutes : float
        Minutes post fertilization, for the subtitle.
    smallest : float, default=0.005
        Neighbours with a smaller mean contact are not shown.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, PercentFormatter

    shown = contacts[contacts.mean(axis=1) >= smallest]
    with plt.rc_context(_example_style()):
        fig, ax = _example_figure(
            "Same cell, mostly the same neighbours",
            f"Contacts of {name} in {contacts.shape[1]} embryos,"
            f" {minutes:.0f} min",
            grid="x",
        )
        rows = np.arange(len(shown))[::-1]
        rng = np.random.default_rng(0)
        for row, (_, values) in zip(rows, shown.iterrows(), strict=True):
            y = row + rng.uniform(-0.2, 0.2, len(values))
            touching = values.values > 0
            ax.plot(
                values.values[touching],
                y[touching],
                "o",
                ms=6.5,
                color=EXAMPLE_COLOR,
                mec="white",
                mew=0.8,
                ls="none",
            )
            ax.plot(
                np.zeros((~touching).sum()),
                y[~touching],
                "o",
                ms=6.5,
                mfc="white",
                mec=_INK["muted"],
                mew=1.2,
                ls="none",
            )
            ax.text(
                1.02,
                row,
                f"{touching.sum()}/{len(values)}",
                transform=ax.get_yaxis_transform(),
                va="center",
                color=_INK["primary"] if touching.all() else EXAMPLE_COLOR,
                fontsize=12,
                fontweight="normal" if touching.all() else "bold",
            )
        ax.set_yticks(
            rows,
            ["embryo surface" if k == "outside" else k for k in shown.index],
            fontsize=11,
        )
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_ylim(-0.6, len(shown) - 0.4)
        ax.set_xlim(-0.006, None)
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.xaxis.set_major_formatter(PercentFormatter(1.0, 0))
        ax.set_xlabel("share of the surface of the cell")
        ax.text(
            1.02,
            1.0,
            "embryos",
            transform=ax.transAxes,
            va="bottom",
            fontsize=11,
            color=_INK["muted"],
        )
    return fig


def lineage_example(tables, alignment, embryos, root, start=None, stop=None):
    """Example figure: the same lineage tree with a different timing.

    Each cell is a horizontal line from the division of its mother to its
    own division, where a vertical line joins its two daughters.

    Parameters
    ----------
    tables : dict of {str: pandas.DataFrame}
        The cell table of each embryo (see ``cell_table``).
    alignment : TimeAlignment
        The alignment of the embryos.
    embryos : list of str
        The embryos, one tree per row.
    root : str
        The cell at the root of the trees.
    start, stop : float, optional
        The minutes shown; by default, the time covered by all the movies.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    import matplotlib.pyplot as plt

    covered = [
        alignment.minutes(e, np.array(alignment.frames[e])) for e in embryos
    ]
    start = max(float(a) for a, _ in covered) if start is None else start
    stop = min(float(b) for _, b in covered) if stop is None else stop
    _, g0, i0, _ = parse_name(root)

    def height(n, row):
        """Height of a cell in the tree of an embryo.

        Parameters
        ----------
        n : str
            The root or one of its descendants.
        row : int
            The row of the embryo.

        Returns
        -------
        float
            The height: the band of the row is split equally between the
            descendants of each generation.
        """
        _, g, i, _ = parse_name(n)
        slots = 2 ** (g - g0)
        return row - 0.36 + (i - (i0 - 1) * slots - 0.5) * 0.72 / slots

    with plt.rc_context(_example_style()):
        fig, ax = _example_figure(
            "Same lineage tree, different timing",
            f"{conklin(root)} and its descendants in {len(embryos)} embryos",
            grid="x",
        )
        style = {"color": EXAMPLE_COLOR, "lw": 2, "solid_capstyle": "butt"}
        for row, e in enumerate(embryos[::-1]):
            t = tables[e]
            for n in t.index:
                if n != root and not is_ancestor(root, n):
                    continue
                divides = pd.notna(t.division[n])
                born = np.nan if n == root else t.division.get(mother(n))
                begin = alignment.minutes(
                    e, born if pd.notna(born) else t.start[n]
                )
                begin = max(float(begin), start)  # from the mother's division
                end = float(
                    alignment.minutes(
                        e, t.division[n] if divides else t.end[n]
                    )
                )
                if begin >= stop:
                    continue
                y = height(n, row)
                ax.plot([begin, min(end, stop)], [y, y], **style)
                if divides and end < stop:
                    ax.plot(
                        [end, end],
                        [height(k, row) for k in daughters(n)],
                        **style,
                    )
        ax.set_yticks(range(len(embryos)), [_short(e) for e in embryos[::-1]])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set(xlim=(start, stop), ylim=(-0.6, len(embryos) - 0.4))
        ax.set_xlabel("minutes post fertilization")
    return fig


# --------------------------------------------------------------------------
# Variability by tissue and developmental time
# --------------------------------------------------------------------------

FOLD_TICKS = [0.25, 0.35, 0.5, 0.71, 1, 1.41, 2, 2.83, 4]


def stage_times(curve, alignment, sizes):
    """When the embryos reach given numbers of cells.

    Parameters
    ----------
    curve : pandas.Series
        Median number of cells by reference time point (``count_curve``).
    alignment : TimeAlignment
        The alignment of the embryos.
    sizes : iterable of int
        Numbers of cells.

    Returns
    -------
    pandas.Series
        For each number of cells, the first time, in minutes post
        fertilization, at which the median embryo has at least that many
        cells; sizes never reached are left out.
    """
    out = {}
    for size in sizes:
        reached = curve[curve >= size]
        if len(reached):
            out[size] = alignment.start_minutes + alignment.interval * (
                reached.index[0]
            )
    return pd.Series(out, dtype=float)


def event_minutes(alignment, event="division"):
    """Time of an event of each cell in each embryo, where it can be seen.

    Parameters
    ----------
    alignment : TimeAlignment
        The alignment of the embryos, with their cell tables.
    event : {"birth", "division"}, default="division"
        The event.

    Returns
    -------
    pandas.DataFrame
        Minutes post fertilization in reference time, indexed by cell name,
        one column per embryo. As in ``TimeAlignment.deviations``, an event
        is NaN in an embryo whose movie does not cover its consensus time
        with margin: near the ends of a movie, only the early or the late
        events would be seen.
    """
    times = alignment.aligned(event)
    consensus = alignment.consensus(event)
    return pd.DataFrame(
        {
            e: (alignment.start_minutes + alignment.interval * times[e]).where(
                alignment.covers(e, consensus.reindex(times.index))
            )
            for e in alignment.embryos
        }
    )


def life_minutes(alignment):
    """Consensus birth, division and middle of the life of each cell.

    Parameters
    ----------
    alignment : TimeAlignment
        The alignment of the embryos, with their cell tables.

    Returns
    -------
    pandas.DataFrame
        Indexed by cell name, in minutes post fertilization: ``birth`` and
        ``division`` (consensus times, NaN if never seen), and ``middle``:
        halfway between the two, or, for cells seen only in part, the
        median over the embryos of the middle of the part seen.
    """
    to_minutes = alignment.start_minutes, alignment.interval
    birth = to_minutes[0] + to_minutes[1] * alignment.consensus("birth")
    division = to_minutes[0] + to_minutes[1] * alignment.consensus("division")
    seen = pd.DataFrame(
        {
            e: alignment.minutes(e, (t.start + t.end) / 2)
            for e, t in alignment.tables.items()
        }
    ).median(axis=1)
    out = pd.DataFrame({"birth": birth, "division": division})
    out = out.reindex(seen.index)
    out["middle"] = ((out.birth + out.division) / 2).fillna(seen)
    return out


def pair_differences(values, relative=False, sides=True):
    """Differences of a measure between embryos and between the two sides.

    Parameters
    ----------
    values : pandas.DataFrame
        A measure, indexed by cell name, one column per embryo.
    relative : bool, default=False
        Divide each absolute difference by the mean of the two values.
    sides : bool, default=True
        Also compare each left cell with its right homologue in the same
        embryo.

    Returns
    -------
    pandas.DataFrame
        One row per comparison, with the columns ``comparison``
        (``"between embryos"``: the same cell in two embryos, or
        ``"left/right"``: a left cell and its homologue in one embryo,
        listed under the name of the left cell), ``name``, ``embryo``,
        ``other`` and ``difference``.
    """
    embryos = list(values.columns)
    x = values.to_numpy(dtype=float)
    rows = []
    for i, j in _pairs(range(len(embryos))):
        ok = ~np.isnan(x[:, i]) & ~np.isnan(x[:, j])
        a, b = x[ok, i], x[ok, j]
        d = np.abs(a - b)
        if relative:
            d = d / ((a + b) / 2)
        rows.append(
            pd.DataFrame(
                {
                    "comparison": "between embryos",
                    "name": values.index[ok],
                    "embryo": embryos[i],
                    "other": embryos[j],
                    "difference": d,
                }
            )
        )
    if sides:
        left = [n for n in values.index if n.endswith("_")]
        right = [homologue(n) for n in left]
        keep = [r in values.index for r in right]
        left = [n for n, k in zip(left, keep, strict=True) if k]
        right = [n for n, k in zip(right, keep, strict=True) if k]
        a = values.loc[left].to_numpy(dtype=float)
        b = values.loc[right].to_numpy(dtype=float)
        for i, e in enumerate(embryos):
            ok = ~np.isnan(a[:, i]) & ~np.isnan(b[:, i])
            d = np.abs(a[ok, i] - b[ok, i])
            if relative:
                d = d / ((a[ok, i] + b[ok, i]) / 2)
            rows.append(
                pd.DataFrame(
                    {
                        "comparison": "left/right",
                        "name": np.array(left)[ok],
                        "embryo": e,
                        "other": e,
                        "difference": d,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def orientation_differences(manager, alignment, embryos, reference=REFERENCE):
    """Angles between the division axes of the same cells.

    The axis of a division goes from the daughter with the even index to
    the daughter with the odd index (``division_table``). To compare two
    embryos, each is first registered onto the reference embryo at the
    matching time (``rotation_to``). To compare the two sides of an embryo,
    the embryo is registered onto its mirror image, with the names of the
    two sides swapped, and the axis of the right cell is reflected onto
    the left side.

    Parameters
    ----------
    manager : LineageTreeManager
        The embryos, with principal axes and positions.
    alignment : TimeAlignment
        Their alignment, with their cell tables.
    embryos : list of str
        The embryos to compare.
    reference : str, default=REFERENCE
        The embryo onto which the others are registered.

    Returns
    -------
    pandas.DataFrame
        One row per comparison, with the columns ``comparison``
        (``"between embryos"`` or ``"left/right"``), ``name`` (the mother),
        ``embryo``, ``other`` and ``difference``, the angle in degrees,
        from 0 (same axis, same direction) to 180.
    """
    divisions = {
        e: division_table(
            manager[e], alignment.tables[e], alignment.frame_minutes(e)
        )
        for e in embryos
    }
    rotations, axes = {}, {}
    for e in embryos:
        for name, d in divisions[e].iterrows():
            t_ref = frame_at(
                alignment, reference, alignment.to_reference(e, d.frame)
            )
            if t_ref is None:
                continue
            key = (e, int(d.frame))
            if key not in rotations:
                rotations[key] = rotation_to(
                    manager, e, int(d.frame), reference, t_ref
                )
            axes.setdefault(name, {})[e] = rotations[key] @ np.array(
                [d.x, d.y, d.z]
            )
    rows = []
    for name, found in axes.items():
        for e, f in _pairs(sorted(found)):
            cosine = np.clip(found[e] @ found[f], -1, 1)
            rows.append(
                ("between embryos", name, e, f, np.degrees(np.arccos(cosine)))
            )
    reflections = {}
    for e in embryos:
        d = divisions[e]
        for name in d.index:
            other = homologue(name)
            if not name.endswith("_") or other not in d.index:
                continue
            t = int(d.at[name, "frame"])
            if t not in reflections:
                cells = positions(manager[e], t)
                mirror = {homologue(n): v for n, v in cells.items()}
                _, own, swapped = common_cells(cells, mirror)
                reflections[t] = similarity_fit(swapped, own, reflection=True)[
                    2
                ]
            axis = reflections[t] @ d.loc[other, ["x", "y", "z"]].to_numpy(
                dtype=float
            )
            cosine = np.clip(
                d.loc[name, ["x", "y", "z"]].to_numpy(dtype=float) @ axis,
                -1,
                1,
            )
            rows.append(
                ("left/right", name, e, e, np.degrees(np.arccos(cosine)))
            )
        reflections.clear()
    return pd.DataFrame(
        rows, columns=["comparison", "name", "embryo", "other", "difference"]
    )


def lineages_from_birth(manager, alignment, embryos, duration=60.0):
    """Tree edit distances between the lineages of the same cells from birth.

    The lineage of each cell is followed for ``duration`` reference minutes
    from its birth in each embryo, and compared with the lineage of the
    same cell in the other embryos and with that of its bilateral
    homologue in the same embryo. Cells whose lineage leaves the movie are
    left out.

    Parameters
    ----------
    manager : LineageTreeManager
        The embryos, with time resolutions (see ``set_time_resolutions``).
    alignment : TimeAlignment
        Their alignment, with their cell tables.
    embryos : list of str
        The embryos to compare.
    duration : float, default=60.0
        How long each lineage is followed, in reference minutes.

    Returns
    -------
    pandas.DataFrame
        One row per comparison, with the columns ``comparison``
        (``"between embryos"`` or ``"left/right"``), ``name``, ``embryo``,
        ``other`` and ``difference``: the normalised tree edit distance.
    """
    roots = {}
    for e in embryos:
        table = alignment.tables[e]
        last = alignment.frames[e][1]
        born = table[table.birth.notna()]
        end = born.birth + duration / alignment.frame_minutes(e)
        kept = born[end <= last]
        roots[e] = {
            name: (row["first"], end[name]) for name, row in kept.iterrows()
        }
    rows = []
    for e, f in _pairs(embryos):
        for name in sorted(set(roots[e]) & set(roots[f])):
            (a, end_a), (b, end_b) = roots[e][name], roots[f][name]
            d = manager.cross_lineage_edit_distance(
                a, e, b, f, end_a, end_b, style="simple"
            )
            rows.append(("between embryos", name, e, f, d))
        manager.clear_comparisons()
    for e in embryos:
        for name in sorted(roots[e]):
            other = homologue(name)
            if name.endswith("_") and other in roots[e]:
                (a, end_a), (b, end_b) = roots[e][name], roots[e][other]
                d = manager.cross_lineage_edit_distance(
                    a, e, b, e, end_a, end_b, style="simple"
                )
                rows.append(("left/right", name, e, e, d))
        manager.clear_comparisons()
    return pd.DataFrame(
        rows, columns=["comparison", "name", "embryo", "other", "difference"]
    )


def annotate(table, tissues, times, sizes=None):
    """Add the tissue, lineage, generation and time of each compared cell.

    Parameters
    ----------
    table : pandas.DataFrame
        Comparisons with a ``name`` column (``pair_differences``).
    tissues : pandas.Series
        The tissue of each cell name (``tissue_of``).
    times : pandas.Series or None
        The time attributed to each comparison, in minutes post
        fertilization, indexed by cell name; None if ``table`` already
        has a ``minutes`` column.
    sizes : pandas.Series, optional
        The volume of each cell name, as a fraction of the embryo.

    Returns
    -------
    pandas.DataFrame
        A copy of ``table`` with the columns ``cell`` (the name without its
        side, shared by the two homologues), ``tissue`` (``"multipotent"``
        for cells whose descendants make several tissues), ``line``,
        ``gen``, ``minutes`` and, if given, ``size``.
    """
    out = table.copy()
    names = out["name"]
    out["cell"] = names.str[:-1]
    tissue = tissues.reindex(names).fillna("unknown").to_numpy()
    out["tissue"] = np.where(tissue == "mixed", "multipotent", tissue)
    out["line"] = [founder(n) for n in names]
    out["gen"] = [parse_name(n)[1] for n in names]
    if times is not None:
        out["minutes"] = times.reindex(names).to_numpy()
    if sizes is not None:
        out["size"] = sizes.reindex(names).to_numpy()
    return out


def _weighted_medians(values, weights):
    """Weighted median of the same values under several sets of weights.

    Parameters
    ----------
    values : numpy.ndarray
        The values, shape (n,).
    weights : numpy.ndarray
        One set of non-negative weights per row, shape (draws, n).

    Returns
    -------
    numpy.ndarray
        The weighted median for each set of weights, NaN where all the
        weights are zero.
    """
    order = np.argsort(values)
    cumulative = np.cumsum(weights[:, order], axis=1)
    total = cumulative[:, -1]
    index = (cumulative >= total[:, None] / 2).argmax(axis=1)
    out = values[order][index].astype(float)
    out[total == 0] = np.nan
    return out


def embryo_bootstrap(
    table,
    by,
    value="difference",
    statistic="median",
    draws=1000,
    seed=0,
    level=0.9,
):
    """Median of a variability measure by group, with confidence intervals.

    The independent units are the embryos, not the cells: the confidence
    intervals come from resampling the embryos with replacement. A
    comparison between two embryos counts as many times as the product of
    the numbers of times the two embryos are drawn, a comparison of the
    two sides of an embryo as many times as that embryo is drawn. The same
    draws are used for all the groups, so that the draws of two groups
    can be compared.

    Parameters
    ----------
    table : pandas.DataFrame
        Comparisons, with the columns ``embryo``, ``other`` and ``value``.
    by : str or list of str
        The columns that define the groups.
    value : str, default="difference"
        The column of the measure.
    statistic : {"median", "mean"}, default="median"
        The statistic of each group; with ``"mean"``, the column ``median``
        of the result holds the mean.
    draws : int, default=1000
        Number of resamplings of the embryos.
    seed : int, default=0
        Seed of the random generator.
    level : float, default=0.9
        Level of the confidence intervals.

    Returns
    -------
    pandas.DataFrame
        One row per group, with the ``median`` of the measure, the bounds
        ``low`` and ``high`` of its confidence interval, the number of
        comparisons (``n``), of distinct cell names (``cells``) and of
        embryos (``embryos``). With fewer than three embryos, the interval
        is NaN: resampling two embryos says nothing about others.
    pandas.DataFrame
        The median of each group (columns) in each draw (rows).
    """
    embryos = sorted(set(table.embryo) | set(table.other))
    index = {e: i for i, e in enumerate(embryos)}
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(embryos), np.full(len(embryos), 1 / len(embryos)), size=draws
    )
    rows, samples = {}, {}
    for key, positions in table.groupby(by, sort=True).indices.items():
        group = table.iloc[positions]
        x = group[value].to_numpy(dtype=float)
        e = group.embryo.map(index).to_numpy()
        f = group.other.map(index).to_numpy()
        weights = np.where(e == f, counts[:, e], counts[:, e] * counts[:, f])
        if statistic == "mean":
            total = weights.sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                samples[key] = (weights @ x) / total
        else:
            samples[key] = _weighted_medians(x, weights)
        low, high = np.nanquantile(
            samples[key], [(1 - level) / 2, (1 + level) / 2]
        )
        if len(set(e) | set(f)) < 3:
            low = high = np.nan
        rows[key] = {
            "median": float(
                np.mean(x) if statistic == "mean" else np.median(x)
            ),
            "low": low,
            "high": high,
            "n": len(x),
            "cells": group.name.nunique(),
            "embryos": len(set(group.embryo) | set(group.other)),
        }
    summary = pd.DataFrame.from_dict(rows, orient="index")
    if isinstance(by, list) and len(by) > 1:
        summary.index = pd.MultiIndex.from_tuples(summary.index, names=by)
    else:
        summary.index.name = by if isinstance(by, str) else by[0]
    return summary, pd.DataFrame(samples)


def adjusted_effects(
    table,
    factors,
    covariates=(),
    value="difference",
    unit=("cell", "minutes"),
    draws=500,
    seed=0,
    min_units=5,
):
    """Effects of tissue, time or other factors on a variability measure.

    The comparisons are first averaged by unit (by default, a bilateral
    pair of cells at one time). The logarithm of the mean difference of
    each unit is then modelled as a sum of one effect per level of each
    factor, plus linear terms for the covariates, by least squares. The
    effects of a factor are centred, so that ``exp(effect)`` is the fold
    change of the variability of a level relative to the average level,
    the other factors and the covariates being equal. Their confidence
    intervals come from resampling the embryos, as in
    ``embryo_bootstrap``: the unit means are recomputed with the weights
    of each draw and the model refitted.

    Parameters
    ----------
    table : pandas.DataFrame
        Comparisons, with the columns ``embryo``, ``other``, ``value``, the
        columns of ``unit``, ``factors`` and ``covariates``.
    factors : list of str
        Categorical columns, e.g. ``["tissue", "period"]``.
    covariates : sequence of str, optional
        Numerical columns, e.g. the logarithm of the size of the cells,
        taken as their mean over each unit.
    value : str, default="difference"
        The column of the measure.
    unit : tuple of str, default=("cell", "minutes")
        The columns that define a unit.
    draws : int, default=500
        Number of resamplings of the embryos.
    seed : int, default=0
        Seed of the random generator.
    min_units : int, default=5
        Levels with fewer units are left out of the model.

    Returns
    -------
    pandas.DataFrame
        One row per factor and level, with the fold change (``fold``), the
        bounds ``low`` and ``high`` of its 90% confidence interval, and
        the number of ``units``. Covariates appear with ``factor`` equal to
        their name and the fold change for an increase of one.
    """
    unit = list(unit)
    data = table.dropna(subset=[value, *unit, *factors, *covariates])
    units = data.groupby(unit, sort=False).agg(
        {**{f: "first" for f in factors}, **{c: "mean" for c in covariates}}
    )
    for factor in factors:
        sizes = units[factor].value_counts()
        units = units[units[factor].isin(sizes[sizes >= min_units].index)]
    position = pd.Series(np.arange(len(units)), index=units.index)
    labels = (
        pd.MultiIndex.from_frame(data[unit])
        if len(unit) > 1
        else pd.Index(data[unit[0]])
    )
    rows_unit = position.reindex(labels).to_numpy()
    keep = ~np.isnan(rows_unit)
    data, rows_unit = data[keep], rows_unit[keep].astype(int)

    blocks, names = [np.ones((len(units), 1))], [("intercept", "")]
    for factor in factors:
        levels = sorted(units[factor].unique())
        blocks.append(
            (units[factor].to_numpy()[:, None] == np.array(levels)).astype(
                float
            )
        )
        names += [(factor, level) for level in levels]
    for covariate in covariates:
        x = units[covariate].to_numpy(dtype=float)
        blocks.append((x - x.mean())[:, None])
        names.append((covariate, ""))
    design = np.hstack(blocks)

    embryos = sorted(set(data.embryo) | set(data.other))
    index = {e: i for i, e in enumerate(embryos)}
    e = data.embryo.map(index).to_numpy()
    f = data.other.map(index).to_numpy()
    x = data[value].to_numpy(dtype=float)
    floor = 0.01 * np.median(x[x > 0])

    def fit(weights):
        """Fit the model with one set of weights of the comparisons.

        Parameters
        ----------
        weights : numpy.ndarray
            Weight of each comparison.

        Returns
        -------
        numpy.ndarray
            The centred coefficients, in the order of ``names``.
        """
        total = np.bincount(rows_unit, weights=weights, minlength=len(units))
        sums = np.bincount(
            rows_unit, weights=weights * x, minlength=len(units)
        )
        seen = total > 0
        y = np.log(sums[seen] / total[seen] + floor)
        coefficients = np.linalg.lstsq(design[seen], y, rcond=None)[0]
        start = 1
        for factor in factors:
            n = int(units[factor].nunique())
            block = coefficients[start : start + n]
            coefficients[start : start + n] = block - block.mean()
            start += n
        return coefficients

    estimate = fit(np.ones(len(x)))
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        len(embryos), np.full(len(embryos), 1 / len(embryos)), size=draws
    )
    boot = np.array(
        [
            fit(np.where(e == f, c[e], c[e] * c[f]).astype(float))
            for c in counts
        ]
    )
    low, high = np.quantile(boot, [0.05, 0.95], axis=0)
    out = pd.DataFrame(
        {
            "factor": [n[0] for n in names],
            "level": [n[1] for n in names],
            "fold": np.exp(estimate),
            "low": np.exp(low),
            "high": np.exp(high),
        }
    )
    out["units"] = [
        int((units[n[0]] == n[1]).sum()) if n[0] in factors else len(units)
        for n in names
    ]
    return out[out.factor != "intercept"].reset_index(drop=True)


def relatedness(first, second):
    """Number of generations back to the last common ancestor of two cells.

    Parameters
    ----------
    first, second : str
        Two cell names of the same generation.

    Returns
    -------
    int or None
        1 for sisters, 2 for first cousins, and so on, as long as the
        common ancestor has a name (from the 8-cell stage on); None
        otherwise.
    """
    gen = parse_name(first)[1]
    for back in range(1, gen - 3):
        try:
            if ancestor(first, gen - back) == ancestor(second, gen - back):
                return back
        except (TypeError, ValueError):
            return None
    return None


def related_deviations(values, relation):
    """Deviations of a measure in pairs of related cells of the same embryo.

    The deviation of a cell in an embryo is its value minus its median
    over the embryos.

    Parameters
    ----------
    values : pandas.DataFrame
        A measure, indexed by cell name, one column per embryo.
    relation : {"mother", "sister", "homologue"}
        The second cell of each pair: the mother of the first, its sister
        (each pair once), or its bilateral homologue (each pair once).

    Returns
    -------
    pandas.DataFrame
        One row per pair and embryo, with the columns ``name`` (the first
        cell), ``relative`` (the second), ``embryo``, ``first`` and
        ``second`` (their deviations).
    """
    deviations = values.sub(values.median(axis=1), axis=0)
    partner = {"mother": mother, "sister": sister, "homologue": homologue}[
        relation
    ]
    rows = []
    for name in deviations.index:
        other = partner(name)
        if other not in deviations.index:
            continue
        if relation == "sister" and parse_name(name)[2] % 2 == 0:
            continue
        if relation == "homologue" and not name.endswith("_"):
            continue
        both = pd.concat(
            [deviations.loc[name], deviations.loc[other]],
            axis=1,
            keys=["first", "second"],
        ).dropna()
        for e, (a, b) in both.iterrows():
            rows.append((name, other, e, a, b))
    return pd.DataFrame(
        rows, columns=["name", "relative", "embryo", "first", "second"]
    )


def deviation_correlations(
    values, tissues, times, window=10.0, by_embryo=False
):
    """Do cells of the same tissue deviate together within an embryo?

    For every pair of cells of the same generation and embryo whose times
    differ by less than ``window`` minutes, the deviations of a measure
    are paired. The deviation of a cell is its value minus its median over
    the embryos, minus the median deviation of the embryo in the same
    20-minute period, so that a late or early period of a whole embryo
    does not count. The pairs are classed by kinship and by whether the
    two cells belong to the same tissue.

    Parameters
    ----------
    values : pandas.DataFrame
        A measure, indexed by cell name, one column per embryo.
    tissues : pandas.Series
        The tissue of each cell name.
    times : pandas.Series
        The time of each cell name, in minutes.
    window : float, default=10.0
        Maximum time between the two cells of a pair.
    by_embryo : bool, default=False
        Give one correlation per embryo instead of pooling them.

    Returns
    -------
    pandas.DataFrame
        One row per class of pairs (and embryo), with the columns
        ``class``, ``embryo`` (``"all"`` when pooled), the Pearson
        ``correlation`` of the deviations and the number of ``pairs``. The
        classes are ``"left/right"`` (bilateral homologues), ``"sisters"``,
        ``"cousins"`` (a common grandmother), and, for more distant kin
        on the same side, ``"same tissue"`` and ``"different tissues"``.
    """
    deviations = values.sub(values.median(axis=1), axis=0)
    time = times.reindex(deviations.index)
    bins = (time // 20) * 20
    for e in deviations.columns:
        shift = deviations[e].groupby(bins).transform("median")
        deviations[e] = deviations[e] - shift
    names = [n for n in deviations.index if not np.isnan(time[n])]
    tissue = tissues.reindex(names).fillna("unknown")
    found = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if abs(time[a] - time[b]) > window:
                continue
            if parse_name(a)[1] != parse_name(b)[1]:
                continue
            if b == homologue(a):
                label = "left/right"
            elif a[-1] != b[-1]:
                continue
            else:
                kin = relatedness(a, b)
                if kin == 1:
                    label = "sisters"
                elif kin == 2:
                    label = "cousins"
                elif {tissue[a], tissue[b]} & {"mixed", "unknown"}:
                    continue
                elif tissue[a] == tissue[b]:
                    label = "same tissue"
                else:
                    label = "different tissues"
            pair = deviations.loc[[a, b]].T.dropna()
            for e, (x, y) in pair.iterrows():
                found.setdefault(label, []).append((e, x, y))
    rows = []
    for label, pairs in found.items():
        table = pd.DataFrame(pairs, columns=["embryo", "x", "y"])
        groups = table.groupby("embryo") if by_embryo else [("all", table)]
        for e, group in groups:
            rows.append(
                {
                    "class": label,
                    "embryo": e,
                    "correlation": float(np.corrcoef(group.x, group.y)[0, 1]),
                    "pairs": len(group),
                }
            )
    return pd.DataFrame(rows)


def landmark_lines(ax, landmarks):
    """Draw vertical lines at developmental landmarks.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The panel.
    landmarks : dict of {str: float}
        Label -> time in minutes post fertilization.
    """
    for label, x in landmarks.items():
        ax.axvline(x, color="0.75", lw=0.8, zorder=0)
        ax.text(
            x,
            1.0,
            f" {label}",
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="right",
            fontsize=7,
            color="0.45",
        )


def plot_time_profiles(profiles, labels, landmarks, min_n=20):
    """Small multiples of variability over developmental time.

    Parameters
    ----------
    profiles : dict of {str: pandas.DataFrame}
        For each measure, the result of ``embryo_bootstrap`` grouped by
        ``["comparison", "period"]``.
    labels : dict of {str: str}
        The axis label of each measure.
    landmarks : dict of {str: float}
        Developmental landmarks, in minutes post fertilization.
    min_n : int, default=20
        Periods with fewer comparisons are not drawn.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per measure: the median of the comparisons between
        embryos and between the two sides of an embryo, with their 90%
        confidence intervals over embryos. Hollow points rest on one or
        two embryos only, and have no interval.
    """
    import matplotlib.pyplot as plt

    columns = 4
    rows = int(np.ceil(len(profiles) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(15, 3.4 * rows), squeeze=False
    )
    for ax, (measure, summary) in zip(
        axes.flat, profiles.items(), strict=False
    ):
        for comparison, color in (
            ("between embryos", RECAP_COLORS["between embryos"]),
            ("left/right", RECAP_COLORS["left/right"]),
        ):
            if comparison not in summary.index.get_level_values(0):
                continue
            s = summary.loc[comparison]
            s = s[s.n >= min_n]
            several = s[s.embryos >= 3]
            ax.plot(s.index, s["median"], "-", color=color, lw=1)
            ax.plot(several.index, several["median"], "o", ms=3.5, color=color)
            ax.plot(
                s.index[s.embryos < 3],
                s["median"][s.embryos < 3],
                "o",
                ms=3.5,
                mfc="white",
                color=color,
            )
            ax.fill_between(s.index, s.low, s.high, color=color, alpha=0.15)
        landmark_lines(ax, landmarks)
        ax.set(title=measure, ylabel=labels[measure], ylim=(0, None))
        ax.set_xlabel("minutes post fertilization")
    for ax in axes.flat[len(profiles) :]:
        ax.set_visible(False)
    axes.flat[0].plot(
        [],
        [],
        color=RECAP_COLORS["between embryos"],
        label="same cell, two embryos",
    )
    axes.flat[0].plot(
        [],
        [],
        color=RECAP_COLORS["left/right"],
        label="same cell, two sides of one embryo",
    )
    axes.flat[0].legend(fontsize=7, loc="lower right")
    figure.tight_layout()
    return figure


def plot_tissue_profiles(profiles, labels, order):
    """Small multiples of variability by tissue.

    Parameters
    ----------
    profiles : dict of {str: pandas.DataFrame}
        For each measure, the result of ``embryo_bootstrap`` grouped by
        ``["comparison", "tissue"]``.
    labels : dict of {str: str}
        The axis label of each measure.
    order : list of str
        The tissues, from top to bottom.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per measure: the median and 90% confidence interval of
        each tissue, between embryos and between sides.
    """
    import matplotlib.pyplot as plt

    columns = 4
    rows = int(np.ceil(len(profiles) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(15, 3.6 * rows), squeeze=False, sharey=True
    )
    y = np.arange(len(order))
    for ax, (measure, summary) in zip(
        axes.flat, profiles.items(), strict=False
    ):
        for shift, comparison in (
            (-0.15, "between embryos"),
            (0.15, "left/right"),
        ):
            if comparison not in summary.index.get_level_values(0):
                continue
            s = summary.loc[comparison].reindex(order)
            color = RECAP_COLORS[comparison]
            ax.errorbar(
                s["median"],
                y + shift,
                xerr=[s["median"] - s.low, s.high - s["median"]],
                fmt="o",
                ms=4,
                color=color,
                lw=1,
            )
        overall = summary.loc["between embryos", "median"].median()
        ax.axvline(overall, color="0.8", lw=0.8, zorder=0)
        ax.set(title=measure, xlabel=labels[measure])
        ax.set_yticks(y, order)
        ax.invert_yaxis()
    for ax in axes.flat[len(profiles) :]:
        ax.set_visible(False)
    figure.tight_layout()
    return figure


def plot_tissue_time(maps, order, limit=1.0):
    """Small multiples of variability by tissue and period.

    Parameters
    ----------
    maps : dict of {str: pandas.DataFrame}
        For each measure, the log2 fold change of the median variability
        of each tissue (rows) and period (columns) over the median of the
        measure in that period; NaN where there are too few data.
    order : list of str
        The tissues, from top to bottom.
    limit : float, default=1.0
        The colour scale goes from ``-limit`` to ``limit``.

    Returns
    -------
    matplotlib.figure.Figure
        One heatmap per measure, red where a tissue is more variable than
        the other cells at the same time, blue where it is less.
    """
    import matplotlib.pyplot as plt

    columns = 4
    rows = int(np.ceil(len(maps) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(15, 3.4 * rows), squeeze=False
    )
    for ax, (measure, values) in zip(axes.flat, maps.items(), strict=False):
        values = values.reindex(order)
        image = ax.imshow(
            values.to_numpy(dtype=float),
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            aspect="auto",
            interpolation="nearest",
        )
        ax.set_yticks(range(len(order)), order, fontsize=8)
        ticks = range(0, values.shape[1], 2)
        ax.set_xticks(
            ticks, [f"{values.columns[i]:.0f}" for i in ticks], fontsize=8
        )
        ax.set(title=measure, xlabel="minutes post fertilization")
        ax.set_facecolor("0.93")
    for ax in axes.flat[len(maps) :]:
        ax.set_visible(False)
    figure.tight_layout()
    figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        fraction=0.015,
        label="log2 fold over all the cells at that time",
    )
    return figure


def plot_effects(effects, factor, order=None):
    """Small multiples of the adjusted effects of one factor.

    Parameters
    ----------
    effects : dict of {str: pandas.DataFrame}
        For each measure, the result of ``adjusted_effects``.
    factor : str
        The factor to show, e.g. ``"tissue"`` or ``"period"``.
    order : list, optional
        The levels, in order; all the levels found by default.

    Returns
    -------
    matplotlib.figure.Figure
        One panel per measure, with the fold change of each level and its
        90% confidence interval over embryos, on a log scale. Categorical
        factors are drawn as dot plots, numerical ones as lines.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import (
        FixedLocator,
        FuncFormatter,
        NullLocator,
    )

    columns = 4
    rows = int(np.ceil(len(effects) / columns))
    numeric = factor == "period"
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(15, 3.4 * rows),
        squeeze=False,
        sharey=not numeric,
        sharex=numeric,
    )
    color = RECAP_COLORS["between embryos"]
    levels = order
    if levels is None:
        levels = sorted(
            set().union(
                *[set(e[e.factor == factor].level) for e in effects.values()]
            )
        )
    for ax, (measure, table) in zip(axes.flat, effects.items(), strict=False):
        s = table[table.factor == factor].set_index("level").reindex(levels)
        significant = (s.low > 1) | (s.high < 1)
        if numeric:
            x = np.array(levels, dtype=float)
            ax.plot(x, s.fold, "o-", color=color, ms=3, lw=1.5)
            ax.fill_between(x, s.low, s.high, color=color, alpha=0.15)
            ax.set_xlabel("minutes post fertilization")
        else:
            y = np.arange(len(levels))
            ax.errorbar(
                s.fold,
                y,
                xerr=[s.fold - s.low, s.high - s.fold],
                fmt="none",
                ecolor=color,
                lw=1,
            )
            ax.scatter(
                s.fold,
                y,
                s=30,
                zorder=3,
                color=[color if v else "white" for v in significant],
                edgecolor=color,
            )
            ax.set_yticks(y, levels)
            ax.invert_yaxis()
            ax.set_xlabel("fold change of the variability")
        reference = ax.axhline if numeric else ax.axvline
        reference(1, color="0.6", lw=0.8, zorder=0)
        axis = ax.yaxis if numeric else ax.xaxis
        (ax.set_yscale if numeric else ax.set_xscale)("log", base=2)
        axis.set_major_locator(FixedLocator(FOLD_TICKS))
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_locator(NullLocator())
        ax.set_title(measure)
    for ax in axes.flat[len(effects) :]:
        ax.set_visible(False)
    if numeric:
        for ax in axes[:-1].flat:
            ax.set_xlabel("")
    figure.tight_layout()
    return figure


def quantization_floor(table, alignment, events=1, draws=20, seed=0):
    """Median difference produced by the time points alone.

    Event times are known to one time point. If two embryos had exactly
    the same timing, their measured times would still differ by up to one
    time point of each, uniformly.

    Parameters
    ----------
    table : pandas.DataFrame
        Comparisons between embryos, with the columns ``embryo`` and
        ``other``.
    alignment : TimeAlignment
        The alignment of the embryos.
    events : int, default=1
        Number of event times in the measure: 1 for a division time, 2
        for a cycle length (birth and division).
    draws : int, default=20
        Number of simulated differences per comparison.
    seed : int, default=0
        Seed of the random generator.

    Returns
    -------
    float
        The median absolute difference, in reference minutes.
    """
    rng = np.random.default_rng(seed)
    a = table.embryo.map(alignment.frame_minutes).to_numpy()
    b = table.other.map(alignment.frame_minutes).to_numpy()
    noise = np.zeros((draws, len(a)))
    for _ in range(events):
        noise += a * rng.uniform(-0.5, 0.5, noise.shape)
        noise -= b * rng.uniform(-0.5, 0.5, noise.shape)
    return float(np.median(np.abs(noise)))


def sister_swaps(manager, alignment, reference_time, embryos):
    """Distances of matched cells to their match and to its sister.

    After registration of two embryos, as in ``position_deviations``, a
    cell lies at some distance from the cell of the same name in the
    other embryo. If the names of two sisters were swapped in one of the
    embryos, the cell would lie closer to the sister of its match.

    Parameters
    ----------
    manager : LineageTreeManager
        The embryos.
    alignment : TimeAlignment
        Their alignment.
    reference_time : float
        The reference time point.
    embryos : list of str
        The embryos to compare.

    Returns
    -------
    pandas.DataFrame
        One row per cell whose sister is also found, and pair of embryos,
        with the columns ``comparison`` (``"between embryos"``), ``name``,
        ``embryo``, ``other``, ``difference`` (the distance to the cell of
        the same name, in typical cell radii) and ``to sister`` (the
        distance to its sister).
    """
    frames = {e: frame_at(alignment, e, reference_time) for e in embryos}
    frames = {e: t for e, t in frames.items() if t is not None}
    cells = {e: positions(manager[e], t) for e, t in frames.items()}
    rows = []
    for e, f in _pairs(frames):
        unit = embryo_radius(manager[e], frames[e]) / len(
            manager[e].time_nodes[frames[e]]
        ) ** (1 / 3)
        names, x, y = common_cells(cells[e], cells[f])
        mapped = similarity_fit(y, x)[0]
        index = {n: i for i, n in enumerate(names)}
        for name, i in index.items():
            other = sister(name)
            if other not in index:
                continue
            own = np.linalg.norm(mapped[i] - x[i]) / unit
            swapped = np.linalg.norm(mapped[index[other]] - x[i]) / unit
            rows.append(("between embryos", name, e, f, own, swapped))
    return pd.DataFrame(
        rows,
        columns=[
            "comparison",
            "name",
            "embryo",
            "other",
            "difference",
            "to sister",
        ],
    )
