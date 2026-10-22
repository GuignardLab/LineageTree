"""Convert the C. elegans lineages of the Bao lab into lineagetree datasets.

The notebooks of this folder use two collections of lineages from the Bao
lab, recorded with tissue reporters: 30 wild-type embryos, and the 1,322
embryos of the RNAi screen of Du et al., Dev. Cell 34, 592 (2015), each with
one of 204 genes knocked down. This script turns the text files of the lab
into one ``LineageTreeManager`` file per collection, compressed with xz, and
describes them in ``datasets.json``, the dataset registry of napari-relax:
file name, download URL, MD5 checksum, description and size. Once the files
are uploaded (on Zenodo, for instance), their URLs go into the registry, and
``celegans_tools.fetch`` downloads them on first use.

Each embryo is a ``LineageTree`` named after its file, with:

- the expression of the reporter at each node (``expression``), and the
  binary call of the lab for each cell (``binary_call``, copied to all the
  nodes of the cell), as node properties;
- the cell names as labels, and times in frames (``time_resolution`` 1);
- its metadata as attributes: ``gene`` (the gene knocked down, or
  ``"wild type"``), ``reporter`` (``"PHA"``, ``"HYP"`` or ``"NEU"``),
  ``marker`` (the reporter gene: pha-4, nhr-25 or cnd-1), ``strain`` and
  ``source_file``.

Usage::

    python prepare_bao_datasets.py RAW_DIR [--output DIR]

``RAW_DIR`` holds ``WT/info_*.txt`` (the wild types) and
``Cell-Lineage-With-Expression/*.txt`` (the screen, one file per embryo,
named ``GENE_REPORTER_EMBRYO.txt``).
"""

import argparse
import hashlib
import json
import lzma
import shutil
from pathlib import Path

from lineagetree import LineageTreeManager, read_from_txt_for_celegans_BAO

HERE = Path(__file__).resolve().parent
STRAINS = {"RW10348": "HYP", "RW10425": "PHA", "RW10434": "NEU"}
MARKERS = {"PHA": "pha-4", "HYP": "nhr-25", "NEU": "cnd-1"}
COLLECTIONS = {
    "C.elegans_wild_types": {
        "filename": "celegans_bao_wild_types.lTM",
        "description": (
            "30 wild-type C. elegans embryos lineaged by the Bao lab with a "
            "tissue reporter (10 per strain: pha-4, nhr-25, cnd-1), as a "
            "LineageTreeManager, xz-compressed"
        ),
    },
    "C.elegans_RNAi_screen": {
        "filename": "celegans_bao_rnai_screen.lTM",
        "description": (
            "The 1,322 C. elegans embryos of the RNAi screen of Du et al. "
            "2015 (204 genes, tissue reporters pha-4, nhr-25, cnd-1), as a "
            "LineageTreeManager, xz-compressed"
        ),
    },
}


def binary_calls(path):
    """Read the binary expression call of each cell of a Bao lab file.

    Parameters
    ----------
    path : pathlib.Path
        A text file of the lab: one line per cell, tab-separated, with the
        cell name first and the binary call third.

    Returns
    -------
    dict of {str: int}
        Maps each cell name to its call, 0 or 1.
    """
    calls = {}
    with open(path) as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if fields[0] != "cell_name":
                calls[fields[0]] = int(fields[2])
    return calls


def read_embryo(path, gene, reporter):
    """Read one embryo and attach its binary calls and metadata.

    Parameters
    ----------
    path : pathlib.Path
        The text file of the embryo.
    gene : str
        The gene knocked down, or ``"wild type"``.
    reporter : str
        The tissue of the reporter: ``"PHA"``, ``"HYP"`` or ``"NEU"``.

    Returns
    -------
    LineageTree
        The lineage of the embryo, named after its file, with the
        ``expression`` and ``binary_call`` node properties and the metadata
        attributes described in the docstring of the module.
    """
    lT = read_from_txt_for_celegans_BAO(path)
    calls = binary_calls(path)
    lT.binary_call = {n: calls[lT.labels[n]] for n in lT.nodes}
    lT.time_resolution = 1
    lT.gene = gene
    lT.reporter = reporter
    lT.marker = MARKERS[reporter]
    lT.strain = next(s for s, r in STRAINS.items() if r == reporter)
    lT.source_file = path.name
    return lT


def read_wild_types(folder):
    """Read the wild-type embryos.

    Parameters
    ----------
    folder : pathlib.Path
        Folder of the ``info_*.txt`` files, whose names contain the strain,
        e.g. ``info_ZD_RW10348_WT_20110126_2_s1_emb2_edited.txt``.

    Returns
    -------
    LineageTreeManager
        The embryos, sorted by name.
    """
    manager = LineageTreeManager()
    for path in sorted(folder.glob("info_*.txt")):
        strain = next(s for s in STRAINS if s in path.stem)
        manager.add(read_embryo(path, "wild type", STRAINS[strain]), path.stem)
    return manager


def read_screen(folder):
    """Read the embryos of the RNAi screen.

    Parameters
    ----------
    folder : pathlib.Path
        Folder of the ``GENE_REPORTER_EMBRYO.txt`` files.

    Returns
    -------
    LineageTreeManager
        The embryos, sorted by name.
    """
    manager = LineageTreeManager()
    for path in sorted(folder.glob("*.txt")):
        gene, reporter, _ = path.stem.rsplit("_", 2)
        manager.add(read_embryo(path, gene, reporter), path.stem)
    return manager


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


def compress(path):
    """Compress a file with xz, next to it.

    Parameters
    ----------
    path : pathlib.Path
        The file to compress; it is kept.

    Returns
    -------
    pathlib.Path
        The compressed file, ``path`` with ``.xz`` appended.
    """
    target = path.with_name(path.name + ".xz")
    with open(path, "rb") as source, lzma.open(target, "wb", preset=9) as sink:
        shutil.copyfileobj(source, sink, 1 << 20)
    return target


def update_registry(registry_path, name, path, description):
    """Describe a dataset file in the registry, in the format of napari-relax.

    The URL is kept if the entry exists, and set to None otherwise: it is
    filled in once the file is uploaded.

    Parameters
    ----------
    registry_path : pathlib.Path
        The ``datasets.json`` file; it is created if missing.
    name : str
        Name of the dataset in the registry.
    path : pathlib.Path
        The file of the dataset, as it will be uploaded.
    description : str
        A description of the dataset.

    Returns
    -------
    dict
        The entry of the dataset: ``filename``, ``url``, ``md5``,
        ``description`` and ``size_mb``.
    """
    registry = (
        json.loads(registry_path.read_text()) if registry_path.exists() else {}
    )
    previous = registry.get(name, {})
    entry = {
        "filename": path.name,
        "url": previous.get("url"),
        "md5": md5sum(path),
        "description": description,
        "size_mb": round(path.stat().st_size / 2**20, 2),
    }
    if previous.get("url") and previous.get("md5") != entry["md5"]:
        print(
            f"  warning: the MD5 of {path.name} changed; "
            "the uploaded file is out of date"
        )
    registry[name] = entry
    registry_path.write_text(json.dumps(registry, indent=2) + "\n")
    return entry


def main(argv=None):
    """Build the two collections, compress them and update the registry.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; ``sys.argv`` by default.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "raw_dir",
        type=Path,
        help="folder with WT/ and Cell-Lineage-With-Expression/",
    )
    parser.add_argument(
        "--output", type=Path, default=HERE, help="where to write the files"
    )
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    readers = {
        "C.elegans_wild_types": (read_wild_types, args.raw_dir / "WT"),
        "C.elegans_RNAi_screen": (
            read_screen,
            args.raw_dir / "Cell-Lineage-With-Expression",
        ),
    }
    for name, (reader, folder) in readers.items():
        manager = reader(folder)
        path = args.output / COLLECTIONS[name]["filename"]
        manager.write(str(path))
        archive = compress(path)
        entry = update_registry(
            HERE / "datasets.json",
            name,
            archive,
            COLLECTIONS[name]["description"],
        )
        print(
            f"{name}: {len(list(manager))} embryos, "
            f"{path.stat().st_size / 2**20:.0f} MB, "
            f"{archive.name} {entry['size_mb']} MB, md5 {entry['md5']}"
        )


if __name__ == "__main__":
    main()
