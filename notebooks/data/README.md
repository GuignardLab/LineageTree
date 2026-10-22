# Datasets of the C. elegans notebooks

The notebooks of this folder load their data with `celegans_tools.fetch`,
which follows the method of napari-relax for its demo datasets:

1. the datasets are described in `datasets.json`: for each one, its file, its
   download URL, its MD5 checksum, a description and its size, in the format
   of the `datasets.json` of napari-relax;
2. `fetch(name)` looks for the file in this folder, and downloads it from its
   URL if it is missing or if its checksum does not match;
3. it checks the MD5 checksum of the download, and decompresses `.xz` files
   once.

The files themselves are not in the repository (see `.gitignore`).

| name | file | size | content |
| --- | --- | --- | --- |
| `C.elegans` | `c.elegans.smoothed.lT` | 41.6 MB | Embryo WT-EMB08 of [Li et al. 2019](https://doi.org/10.1016/j.celrep.2018.12.052), with cell positions and, mapped on its cells, 104 tissue-reporter recordings of the RNAi screen of [Du et al. 2015](https://doi.org/10.1016/j.devcel.2015.07.014). On [Zenodo](https://zenodo.org/records/17061487). |
| `C.elegans_wild_types` | `celegans_bao_wild_types.lTM.xz` | 3.8 MB | 30 wild-type embryos lineaged by the Bao lab, 10 per reporter strain. Not online yet. |
| `C.elegans_RNAi_screen` | `celegans_bao_rnai_screen.lTM.xz` | 112 MB | The 1,322 embryos of the RNAi screen of Du et al. 2015: 204 genes knocked down, about 6 embryos per gene, with one of the three reporters. Not online yet. |

## The collections of the Bao lab

Each collection is a `LineageTreeManager` saved with `write` and compressed
with xz; `fetch` returns the decompressed `.lTM` file, read with
`LineageTreeManager.load`. Each embryo is a `LineageTree` named after the
file of the lab it comes from, with:

- the cell names (Sulston nomenclature) as labels;
- times in frames (`time_resolution` 1), starting at 0 for P0;
- the node properties `expression`, the value of the reporter at each node,
  and `binary_call`, the call of the lab for each cell (0 or 1, copied to
  all the nodes of the cell);
- the attributes `gene` (the gene knocked down, or `"wild type"`), `reporter`
  (`"PHA"`, `"HYP"` or `"NEU"`), `marker` (the reporter gene), `strain` and
  `source_file`.

| reporter | marker | strain | tissue |
| --- | --- | --- | --- |
| PHA | pha-4 | RW10425 | pharynx (and gut) |
| HYP | nhr-25 | RW10348 | hypodermis |
| NEU | cnd-1 | RW10434 | neurons |

The lineages have no cell positions: the files of the lab give the cell
names and the reporter values at each time point only.

The collections are made from the text files of the lab with:

```bash
python prepare_bao_datasets.py RAW_DIR
```

where `RAW_DIR` holds `WT/info_*.txt` and
`Cell-Lineage-With-Expression/GENE_REPORTER_EMBRYO.txt`. The script reads
each file with `lineagetree.read_from_txt_for_celegans_BAO`, adds the binary
calls and the metadata, writes the two `.lTM` files and their `.xz`
archives here, and updates `datasets.json` (checksums and sizes; the URLs
are kept). The output is the same from one run to the next, so the
checksums are stable.

## To put the collections online

1. Check with the Bao lab that the data can be redistributed, and under
   which license (CC BY 4.0, for instance). The text files were distributed
   by the lab at digital-development.org.
2. Create a Zenodo record with the two `.xz` files, as for record 17061487.
   A draft of its description is below.
3. Write the download URLs in `datasets.json`, in the form
   `https://zenodo.org/records/RECORD/files/FILE?download=1`.

`.lT` and `.lTM` files are pickles, which can run code when they are loaded:
the MD5 check of `fetch` makes sure that the file is the one described in
the registry.

### Draft of the Zenodo description

**Title:** *C. elegans* embryonic lineages of a tissue-reporter RNAi screen,
as lineagetree files

**Description:** Lineages of *C. elegans* embryos from the Bao lab, recorded
with one of three tissue reporters: pha-4 (pharynx, strain RW10425), nhr-25
(hypodermis, RW10348) or cnd-1 (neurons, RW10434). Two files:

- `celegans_bao_wild_types.lTM.xz`: 30 wild-type embryos, 10 per reporter;
- `celegans_bao_rnai_screen.lTM.xz`: the 1,322 embryos of the RNAi screen of
  Du et al. (2015), in which 204 genes were knocked down, about 6 embryos per
  gene.

Each file is a `LineageTreeManager` of the lineagetree library, compressed
with xz. Each embryo is a lineage tree with the names of the cells, the
reporter value at each time point and the binary expression call of each
cell, from the text files of the lab. The files are used by the notebooks of
lineagetree (`notebooks/lineagetree_showcase.ipynb`).

**Related identifiers:** Du Z, Santella A, He F, Shah PK, Kamikawa Y, Bao Z.
The regulatory landscape of lineage differentiation in a metazoan embryo.
*Dev. Cell* 34, 592–607 (2015), doi:10.1016/j.devcel.2015.07.014.
lineagetree: https://github.com/GuignardLab/LineageTree.
