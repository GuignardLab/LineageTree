# LineageTree

This library allows to import and work with cell (but not limited to cells) lineage trees.
With LineageTree you can read from:

- TGMM algorithm outputs described in [Amat et al. 2014](https://www.nature.com/articles/nmeth.3036)
- TrackMate files described in [Tinevez et al. 2017](https://doi.org/10.1016/j.ymeth.2016.09.016)
- MaMuT files described in [Wolff et al. 2018](https://doi.org/10.7554/eLife.34410)
- SVF algorithm outputs described in [McDole, Guignard et al. 2018](https://doi.org/10.1016/j.cell.2018.09.031)
- ASTEC algorithm outputs described in [Guignard, Fiuza et al. 2020](https://doi.org/10.1126/science.aar5663)
- Data from the [Digital development Database](http://digital-development.org/index.html) described in [Du et al. 2014](https://www.cell.com/fulltext/S0092-8674(13)01542-0) and [Du et al. 2015](https://www.sciencedirect.com/science/article/pii/S1534580715004876?via%3Dihub)
- and few others

## Basic usage

Once installed the library can be called the following way (as an example):

```python
from lineagetree import LineageTree
```

and one can then load lineage trees the following way:

For `.lT` files:

```python
lT = LineageTree.load('path/to/file.lT')
```

For ASTEC data:

```python
from lineagetree import read_from_ASTEC
lT = read_from_ASTEC('path/to/ASTEC.pkl')
```

For MaMuT or TrackMate:

```python
from lineagetree import read_from_mamut_xml
lT = read_from_mamut_xml('path/to/MaMuT.xml')
```

For TGMM:

```python
from lineagetree import read_from_tgmm_xml
lT = read_from_tgmm_xml('path/to/single_time_file{t:04d}.xml', tb=0, te=500)
```

For Mastodon:

```python
from lineagetree import read_from_mastodon
lT = read_from_mastodon('path/to/Mastodon.mastodon')
```

or, for Mastodon csv file:

```python
from lineagetree import read_from_mastodon_csv
lT = read_from_mastodon_csv(['path/to/nodes.csv', 'path/to/links.csv'])
```

## Quick install

To quickly install the library together with its dependencies one can run:

```shell
pip install LineageTree
```

or, for the latest version if you have cloned the directory:

```shell
pip install .
```

or for the latest version wihtout cloning the directory

```shell
pip install git+https://github.com/leoguignard/LineageTree
```
