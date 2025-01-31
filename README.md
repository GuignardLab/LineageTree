# LineageTree

This library allows to import and work with cell (but not limited to cells) lineage trees.
With LineageTree you can read from:

- TGMM algorithm outputs described in [Fernando et al. 2014](https://www.nature.com/articles/nmeth.3036)
- TrackMate files described in [Tinevez et al. 2017](https://doi.org/10.1016/j.ymeth.2016.09.016)
- MaMuT files described in [Wolff et al. 2018](https://doi.org/10.7554/eLife.34410)
- SVF algorithm outputs described in [McDole, Guignard et al. 2018](https://doi.org/10.1016/j.cell.2018.09.031)
- ASTEC algorithm outputs described in [Guignard, Fiuza et al. 2020](https://doi.org/10.1126/science.aar5663)
- Data from the [[Digital development Database](http://digital-development.org/index.html)] described in [[Du et al. 2014](https://www.cell.com/fulltext/S0092-8674(13)01542-0)] and [[Du et al. 2015](https://www.sciencedirect.com/science/article/pii/S1534580715004876?via%3Dihub)]
- and few others

## Basic usage

Once installed the library can be called the following way (as an example):

```python
from LineageTree import lineageTree
```

and one can then load lineage trees the following way:

For `.lT` files:

```python
lT = lineageTree('path/to/file.lT')
```

For ASTEC data:

```python
lT = lineageTree('path/to/ASTEC.pkl', file_type='ASTEC')
```

or

```python
lT = lineageTree('path/to/ASTEC.xml', file_type='ASTEC')
```

For SVF:

```python
lT = lineageTree('path/to/SVF.bin')
```

For MaMuT:

```python
lT = lineageTree('path/to/MaMuT.xml', file_type='MaMuT')
```

For TrackMate:

```python
lT = lineageTree('path/to/MaMuT.xml', file_type='TrackMate')
```

For TGMM:

```python
lT = lineageTree('path/to/single_time_file{t:04d}.xml', tb=0, te=500, file_type='TGMM')
```

For Mastodon:

```python
lT = lineageTree('path/to/Mastodon.mastodon', file_type='mastodon')
```

or, for Mastodon csv file:

```python
lT = lineageTree(['path/to/nodes.csv', 'path/to/links.csv'], file_type='mastodon')
```

## Dependencies

Some dependecies are requiered:

- general python dependecies:
  - numpy, scipy
- specific dependency:
  - svgwrite if svg output is needed

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
