## Quick installation

To quickly install the library together with its dependencies one can run:

```shell
pip install LineageTree
```

or, for the latest version if you have cloned the directory:

```shell
cd 'path/to/folder'
pip install .
```

or for the latest version wihtout cloning the directory

```shell
pip install git+https://github.com/leoguignard/LineageTree
```

## Basic usage

Once installed the library can be called the following way (as an example):

```python
from LineageTree import lineageTree
```

and one can then load lineage trees the following way:

For `.lT` files:

```python
lT = lineageTree.load('path/to/file.lT')
```

For ASTEC data:

```python
from LineageTree import read_from_ASTEC
lT = read_from_ASTEC('path/to/ASTEC.pkl')
```

For MaMuT or TrackMate:

```python
from LineageTree import read_from_mamut_xml
lT = read_from_mamut_xml('path/to/MaMuT.xml')
```

For TGMM:

```python
from LineageTree import read_from_tgmm_xml
lT = read_from_tgmm_xml('path/to/single_time_file{t:04d}.xml', tb=0, te=500)
```

For Mastodon:

```python
from LineageTree import read_from_mastodon
lT = read_from_mastodon('path/to/Mastodon.mastodon')
```

or, for Mastodon csv file:

```python
from LineageTree import read_from_mastodon_csv
lT = read_from_mastodon_csv(['path/to/nodes.csv', 'path/to/links.csv'])
```

