# Refactoring

This is the first try as refactoring

## Naming changes

On top of the refactoring some major changes have been made in the naming, the library is now named `lineagetree` instead of `LineageTree`. The lineage tree class is now named `LineageTree` instead of `lineageTree`.

So, for example, when one used to do:

```python
from LineageTree import lineageTree
```

it should now do:

```python
from lineagetree import LineageTree
```

This is to comply with the [PEP8 naming conventions](https://peps.python.org/pep-0008/#package-and-module-names).

## The new structure

The goal is to breakdown the once huge `LineageTree` class code into multiple files such that there is no 3k line of code file.

Here is the proposed breakdown:

A `_core.py` file that contains the class together with the `__init__` function and very few little functions.

A set of files that contains `LineageTree` methods grouped according to what they do:

- `_dynamic_time_warping.py` (all DTW function)
- `_loaders.py` (all loaders)
- `_modifiers.py` (all modifier functions)
- `_navigation.py` (all _navigation_ functions)
- `_plot.py` (all plotting functions)
- `_properties.py` (all properties)
- `_spatial.py` (all spatial functions)
- `_uted.py` (all uted functions)
- `_writers.py` (all writting functions)
- `lineage_tree_manager.py` (lineage tree manager class)
- `tree_approximation.py` (lineage tree approximation classes)
- `utils.py` (some useful functions)

and a `_assembly.py` file that attaches all the functions to the `LineageTree` class.

## One side effect

One side effect that the new structure allows is the following:

```python
from lineagetree import LineageTree
from lineagetree._plot import some_function

lT = LineageTree.load("...")

# Both calls are identical:
some_function(lT, ...)
# or:
lT.some_function(...)
```

## Discussion

Potential points of discussion:

- Is the scheme `_core.py`, `_function_type.py`, `_assembly.py` good?
- Is the proposed split good (ie, would you have grouped functions differently, ...)?
- Is the file naming good? (for example, right now I don't like `_navigation.py`)
- Should we have a folder structure and therefore sub-categories?
- Should all the function files indeed be private `_` in front of the name of the file (making the side effect above more or less accessible)?
- As of right now the functions are declared as follow:

```python
def some_function(lT: LineageTree, ...) -> None:
    ...
```

to specify that they are class functions, should they be shifted back to:

```python
def some_function(self, ...) -> None:
    ...
```

- ...

## Remaining to be done

- Having better coverage for the tests
Current coverage:

```text
Name                                       Stmts   Miss  Cover
--------------------------------------------------------------
src/lineagetree/__init__.py                    6      0   100%
src/lineagetree/_assembly.py                  59      0   100%
src/lineagetree/_core.py                     146     18    88%
src/lineagetree/_dynamic_time_warping.py     118     14    88%
src/lineagetree/_loaders.py                  507    403    21%
src/lineagetree/_modifier.py                  88     19    78%
src/lineagetree/_navigation.py               137     29    79%
src/lineagetree/_plot.py                     186     45    76%
src/lineagetree/_properties.py               140     40    71%
src/lineagetree/_spatial.py                   82      2    98%
src/lineagetree/_uted.py                     191     34    82%
src/lineagetree/_writers.py                  250    228     9%
src/lineagetree/lineage_tree_manager.py      268     73    73%
src/lineagetree/test/test_lineageTree.py     275      1    99%
src/lineagetree/test/test_uted.py             66      5    92%
src/lineagetree/tree_approximation.py        253     15    94%
src/lineagetree/utils.py                      66      5    92%
--------------------------------------------------------------
TOTAL                                       2838    931    67%
```

- Checking naming and documentation consistency
- ...
