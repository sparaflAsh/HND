# Hierarchical Novelty Detection

You can find the theoretical explanations [here](https://link.springer.com/chapter/10.1007/978-3-319-68765-0_26)

## Usage

To test on a simple toy data just run ```python HND.py```

The module can be imported and the optimization function can be used.

```python
from HND import *
Optimization(Data, Hierarchy)
```

The optimization can be also run with slacks. By default the two arguments fraction_novel and root are set to None.

```Optimization(data, hierarchy, fraction_novel=None, root=None)```

```fraction_novel``` may be set to a ```float``` representing the percentage of novelties (ex. 0.1) the user wish to discover and ```root``` is a ```string``` or ```int``` representing the root node of the DAG if known. This last option may shorten execution time.

Input:

Data is a dictionary with two keys:
```
>>> data.keys()
dict_keys(['features', 'tags'])
```

Hierarchy is a list of tuples representing edges.

Output: 

Always a dictionary with the following structure.

```
>>> out.keys()
dict_keys(['Radii', 'Means', 'Slacks'])
```
