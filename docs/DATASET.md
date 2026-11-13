# The maze dataset

This document describes the maze data and the code that makes it. It uses the ASD-STE100
Simplified Technical English standard. Names from the code stay as they are in the code.

The data code is in three locations:

| File | Contents |
| --- | --- |
| [src/generators/](../src/generators/) | The maze generators and the wall removal. |
| [src/dataset.py](../src/dataset.py) | The PyTorch dataset, the targets and the data splits. |
| [src/runner.py](../src/runner.py) | The training data, the evaluation sets and the data cache for one run. |

The notebook `notebooks/dataset_generation_ex.ipynb` shows examples of the mazes.

## 1. The maze format

A maze has an odd size `N`, for example 15, 63 or 127. The code keeps a maze as a NumPy array
of `uint8` values:

- The array has the shape `(N + 2, N + 2)`. A wall of 1 pixel goes around the maze.
- The value 1 (`PASSAGE`) is an open pixel. The value 0 (`WALL`) is a wall.

A cell is a pixel with odd coordinates for the row and for the column. The maze has
`(N + 1) / 2` × `(N + 1) / 2` cells. Between two adjacent cells, there is one pixel. If this
pixel is open, there is a passage between the two cells.

Example: a maze of size 15 is a 17 × 17 image with 8 × 8 cells.

## 2. The generators

### 2.1 The available generators

All generators make "perfect" mazes. A perfect maze has exactly one path between two cells.
It has no loops.

| Name in the code | Algorithm | Default use |
| --- | --- | --- |
| `dfs` | Depth-first search (recursive backtracker) | Training |
| `prim` | Prim's algorithm | Training |
| `kruskal` | Kruskal's algorithm | Training |
| `eller` | Eller's algorithm | Test |
| `wilson` | Wilson's algorithm | Test |
| `hunt_and_kill` | Hunt-and-kill | Test |
| `recursive_division` | Recursive division | Test |
| `sidewinder` | Sidewinder | Test |
| `aldous_broder` | Aldous-Broder | Not used |

Each algorithm makes mazes with a different structure. For example, some algorithms make long
corridors, and other algorithms make many short dead ends. Thus, a test on different
generators shows if the model learned a general method.

### 2.2 How to use a generator

```python
from generators import get_generator

maze = get_generator("dfs")(15, rng)   # rng is a numpy.random.Generator
```

The function `get_generator` gives the generator for a name. If the name is not correct, it
gives a `ValueError` with the list of the available names.

A generator accepts only an odd size of 1 or more. If the size is not correct, it gives a
`ValueError`.

### 2.3 How to add a generator

1. Make a new module in `src/generators/`.
2. In this module, make a subclass of `MazeGenerator`.
3. Give the subclass a unique `name`.
4. Write the method `_carve`. This method opens the passages in the grid. The helper `carve`
   opens two cells and the pixel between them.
5. Add the class to `GENERATORS` in `src/generators/__init__.py`.

### 2.4 The wall removal (`remove_walls`)

The function `remove_walls` adds loops to a perfect maze. It opens a fraction of the walls
between cells. It does not change the original maze. It gives a copy.

The function opens only the walls between two adjacent cells. It does not open these pixels:

- The pixels at the intersections of walls.
- The wall around the maze.

Thus, the result is a maze with loops, and not a large open area.

Wall removal only opens pixels. Thus, a path from the start to the exit always exists after
wall removal.

## 3. One sample

Each sample is a pair of float tensors: `(input, target)`. `H` and `W` are both `N + 2`.

| Tensor | Shape | Contents |
| --- | --- | --- |
| `input` | `(3, H, W)` | Channel 0: the maze (1 = passage, 0 = wall). Channel 1: the start (1 at one pixel). Channel 2: the exit (1 at one pixel). |
| `target` | `(1, H, W)` | 1 on each pixel of a shortest path from the start to the exit. The start and the exit are on the path. |

A batch from a `DataLoader` adds the batch dimension `B` at the start of each shape.

### 3.1 The start and the exit

The function `random_endpoints` selects two different random cells. The start and the exit
are always cells. They are never the pixels between cells.

The code selects the start and the exit before the wall removal. Thus, the same maze has the
same start and exit with or without wall removal.

### 3.2 The target (`shortest_path`)

The function `shortest_path` makes the target. It uses two breadth-first searches: one from
the start and one from the exit. A pixel is on a shortest path if this equation is true:

```
distance(start, pixel) + distance(pixel, exit) = distance(start, exit)
```

A perfect maze has only one shortest path. Thus, the target is this path.

A maze with loops can have more than one shortest path of the same length. The target marks
all of these paths. Thus, the target depends only on the maze. It does not depend on the
order in which the search examines the pixels.

### 3.3 How to examine a prediction (`is_shortest_path`)

The function `is_shortest_path` examines a predicted path. The prediction is correct if one
of these conditions is true:

- It is equal to the full target.
- It is exactly one shortest path. All its pixels are passages, it connects the start to the
  exit, and it has no extra pixels.

Use this function for a method that gives only one path, for example a BFS or A* baseline.
The models in this repository use the full target.

## 4. The dataset class (`MazeDataset`)

### 4.1 Parameters

| Parameter | Description |
| --- | --- |
| `generators` | The list of generator names. A name can occur only one time. |
| `size` | The odd maze size `N`. It must be 3 or more. |
| `num_samples` | The number of mazes. |
| `seed` | The base seed. |
| `wall_removal` | The fraction of walls to open. The value 0 gives perfect mazes. A pair `(min, max)` gives a random fraction from this range for each maze. Thus, one dataset can contain perfect mazes and mazes with loops. |
| `exclude_generators` | The generators that must not be in `generators`. If a generator is in both lists, the class gives a `ValueError`. |

### 4.2 How the dataset makes a maze

The dataset does not keep the mazes in memory or on the disk. It makes each maze when you ask
for it. Sample `i` uses these items:

- The generator `generators[i % len(generators)]`. Thus, each generator makes the same number
  of mazes, with a maximum difference of one maze.
- A random generator with the seed `(seed, i)`. Thus, sample `i` is always the same maze.

This method has two advantages:

- You can make the same dataset again from its parameters.
- `DataLoader` workers do not share data.

### 4.3 Methods and properties

| Name | Description |
| --- | --- |
| `len(ds)` | The number of mazes. |
| `ds[i]` | The sample `i` as `(input, target)`. |
| `ds.generate(i)` | The maze array, the start and the exit for sample `i`, as NumPy data. Use it to show or examine a maze. |
| `ds.generator_index(i)` | The index of the generator for sample `i`. |
| `ds.labels` | The generator index of each sample, in the same order as `ds.generator_names`. |

## 5. The helper functions in `dataset.py`

### 5.1 `make_train_dataset`

This function makes a `MazeDataset` for training. The default generators are `dfs`, `prim`
and `kruskal`. You can give one name, for example `"dfs"`, to use only one generator.

### 5.2 `make_test_datasets`

This function makes one test dataset for each size. It gives a dictionary with the size as
the key.

- It uses `train_generators` as `exclude_generators`. Thus, a test generator cannot also be a
  training generator.
- Each size has a different seed: `seed × 1,000,003 + size`. Thus, the mazes of different
  sizes are not related.
- Each size has its own dataset, because a batch can contain only one image size.

### 5.3 `stratified_split`

This function divides a dataset into a training subset and a validation subset. It divides
the mazes of each generator separately. Thus, the two subsets have the same mix of generators
as the full dataset.

The value `val_fraction` must be more than 0 and less than 1.

## 6. The data for one run (`runner.py`)

### 6.1 `materialize`

The function `materialize` makes all the samples of a dataset one time. It keeps them in
memory as a `TensorDataset`.

- It keeps the values as `uint8`, not as float. This decreases the memory by a factor of 4.
  The training code changes the values to float for each batch.
- It uses a maximum of 8 `DataLoader` workers to make the mazes in parallel.
- It starts the workers before it shows the progress bar. If the progress bar is first, the
  workers do not start correctly in Jupyter.

Without `materialize`, the code makes each maze again in each epoch. The breadth-first search
for the target is slow, so this takes much time.

### 6.2 The training data (`training_data`)

The function `training_data` gives the training set and the validation set for a `Config`. It
does these steps:

1. It makes the training dataset with `make_train_dataset`.
2. It divides the dataset with `stratified_split`.
3. It makes all the samples with `materialize`.

The function keeps the result of the most recent call in memory (`lru_cache`). If the next
run uses the same data settings, the function does not make the mazes again. Optuna trials
use this function, because all trials use the same data.

The relevant `Config` settings are `train_generators`, `train_size`, `n_train`, `train_wr`,
`val_fraction` and `seed`.

### 6.3 The evaluation data

There are two types of evaluation data:

| Type | Function | Generators | Seed |
| --- | --- | --- | --- |
| Extrapolation | `_extrap_dataset` | The training generators | `10,000 + seed` |
| Test | `_test_dataset` | The test generators | `seed + 1`, then the formula in 5.2 |

The extrapolation sets contain larger mazes from the training generators. They do not use the
test generators. Thus, you can use them to tune the settings, and the test results stay
independent.

### 6.4 The list of evaluation sets (`eval_sets`)

The function `eval_sets` gives a dictionary of all the evaluation sets of a `Config`. Each item
has this form:

```
result key: (function that makes the dataset, number of mazes, batch size)
```

The keys have these forms:

| Key | Contents | `Config` setting |
| --- | --- | --- |
| `extrap/<size>` | Extrapolation mazes of this size. | `extrap_sizes` |
| `<wall_removal>/<size>` | Test mazes of this size with this wall removal fraction. | `test_sizes`, `test_wr` |

The settings `extrap_sizes` and `test_sizes` are dictionaries of this form:
`{size: (number of mazes, batch size)}`.

The code makes test sets with loops only up to the size `loop_test_max_size` (default 127).
These tests are slow, and the models solve almost none of these mazes.

The function does not make the datasets. It gives only the functions that make them. Thus, if
a set already has results, the code does not make its mazes.
