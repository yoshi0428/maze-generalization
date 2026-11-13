# Experiment results, 2026-11-09

This document gives the results of the runs from `notebooks/experiments.ipynb`. It uses the
ASD-STE100 Simplified Technical English standard. Run names and code names stay as they are in
the code.

All runs used one RTX 5060 Ti GPU and bf16. All runs used seed 0, if the text does not give a
different seed.

Each run has a folder in `results/`. The folder contains the log, the configuration, the
history, the results and the plots. The files `error_analysis.*` come from
`notebooks/pipeline.ipynb`.

## Setup

**Training data.** 30,000 mazes of size 15. The generators are DFS, Prim's and Kruskal's. The
code keeps 10% of the mazes for validation.

**Training settings.**

| Setting | Value |
| --- | --- |
| Epochs | 20 |
| Batch size | 64 |
| Learning rate | 1e-3. Runs with loops use 3e-4. |
| Iterations | 30 |
| Progressive loss weight (α) | 0.01 |

**Test data.** The test mazes come from generators that the training does not use: Eller's,
Wilson's, hunt-and-kill, recursive division and sidewinder.

| Size | 15 | 31 | 63 | 127 | 255 | 511 |
| --- | --- | --- | --- | --- | --- | --- |
| Number of mazes | 250 | 250 | 200 | 100 | 40 | 16 |

The test sets for sizes 255 and 511 are small. Thus, the results for these sizes have a large
error: approximately ±8 points at size 255 and ±12 points at size 511.

**Metric.** A maze is "solved" when the model gives the correct class to each passage pixel.
The target marks each pixel that is on a shortest path. If a maze has loops, it can have more
than one shortest path. Then, the target marks all of these paths.

Each value in the tables is the best result for 30, 100, 300, 1000 and 3000 iterations.

## Results

"Perfect" mazes have no loops. "Loops" mazes have 10% of the walls removed.

| Run | Val | Perfect 31 | 63 | 127 | 255 | 511 | Loops 15 | 31 | 63 | 127 |
|---|---|---|---|---|---|---|---|---|---|---|
| `dtnet_s15_noloops` | 100% | 100% | 100% | 99% | 92.5% | 62.5% | 0% | 0% | 0% | 0% |
| `dtnet_s15_noloops_zeroinit` | 100% | 100% | **100%** | **100%** | **100%** | **100%** | 0% | 0% | 0% | 0% |
| `dtnet_s15_dfs` (DFS only) | 100% | 90.8% | 55.5% | 12% | 0% | 0% | 0% | 0% | 0% | 0% |
| `dtnet_s15_loops` (lr 3e-4) | 96.4% | 70.8% | 16.5% | 5% | 2.5% | 0% | 94.4% | 54% | 14.5% | 5% |
| `dtnet_s15_loops_zeroinit` (lr 3e-4)³ | 96.5% | 62.4% | 16% | 5% | 2.5% | 0% | 93.6% | 54.8% | 16% | 4% |
| `ff_s15_noloops` (30 blocks) | 100% | 28.8% | 0% | 0% | 0% | 0% | 3.2% | 0% | 0% | 0% |

Notes:

- All runs solve 95.6% to 100% of the perfect mazes of size 15.
- For the runs with loops, the validation set also contains mazes with loops.
- ³ All the results for this run are at 30 iterations. With more iterations, the results
  become much worse. For perfect mazes of size 15, the run solves 100% at 30 iterations, 17.2%
  at 100 iterations and 2% at 300 iterations.

This table shows the number of iterations that `dtnet_s15_noloops_zeroinit` needs. Each value
is the fraction of solved mazes.

| Size | 30 | 100 | 300 | 1000 |
|---|---|---|---|---|
| 63 | 7% | 99% | 100% | 100% |
| 127 | 0% | 43% | 100% | 100% |
| 255 | 0% | 0% | 72.5% | 100% |
| 511 | 0% | 0% | 18.8% | 100% |

## Findings

1. **DTNet solves larger mazes when it does more iterations.** The zero-init run trains on
   mazes of size 15. It solves all the test mazes up to size 511. Size 511 is 34 times the
   training size. Larger mazes need more iterations: approximately 100 iterations at size 63,
   300 at size 127 and 1000 at sizes 255 and 511.

2. **Shared weights are necessary for this result.** The feedforward model solves the mazes
   of size 15. But it solves only 28.8% at size 31, and it solves no larger mazes.

3. **Different generators in the training data improve the result.** DTNet with only DFS
   mazes solves 55.5% at size 63 and 0% at size 255.

4. **With loops, DTNet does not learn a method that works on larger mazes.**
   - If the training data has no loops, DTNet fills the dead ends and keeps all the loops.
     Thus, it fails on all the mazes with loops.
   - If the training data has loops, DTNet solves the mazes of size 15. But on larger mazes,
     the results are bad and they change much between runs. For perfect mazes of size 63,
     three seeds give 16.5% to 56%.

5. **Zero-init improves the result without loops, but it makes the result worse with
   loops.**
   - Without loops, the result is better: 100% against 62.5% at size 511. The largest loss
     spike is 118 against 8e7. Each value comes from only one run.
   - With loops, the result is the same as `dtnet_s15_loops` at 30 iterations. With more
     iterations, the result becomes very bad: 0% at 300 iterations on all sizes larger than
     15. Thus, the model learned a method that works only for the training number of
     iterations.
   - For a description of zero-init, see section 1.7 in [TRAINING.md](TRAINING.md).

6. **A learning rate of 1e-3 is not stable.** Most runs have loss spikes and drops in the
   validation result. Without loops, the code keeps the best checkpoint, so these problems
   have no effect. With loops, these problems can stop the training. Thus, the runs with loops
   use a learning rate of 3e-4.

7. **With loops, many mazes have more than one shortest path.** When 10% of the walls are
   removed, 26% of the mazes of size 15 and 68% of the mazes of size 63 have more than one
   shortest path. One maze has 960 shortest paths. Thus, the target marks all the shortest
   paths.

## Training stability (DTNet with loops)

We tested possible fixes with runs of 8 epochs on seeds 0, 1 and 2. We did not keep these runs.
This table gives the largest loss of one step:

| Change | Seed 0 / 1 / 2 |
|---|---|
| None (lr 1e-3) | 34 / 1.3e6 / 5.7e3 |
| cuDNN benchmark off | 5.8e6 / NaN / 1.1e8 |
| fp32 | 1.2e6 / NaN / 1.0e9 |
| Zero-init | 5.0 / 36 / 4.4e5 |
| Skip a step if the gradient norm is more than 10 times the median | 988 / – / 421 |
| **lr 3e-4** | **1.4 / 1.5 / 3.4** |

Numerical noise does not cause the spikes. On a batch with a spike, each iteration makes the
hidden state larger. After 300 iterations, the hidden state is between 1e7 and 1e31. A learning
rate of 3e-4 removes the spikes. The validation result does not become worse.

This table gives the full runs. For seeds 1 and 2, the tests stop at size 127.

| Run | lr | Seed | Val | Perfect 31 | 63 | 127 | Loops 15 | 31 | 63 | 127 |
|---|---|---|---|---|---|---|---|---|---|---|
| `dtnet_s15_loops_lr1e-3` | 1e-3 | 0 | 93.9% | 72.8% | 24% | 7% | 87.6% | 60% | 19.5% | 4% |
| `explore_loops_s1` | 1e-3 | 1 | 91.2% | 82.8% | 10.5% | 1% | 88% | 57.6% | 10% | 1% |
| `explore_loops_s2` (diverged) | 1e-3 | 2 | 69.2% | 46% | 13% | 6% | 63.6% | 33.2% | 7.5% | 3% |
| `stab_base_s2` | 1e-3 | 2 | 96.9% | 93.6% | 45.5% | 13% | 92.4% | 69.6% | 24% | 8% |
| `dtnet_s15_loops` | 3e-4 | 0 | 96.4% | 70.8% | 16.5% | 5% | 94.4% | 54% | 14.5% | 5% |
| `explore_loops_lr3e-4_s1` | 3e-4 | 1 | 97.4% | 94.4% | 56% | 19% | 94.4% | 70.4% | 31% | 11% |
| `stab_lr3e-4_s2` | 3e-4 | 2 | 97.2% | 96% | 48.5% | 17% | 92.8% | 70% | 23.5% | 5% |

At a learning rate of 1e-3, we trained the same configuration two times with seed 2. One run
diverged and one run did not. Thus, the GPU nondeterminism alone can change the result.

At a learning rate of 3e-4, the training is stable for all seeds. But the result on larger
mazes changes much between seeds: 16.5% to 56% at size 63.

## Error analysis

This table shows the incorrect pixels in the failed test mazes at 300 iterations.

| Run | Test mazes | Size | Failed | Connected¹ | One shortest path² | Missed forced | Missed tied | Extra route | Extra dead end |
|---|---|---|---|---|---|---|---|---|---|
| `dtnet_s15_noloops` | 10% removed | 31 | 200/200 | 198 | 0 | 0% | 0% | 100% | 0% |
| `dtnet_s15_noloops` | 10% removed | 63 | 100/100 | 98 | 0 | 0% | 0% | 100% | 0% |
| `dtnet_s15_loops` | perfect | 31 | 61/200 | 2 | – | 85% | – | – | 15% |
| `dtnet_s15_loops` | perfect | 63 | 83/100 | 0 | – | 89% | – | – | 11% |
| `dtnet_s15_loops` | 10% removed | 31 | 93/200 | 37 | 11 | 37% | 23% | 39% | 0% |
| `dtnet_s15_loops` | 10% removed | 63 | 87/100 | 13 | 4 | 42% | 29% | 29% | 0% |

¹ The number of failed mazes in which the prediction still connects the start and the exit.

² The number of failed mazes that have exactly one shortest path.

The last four columns give the percentage of incorrect pixels in each category:

- **Missed forced:** a gap in pixels that all shortest paths use.
- **Missed tied:** a shortest path that the model did not mark.
- **Extra route:** a longer path around a loop that the model marked.
- **Extra dead end:** a dead end that the model did not fill.

Conclusions:

- **If the training data has no loops,** DTNet keeps all the pixels that remain after it fills
  the dead ends.
- **If the training data has loops,** DTNet does not fill the dead ends. It learned a local
  method. On perfect mazes, 85% to 89% of the errors are gaps.
- **The target with more than one shortest path is not the main problem.** For each size, only
  4 to 11 of the failed mazes with loops have exactly one shortest path.
