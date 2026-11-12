# Experiment results, 2026-11-09

Runs from `notebooks/experiments.ipynb` (RTX 5060 Ti, bf16, seed 0 unless stated). Each run
folder holds its log, config, history, results and plots; `error_analysis.*` comes from
`notebooks/pipeline.ipynb`.

**Setup.** Training: 30k size-15 mazes (10% for validation) from DFS, Prim's and Kruskal's,
20 epochs, batch 64, lr 1e-3 (3e-4 with loops), 30 iterations, progressive loss α = 0.01.
Testing: unseen generators (Eller's, Wilson's, hunt-and-kill, recursive division,
sidewinder), 250 / 250 / 200 / 100 / 40 / 16 mazes at sizes 15–511, so 255 and 511 are rough
(±8 and ±12 points).

**Metric.** A maze is solved if every passage pixel is right. The target marks every pixel
on some shortest path, so with loops all tied routes count. Numbers are the best over 30 /
100 / 300 / 1000 / 3000 iterations.

## Results

| Run | Val | Perfect 31 | 63 | 127 | 255 | 511 | Loops 15 | 31 | 63 | 127 |
|---|---|---|---|---|---|---|---|---|---|---|
| `dtnet_s15_noloops` | 100% | 100% | 100% | 99% | 92.5% | 62.5% | 0% | 0% | 0% | 0% |
| `dtnet_s15_noloops_zeroinit` | 100% | 100% | **100%** | **100%** | **100%** | **100%** | 0% | 0% | 0% | 0% |
| `dtnet_s15_dfs` (DFS only) | 100% | 90.8% | 55.5% | 12% | 0% | 0% | 0% | 0% | 0% | 0% |
| `dtnet_s15_loops` (lr 3e-4) | 96.4% | 70.8% | 16.5% | 5% | 2.5% | 0% | 94.4% | 54% | 14.5% | 5% |
| `dtnet_s15_loops_zeroinit` (lr 3e-4)³ | 96.5% | 62.4% | 16% | 5% | 2.5% | 0% | 93.6% | 54.8% | 16% | 4% |
| `ff_s15_noloops` (30 blocks) | 100% | 28.8% | 0% | 0% | 0% | 0% | 3.2% | 0% | 0% | 0% |

The loops test uses 10% of walls removed. All runs solve 95.6–100% of size-15 perfect mazes.
Validation for the loops runs includes mazes with loops. ³ All at 30 iterations; it
collapses with more (size-15 perfect: 100% at 30, 17.2% at 100, 2% at 300).

Iterations `dtnet_s15_noloops_zeroinit` needs (share solved):

| Size | 30 | 100 | 300 | 1000 |
|---|---|---|---|---|
| 63 | 7% | 99% | 100% | 100% |
| 127 | 0% | 43% | 100% | 100% |
| 255 | 0% | 0% | 72.5% | 100% |
| 511 | 0% | 0% | 18.8% | 100% |

## Findings

1. **DTNet extrapolates by thinking longer.** Trained at size 15, the zero-init run solves
   every test maze up to 511 (34x the training side). Larger mazes need more iterations:
   about 100 at size 63, 300 at 127 and 1000 at 255–511.
2. **Weight tying is what makes this possible.** The feedforward baseline fits size 15 but
   solves 28.8% at 31 and nothing larger.
3. **Mixing generators helps.** Trained on DFS only, DTNet drops to 55.5% at 63 and 0% at 255.
4. **Loops: no procedure that scales.** Without loops in training, DTNet fills dead ends and
   keeps every loop, so it fails every loop maze. With loops in training it fits size 15 but
   extrapolates poorly and inconsistently: 16.5–56% at perfect 63 over three seeds.
5. **Zero-init helps without loops, hurts with them.** Without loops it looks better (100% vs
   62.5% at 511; largest loss spike 118 vs 8e7; one run each). With loops it matches
   `dtnet_s15_loops` at 30 iterations, then falls apart with more iterations (0% at 300 on
   every size above 15). So it learned a procedure that only works for the training depth.
6. **lr 1e-3 is unstable.** Loss spikes and validation collapses happen in most runs. Saving
   the best checkpoint handles them without loops, but with loops they can be fatal. Loop runs
   use lr 3e-4.
7. **Tied shortest paths are common with loops.** At 10% removal, 26% of size-15 and 68% of
   size-63 mazes have several (up to 960). So the target marks them all.

## Training stability (DTNet with loops)

Fixes were screened with 8-epoch runs on seeds 0–2 (not kept). Largest single-step loss:

| Change | Seed 0 / 1 / 2 |
|---|---|
| none (lr 1e-3) | 34 / 1.3e6 / 5.7e3 |
| cuDNN benchmark off | 5.8e6 / NaN / 1.1e8 |
| fp32 | 1.2e6 / NaN / 1.0e9 |
| zero-init | 5.0 / 36 / 4.4e5 |
| skip steps with grad norm > 10x median | 988 / – / 421 |
| **lr 3e-4** | **1.4 / 1.5 / 3.4** |

The spikes are not numerical noise. On spiking batches the recurrence becomes expansive: the
hidden state reaches 1e7–1e31 after 300 iterations. lr 3e-4 removes the spikes at no cost in
validation accuracy.

Full runs (seeds 1–2 tested up to size 127):

| Run | lr | Seed | Val | Perfect 31 | 63 | 127 | Loops 15 | 31 | 63 | 127 |
|---|---|---|---|---|---|---|---|---|---|---|
| `dtnet_s15_loops_lr1e-3` | 1e-3 | 0 | 93.9% | 72.8% | 24% | 7% | 87.6% | 60% | 19.5% | 4% |
| `explore_loops_s1` | 1e-3 | 1 | 91.2% | 82.8% | 10.5% | 1% | 88% | 57.6% | 10% | 1% |
| `explore_loops_s2` (diverged) | 1e-3 | 2 | 69.2% | 46% | 13% | 6% | 63.6% | 33.2% | 7.5% | 3% |
| `stab_base_s2` | 1e-3 | 2 | 96.9% | 93.6% | 45.5% | 13% | 92.4% | 69.6% | 24% | 8% |
| `dtnet_s15_loops` | 3e-4 | 0 | 96.4% | 70.8% | 16.5% | 5% | 94.4% | 54% | 14.5% | 5% |
| `explore_loops_lr3e-4_s1` | 3e-4 | 1 | 97.4% | 94.4% | 56% | 19% | 94.4% | 70.4% | 31% | 11% |
| `stab_lr3e-4_s2` | 3e-4 | 2 | 97.2% | 96% | 48.5% | 17% | 92.8% | 70% | 23.5% | 5% |

At lr 1e-3 the same configuration diverged once and trained cleanly once (seed 2), so GPU
nondeterminism alone changes the outcome. At lr 3e-4 all seeds train cleanly, but
extrapolation still varies widely (16.5–56% at 63).

## Error analysis

Wrong pixels of failed test mazes at 300 iterations:

| Run | Test mazes | Size | Failed | Connected¹ | One shortest path² | Missed forced | Missed tied | Extra route | Extra dead end |
|---|---|---|---|---|---|---|---|---|---|
| `dtnet_s15_noloops` | 10% removed | 31 | 200/200 | 198 | 0 | 0% | 0% | 100% | 0% |
| `dtnet_s15_noloops` | 10% removed | 63 | 100/100 | 98 | 0 | 0% | 0% | 100% | 0% |
| `dtnet_s15_loops` | perfect | 31 | 61/200 | 2 | – | 85% | – | – | 15% |
| `dtnet_s15_loops` | perfect | 63 | 83/100 | 0 | – | 89% | – | – | 11% |
| `dtnet_s15_loops` | 10% removed | 31 | 93/200 | 37 | 11 | 37% | 23% | 39% | 0% |
| `dtnet_s15_loops` | 10% removed | 63 | 87/100 | 13 | 4 | 42% | 29% | 29% | 0% |

Categories:

- **Missed forced:** a gap in pixels that every shortest path uses.
- **Missed tied:** a tied shortest route left out.
- **Extra route:** a longer route kept around a loop.
- **Extra dead end:** a dead end left unfilled.

Category columns are shares of wrong pixels. ¹ Still connects start and exit.
² Exactly one valid shortest path.

- **Without loops in training,** DTNet keeps everything that survives dead-end filling.
- **With loops in training,** it lost dead-end filling and learned something local. On perfect
  mazes 85–89% of the errors are gaps.
- **The wide target is not the main problem:** only 4–11 failed loop mazes per size are a
  single valid shortest path.
