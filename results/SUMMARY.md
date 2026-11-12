# Experiment results, 2026-11-09

All runs: `notebooks/experiments.ipynb` on an RTX 5060 Ti 16 GB with bf16 autocast, seed 0. Each
run folder holds `log.txt`, `config.json`, `history.json`, `results.json`, `curves.png`
(train/val loss, val solved) and `test.png`; `error_analysis.json` / `.png` come from
the error analysis in `notebooks/pipeline.ipynb`.

Common settings: 30k training mazes of size 15 (10% for validation), 20 epochs, batch 64, lr 1e-3
(3e-4 for runs with loops in training, see Training stability), 2 warmup epochs then cosine
decay, clip 2, 30 training iterations, progressive loss alpha 0.01. Training generators: dfs,
prim, kruskal (unless stated). Test generators (unseen): eller, wilson, hunt-and-kill, recursive
division, sidewinder.

"Solved" = every passage pixel correct. The target marks every pixel on *some* shortest path
from start to exit (in a perfect maze this is the unique path; with loops, all tied routes).
Numbers are the best over test iterations (30 / 100 / 300 / 1000 / 3000); per-iteration numbers
are in the logs. Test mazes per size: 250 (15, 31), 200 (63), 100 (127), 40 (255), 16 (511), so
the 255 and 511 numbers are rough (±8 and ±12 points at one standard error). Each configuration
was run once, except DTNet with loops (three seeds, see Training stability).

## Perfect test mazes (no loops)

| Run | Model | Loops in training | Val | 31 | 63 | 127 | 255 | 511 |
|---|---|---|---|---|---|---|---|---|
| `dtnet_s15_noloops` | DTNet | no | 100% | 100% | 100% | 99% | 92.5% | 62.5% |
| `dtnet_s15_noloops_zeroinit` | DTNet, zero-init | no | 100% | 100% | **100%** | **100%** | **100%** | **100%** |
| `dtnet_s15_dfs` | DTNet, DFS only | no | 100% | 90.8% | 55.5% | 12% | 0% | 0% |
| `dtnet_s15_loops` | DTNet, lr 3e-4 | 0–10% | 96.4%¹ | 70.8% | 16.5% | 5% | 2.5% | 0% |
| `ff_s15_noloops` | Feedforward, 30 blocks | no | 100% | 28.8% | 0% | 0% | 0% | 0% |

All runs solve 95.6–100% of size-15 test mazes. ¹ validation includes mazes with loops.

Iterations needed by `dtnet_s15_noloops_zeroinit`:

| Size | 30 | 100 | 300 | 1000 | 3000 |
|---|---|---|---|---|---|
| 63 | 7% | 99% | 100% | 100% | 100% |
| 127 | 0% | 43% | 100% | 100% | 100% |
| 255 | 0% | 0% | 72.5% | 100% | 100% |
| 511 | 0% | 0% | 18.8% | 100% | 100% |

## Test mazes with loops (10% of walls removed)

| Run | 15 | 31 | 63 | 127 |
|---|---|---|---|---|
| `dtnet_s15_noloops` | 0% | 0% | 0% | 0% |
| `dtnet_s15_noloops_zeroinit` | 0% | 0% | 0% | 0% |
| `dtnet_s15_dfs` | 0% | 0% | 0% | 0% |
| `ff_s15_noloops` | 3.2% | 0% | 0% | 0% |
| `dtnet_s15_loops` | 94.4% | 54% | 14.5% | 5% |

## Findings

1. **DTNet trained at size 15 extrapolates to much larger mazes by running more iterations.**
   Without loops in training it solves all size-63 test mazes from unseen generators. The
   zero-init run solves every test maze up to 511 (34 times the training side length) after 1000
   iterations; the default-init run reaches 99% at 127, 92.5% at 255 and 62.5% at 511.
   Larger mazes need more iterations: about 100 for 63, 300 for 127 and 1000 for 255 and 511.

2. **Weight tying is what makes this possible.** The feedforward baseline (the same blocks
   without weight tying, 30 blocks deep) fits the training size (100% validation and size-15
   test) but solves only 28.8% at 31 and nothing larger: it cannot run more iterations.

3. **Mixing generators helps (proposal step 3).** DTNet trained on DFS mazes only, with the same
   settings and number of mazes, does clearly worse on unseen generators: 55.5% vs 100% at size
   63, 12% vs 99% at 127, and 0% at 255 and 511.

4. **Loops: DTNet learns no procedure that scales.** Without loops in training, DTNet fills dead
   ends and keeps every loop, so it fails every loop test. With loops in training it fits the
   training size (96–97% validation) but extrapolates poorly, and very differently between seeds:
   16.5–56% of perfect size-63 mazes and 14.5–31% of loop size-63 mazes over three seeds. Most
   failures are paths with gaps (error analysis below).

5. **Zero-init residual blocks look better.** Starting the last conv of each block at zero gave
   the only run that solves all sizes up to 511 (16/16 vs 10/16 at 511, 40/40 vs 37/40 at 255),
   and a much smaller largest training-loss spike (118 vs 8e7). One run each, so this needs
   2–3 seeds before it is a result.

6. **Training is unstable at lr 1e-3.** Loss spikes (up to 8e7 in `dtnet_s15_noloops`) and
   validation collapses below 50% (noloops epoch 5, zero-init epoch 4) happen in most DTNet runs
   and recover; saving the best checkpoint by validation handles it. With loops in training the
   spikes can be fatal (1 of 4 full runs diverged), so the loops runs use lr 3e-4, which removes them
   (Training stability). The feedforward and DFS-only runs had no collapses.

7. **Shortest paths in mazes with loops are often tied.** At 10% wall removal, 26% of size-15 and
   68% of size-63 test mazes have several equally short routes (up to 960); at 30% removal, 90%
   of size-63 mazes do (`dataset_generation_ex.ipynb`). This is why the target marks every
   shortest path; a single BFS path would be an arbitrary choice.

## Training stability (DTNet with loops)

Plain DTNet trained with loops, logging every step's loss and, every 50 steps, the largest
hidden-state value after 30 and 300 iterations on a fixed batch. The fixes were first screened
with 8-epoch runs (the usual 20-epoch lr schedule, stopped after epoch 8; spikes start in epochs
1–9) on seeds 0–2. Those screening runs were not kept; their largest single-step losses were:

| Change | Largest step loss (seeds 0 / 1 / 2) |
|---|---|
| none (lr 1e-3) | 34 / 1.3e6 / 5.7e3 |
| cuDNN benchmark off | 5.8e6 / NaN / 1.1e8 |
| fp32 (no bf16 autocast) | 1.2e6 / NaN / 1.0e9 |
| zero-init | 5.0 / 36 / 4.4e5 |
| skip step if grad norm > 10x recent median | 988 / – / 421 |
| **lr 3e-4** | **1.4 / 1.5 / 3.4** |

- **The spikes are not numerical noise.** fp32 and turning off cuDNN autotuning are no better
  (2 of 6 runs went to NaN). The recurrence becomes expansive while training at lr 1e-3: on
  spiking batches the hidden state reaches 1e2–1e4 after 30 iterations and 1e7–1e31 after 300.
  Late in training it settles (1–3 even after 300 iterations) in every run that survives.
- **lr 3e-4 removes the spikes** on all three seeds (no step loss above 3.4, vs up to 1e6) and
  reaches the same validation accuracy by epoch 8. Zero-init helps on two seeds. Skipping
  high-gradient steps hurts: it skipped 470–620 steps and the loss stayed high.

Full 20-epoch runs with loops in training (seeds 1–2 tested up to size 127):

| Run | lr | Seed | Val | Perfect 31 | 63 | 127 | Loops 15 | 31 | 63 | 127 |
|---|---|---|---|---|---|---|---|---|---|---|
| `dtnet_s15_loops_lr1e-3` | 1e-3 | 0 | 93.9% | 72.8% | 24% | 7% | 87.6% | 60% | 19.5% | 4% |
| `explore_loops_s1` | 1e-3 | 1 | 91.2% | 82.8% | 10.5% | 1% | 88% | 57.6% | 10% | 1% |
| `explore_loops_s2` (diverged) | 1e-3 | 2 | 69.2% | 46% | 13% | 6% | 63.6% | 33.2% | 7.5% | 3% |
| `stab_base_s2` (no spikes this time) | 1e-3 | 2 | 96.9% | 93.6% | 45.5% | 13% | 92.4% | 69.6% | 24% | 8% |
| `dtnet_s15_loops` | 3e-4 | 0 | 96.4% | 70.8% | 16.5% | 5% | 94.4% | 54% | 14.5% | 5% |
| `explore_loops_lr3e-4_s1` | 3e-4 | 1 | 97.4% | 94.4% | 56% | 19% | 94.4% | 70.4% | 31% | 11% |
| `stab_lr3e-4_s2` | 3e-4 | 2 | 97.2% | 96% | 48.5% | 17% | 92.8% | 70% | 23.5% | 5% |

- **Runs at lr 1e-3 are chaotic.** The same configuration (seed 2) diverged once
  (`explore_loops_s2`) and trained without a spike once (`stab_base_s2`); GPU nondeterminism is
  enough to change the outcome. A run that diverges extrapolates much worse (13% vs 45.5% at 63).
- **At lr 3e-4 all three seeds train cleanly** (96.4–97.4% validation), but extrapolation still
  varies a lot between seeds (16.5–56% at 63). Seed 0 is the weakest and also overthinks: it
  solves 100% of size-15 perfect mazes at 30 iterations and 95.6% from 100 on.

## Auxiliary distance output (tried and removed)

DTNet with an extra head output for each pixel's BFS distance from the start and from the exit
(mod 3), meant to push it towards a wavefront that handles loops. Trained with loops at lr 3e-4
on seeds 0–2, it was close to plain DTNet on loop mazes (size 63: 40 / 28.5 / 19.5% vs 14.5 / 31
/ 23.5%; size 31: 66% vs 65% on average) and never learned the distances with loops (65–75% of
pixels right at size 15, no maze fully right). Without loops it learned the distances along the
tree but extrapolated worse (42.5% vs 92.5% at 255). The code and runs were deleted.

## Error analysis (`notebooks/pipeline.ipynb`)

Wrong pixels of failed test mazes at 300 iterations, split into: gaps in pixels every shortest
path uses (missed forced), dropped tied routes (missed tied), longer routes kept around loops
(extra route) and unfilled dead ends (extra dead end).

| Run | Test mazes | Size | Failed | Connected¹ | One shortest path² | Missed forced | Missed tied | Extra route | Extra dead end |
|---|---|---|---|---|---|---|---|---|---|
| `dtnet_s15_noloops` | 10% removed | 31 | 200/200 | 198 | 0 | 0% | 0% | 100% | 0% |
| `dtnet_s15_noloops` | 10% removed | 63 | 100/100 | 98 | 0 | 0% | 0% | 100% | 0% |
| `dtnet_s15_loops` | perfect | 31 | 61/200 | 2 | – | 85% | – | – | 15% |
| `dtnet_s15_loops` | perfect | 63 | 83/100 | 0 | – | 89% | – | – | 11% |
| `dtnet_s15_loops` | 10% removed | 31 | 93/200 | 37 | 11 | 37% | 23% | 39% | 0% |
| `dtnet_s15_loops` | 10% removed | 63 | 87/100 | 13 | 4 | 42% | 29% | 29% | 0% |

Category columns are the share of wrong pixels. ¹ failed predictions that still connect start and
exit. ² failed predictions that are exactly one correct shortest path (only these would pass a
looser "any shortest path" metric).

- **Without loops in training**, DTNet marks everything that survives dead-end filling: correct
  for perfect mazes, wrong for every maze with a loop.
- **With loops in training**, it lost dead-end filling and learned something local: on perfect
  mazes 85–89% of the wrong pixels are gaps and no failed size-63 prediction reaches the exit.
- **The wide target (all tied routes) is not the main problem:** only 4–11 failed loop mazes per
  size are a single valid shortest path.

## Next steps

1. Repeat the key comparisons (mixed vs DFS only, zero-init vs default) with 2–3 seeds; the
   loops runs show that one seed can be far off.
2. Classical baselines for proposal step 4: BFS and A* on CPU, BFS as repeated 3x3 dilation on GPU.
3. Loops: a distance-propagating recurrent model (VIN-style). An auxiliary distance target on
   DTNet did not help (see above).
