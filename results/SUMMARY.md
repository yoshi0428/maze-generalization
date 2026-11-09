# Experiment results, 2026-11-08

All runs: `notebooks/experiments.ipynb` on an RTX 5060 Ti 16 GB with bf16 autocast, seed 0. Each
run folder holds `log.txt`, `config.json`, `history.json`, `results.json`, `curves.png`
(train/val loss, val solved) and `test.png`; `error_analysis.json` / `.png` come from
`notebooks/error_analysis.ipynb`.

Common settings: 30k training mazes of size 15 (10% for validation), 20 epochs, batch 64, lr 1e-3
(2 warmup epochs, cosine decay), clip 2, 30 training iterations, progressive loss alpha 0.01.
Training generators: dfs, prim, kruskal (unless stated). Test generators (unseen): eller, wilson,
hunt-and-kill, recursive division, sidewinder.

"Solved" = every passage pixel correct. The target marks every pixel on *some* shortest path
from start to exit (in a perfect maze this is the unique path; with loops, all tied routes).
Numbers are the best over test iterations (30 / 100 / 300 / 1000 / 3000); per-iteration numbers
are in the logs. Test mazes per size: 250 (15, 31), 200 (63), 100 (127), 40 (255), 16 (511), so
the 255 and 511 numbers are rough (±8 and ±12 points at one standard error). Each configuration
was run once.

## Perfect test mazes (no loops)

| Run | Model | Loops in training | Val | 31 | 63 | 127 | 255 | 511 |
|---|---|---|---|---|---|---|---|---|
| `dtnet_s15_noloops` | DTNet | no | 100% | 100% | 100% | 99% | 92.5% | 62.5% |
| `dtnet_s15_noloops_zeroinit` | DTNet, zero-init | no | 100% | 100% | **100%** | **100%** | **100%** | **100%** |
| `dtnet_s15_dfs` | DTNet, DFS only | no | 100% | 90.8% | 55.5% | 12% | 0% | 0% |
| `dtnet_s15_loops` | DTNet | 0–10% | 93.9%¹ | 72.8% | 24% | 7% | 2.5% | 0% |
| `ff_s15_noloops` | Feedforward, 30 blocks | no | 100% | 28.8% | 0% | 0% | 0% | 0% |

All runs solve 99.6–100% of size-15 test mazes. ¹ validation includes mazes with loops.

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
| `dtnet_s15_loops` | 87.6% | 60% | 19.5% | 4% |

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
   ends and keeps every loop, so it fails every loop test. With loops in training it stops
   extrapolating: accuracy is flat from ~300 iterations on, even on perfect mazes (72.8% at 31,
   24% at 63), and most failures are paths with gaps (error analysis below).

5. **Zero-init residual blocks look better.** Starting the last conv of each block at zero gave
   the only run that solves all sizes up to 511 (16/16 vs 10/16 at 511, 40/40 vs 37/40 at 255),
   and a much smaller largest training-loss spike (118 vs 8e7). One run each, so this needs
   2–3 seeds before it is a result.

6. **Training is unstable.** Loss spikes (up to 8e7 in `dtnet_s15_noloops`, 3e6 in
   `dtnet_s15_loops`) and validation collapses below 50% (noloops epoch 5, zero-init epoch 4,
   loops epoch 7) happen in most DTNet runs and recover. Saving the best checkpoint by
   validation handles it. The feedforward and DFS-only runs had no collapses.

7. **Shortest paths in mazes with loops are often tied.** At 10% wall removal, 26% of size-15 and
   68% of size-63 test mazes have several equally short routes (up to 960); at 30% removal, 90%
   of size-63 mazes do (`dataset_generation_ex.ipynb`). This is why the target marks every
   shortest path; a single BFS path would be an arbitrary choice.

## Error analysis (`notebooks/error_analysis.ipynb`)

Wrong pixels of failed test mazes at 300 iterations, split into: gaps in pixels every shortest
path uses (missed forced), dropped tied routes (missed tied), longer routes kept around loops
(extra route) and unfilled dead ends (extra dead end).

| Run | Test mazes | Size | Failed | Connected¹ | One shortest path² | Missed forced | Missed tied | Extra route | Extra dead end |
|---|---|---|---|---|---|---|---|---|---|
| `dtnet_s15_noloops` | 10% removed | 31 | 200/200 | 198 | 0 | 0% | 0% | 100% | 0% |
| `dtnet_s15_noloops` | 10% removed | 63 | 100/100 | 98 | 0 | 0% | 0% | 100% | 0% |
| `dtnet_s15_loops` | perfect | 31 | 52/200 | 6 | – | 86% | – | – | 14% |
| `dtnet_s15_loops` | perfect | 63 | 77/100 | 0 | – | 90% | – | – | 10% |
| `dtnet_s15_loops` | 10% removed | 31 | 75/200 | 26 | 8 | 39% | 29% | 32% | 0% |
| `dtnet_s15_loops` | 10% removed | 63 | 82/100 | 12 | 4 | 43% | 34% | 23% | 0% |

Category columns are the share of wrong pixels. ¹ failed predictions that still connect start and
exit. ² failed predictions that are exactly one correct shortest path (only these would pass a
looser "any shortest path" metric).

- **Without loops in training**, DTNet marks everything that survives dead-end filling: correct
  for perfect mazes, wrong for every maze with a loop.
- **With loops in training**, it lost dead-end filling and learned something local: on perfect
  mazes 86–90% of the wrong pixels are gaps and no failed size-63 prediction reaches the exit.
- **The wide target (all tied routes) is not the main problem:** only 4–8 failed loop mazes per
  size are a single valid shortest path.

## Next steps

1. Repeat the key comparisons (mixed vs DFS only, zero-init vs default) with 2–3 seeds.
2. Classical baselines for proposal step 4: BFS and A* on CPU, BFS as repeated 3x3 dilation on GPU.
3. Loops: a distance-propagating recurrent model (VIN-style), or DTNet with an auxiliary
   distance target.
