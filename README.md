# maze-generalization

Do neural maze solvers generalize to mazes from **unseen generation algorithms**, **larger
sizes** and **mazes with loops**? Models are trained on mazes from a few generators and tested
on the others.

## Installation

Requires [conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda).

```bash
git clone https://github.com/yoshi0428/maze-generalization.git
cd maze-generalization
conda env create -f environment.yml
conda activate maze-generalization
```

`environment.yml` installs PyTorch with pip. On Linux this is a CUDA build; on Windows it is the
CPU-only build. For a GPU on Windows (or a specific CUDA version), reinstall torch from the
PyTorch index. Run `nvidia-smi` and read "CUDA Version" in its header: that is the newest CUDA
your driver supports. Pick a build at or below it, e.g. for CUDA 13.2:

```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu132
```

The wheels bundle their own CUDA runtime, so you only need an NVIDIA driver, not the CUDA
toolkit. Older builds such as `cu126` also run on newer drivers.

Check that the GPU is visible (training falls back to the CPU otherwise, which is much slower):

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Then start Jupyter from the repo root and open a notebook in `notebooks/`:

```bash
jupyter lab
```

The notebooks add `src/` to the import path themselves, so no package install is needed. To
update the environment after `environment.yml` changes, run
`conda env update -f environment.yml --prune`.

## Layout

- `src/generators/`: nine maze generators (DFS, Prim's, Kruskal's, Eller's, Aldous-Broder,
  Wilson's, recursive division, hunt-and-kill, sidewinder) and wall removal \[10\]–\[17\].
- `src/dataset.py`: PyTorch datasets. Each sample is a 3-channel input (maze, start, exit) and
  the shortest path as the target. Includes an even per-generator training split, a
  stratified validation split, and a check that test generators differ from training ones.
- `src/models/dt_net.py`: DTNet, a recurrent ResNet with recall and progressive loss \[2\],
  based on the reference implementation \[8\].
- `src/models/feedforward.py`: fixed-depth feedforward baseline \[1\]: DTNet's blocks without
  weight tying, so it cannot run more iterations on larger mazes.
- `notebooks/dataset_generation_ex.ipynb`: dataset examples and visualizations.
- `notebooks/pipeline.ipynb`: pick a trained run (DTNet or feedforward) by name; loads its
  checkpoint and saved config, then shows its training curves, sample mazes, test results on
  unseen generators, sizes and loops, and predictions. Ends with an error analysis of why it
  fails: splits the wrong pixels of failed test mazes into gaps, dropped tied routes, longer
  routes kept around loops, and unfilled dead ends. Run locally or in Google Colab.
- `notebooks/experiments.ipynb`: runs the study's comparisons (DTNet vs feedforward, mixed vs
  single generator, loops in training, zero-init) and an Optuna hyperparameter search, then
  retrains the best configuration. Finished runs are loaded from `results/`, so it resumes.
- `src/runner.py`: train/evaluate one `Config`; used by the experiments and pipeline notebooks. Each run writes
  logs, loss curves and test results to `results/<name>/` and the checkpoint to
  `checkpoints/<name>.pt`. `results/SUMMARY.md` summarizes the runs so far.

## References

### Models and maze solving

\[1\] A. Schwarzschild, E. Borgnia, A. Gupta, F. Huang, U. Vishkin, M. Goldblum, T. Goldstein.
"Can You Learn an Algorithm? Generalizing from Easy to Hard Problems with Recurrent Networks."
*NeurIPS*, 2021. [arXiv:2106.04537](https://arxiv.org/abs/2106.04537)

\[2\] A. Bansal, A. Schwarzschild, E. Borgnia, Z. Emam, F. Huang, M. Goldblum, T. Goldstein.
"End-to-end Algorithm Synthesis with Recurrent Networks: Logical Extrapolation Without
Overthinking." *NeurIPS*, 2022. [arXiv:2202.05826](https://arxiv.org/abs/2202.05826)

\[3\] R. Yonetani, T. Taniai, M. Barekatain, M. Nishimura, A. Kanezaki. "Path Planning using
Neural A\* Search." *ICML*, PMLR 139, pp. 12029–12039, 2021.
[PMLR](https://proceedings.mlr.press/v139/yonetani21a.html)

\[4\] A. Tamar, Y. Wu, G. Thomas, S. Levine, P. Abbeel. "Value Iteration Networks." *NeurIPS*,
2016. [arXiv:1602.02867](https://arxiv.org/abs/1602.02867)

\[5\] O. Ronneberger, P. Fischer, T. Brox. "U-Net: Convolutional Networks for Biomedical Image
Segmentation." *MICCAI*, LNCS 9351, pp. 234–241, 2015.
[arXiv:1505.04597](https://arxiv.org/abs/1505.04597)

\[6\] B. Knutson, A. C. Rabeendran, M. Ivanitskiy, J. Pettyjohn, C. Diniz-Behn, S. Wu Fung,
D. McKenzie. "On Logical Extrapolation for Mazes with Recurrent and Implicit Networks."
arXiv preprint, 2024. [arXiv:2410.03020](https://arxiv.org/abs/2410.03020). Tests DTNet and
implicit (fixed-point) networks on maze shifts beyond size: some extrapolate in size but fail
on other shifts, more diverse training data does not fix logical extrapolation, and models not
trained to reach a fixed point can end in limit cycles.

### Datasets and code

\[7\] A. Schwarzschild, E. Borgnia, A. Gupta, A. Bansal, Z. Emam, F. Huang, M. Goldblum,
T. Goldstein. "Datasets for Studying Generalization from Easy to Hard Examples." arXiv preprint,
2021. [arXiv:2108.06011](https://arxiv.org/abs/2108.06011)

\[8\] DeepThinking reference implementation (DTNet, progressive loss).
<https://github.com/aks2203/deep-thinking>

\[9\] M. I. Ivanitskiy, R. Shah, A. F. Spies, T. Räuker, D. Valentine, C. Rager, L. Quirke,
C. Mathwin, G. Corlouer, C. Diniz Behn, S. Wu Fung. "A Configurable Library for Generating and
Manipulating Maze Datasets." arXiv preprint, 2023.
[arXiv:2309.10498](https://arxiv.org/abs/2309.10498)

### Maze generation algorithms

\[10\] J. Buck. *Mazes for Programmers: Code Your Own Twisty Little Passages.* Pragmatic Bookshelf,
2015. Covers Eller's, sidewinder, hunt-and-kill, recursive division and the other generators.

\[11\] D. B. Wilson. "Generating Random Spanning Trees More Quickly than the Cover Time."
*STOC*, 1996.

\[12\] D. J. Aldous. "The Random Walk Construction of Uniform Spanning Trees and Uniform Labelled
Trees." *SIAM Journal on Discrete Mathematics* 3(4), 1990.

\[13\] A. Z. Broder. "Generating Random Spanning Trees." *FOCS*, 1989.

\[14\] J. B. Kruskal. "On the Shortest Spanning Subtree of a Graph and the Traveling Salesman
Problem." *Proceedings of the American Mathematical Society* 7(1), 1956.

\[15\] R. C. Prim. "Shortest Connection Networks and Some Generalizations." *Bell System
Technical Journal* 36(6), 1957.

\[16\] professor-l. "Maze Generation Algorithms - An Exploration." Interactive explanations of
backtracking (DFS), Prim's, Eller's, Kruskal's, Aldous-Broder, Wilson's, recursive division,
hunt-and-kill, sidewinder and binary tree. <https://professor-l.github.io/mazes/>

\[17\] professor-l. *mazes*: source code for \[16\] (MIT license).
<https://github.com/professor-l/mazes>

### Neural algorithmic reasoning

\[18\] P. Veličković, R. Ying, M. Padovano, R. Hadsell, C. Blundell. "Neural Execution of Graph
Algorithms." *ICLR*, 2020. [arXiv:1910.10593](https://arxiv.org/abs/1910.10593). Graph networks
trained to imitate the steps of BFS and Bellman-Ford; max aggregation matches the algorithm's
structure and generalizes to larger graphs than the training ones.

\[19\] J. Bear, A. Prügel-Bennett, J. Hare. "Rethinking Deep Thinking: Stable Learning of
Algorithms using Lipschitz Constraints." *NeurIPS*, 2024.
[arXiv:2410.23451](https://arxiv.org/abs/2410.23451). DT-L: constrains DTNet's recurrent step to
a contraction (spectral normalisation, ELU, weighted residuals), which makes training stable and
guarantees convergence to a unique solution at test time, with far fewer parameters.

\[20\] C. Anil, A. Pokle, K. Liang, J. Treutlein, Y. Wu, S. Bai, J. Z. Kolter, R. Grosse. "Path
Independent Equilibrium Models Can Better Exploit Test-Time Computation." *NeurIPS*, 2022.
[arXiv:2211.09961](https://arxiv.org/abs/2211.09961). Models that reach the same final state
whatever their initial state (path independent) gain from extra test-time iterations, while
path-dependent ones degrade; starting part of each batch from noise and randomising the depth
during training promote it.
