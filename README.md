# maze-generalization

Do neural maze solvers generalize to mazes from **unseen generation algorithms**, **larger
sizes** and **mazes with loops**? Models are trained on mazes from a few generators and tested
on the others.

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
- `notebooks/dtnet_pipeline.ipynb`, `notebooks/feedforward_pipeline.ipynb`: the full pipeline for
  each model (data, training, validation, testing on unseen generators and sizes), with identical
  data and evaluation settings. Run locally or in Google Colab.
- `notebooks/experiments.ipynb`: runs the study's comparisons (DTNet vs feedforward, mixed vs
  single generator, loops in training, zero-init) and an Optuna hyperparameter search, then
  retrains the best configuration. Finished runs are loaded from `results/`, so it resumes.
- `notebooks/error_analysis.ipynb`: why trained models fail. Splits the wrong pixels of failed
  test mazes into gaps, dropped tied routes, longer routes kept around loops, and unfilled dead ends.
- `src/runner.py`: train/evaluate one `Config`; shared by the experiments notebook and
  `scripts/experiment.py`, the command-line version for unattended runs. Each run writes logs,
  loss curves and test results to `results/<name>/` and the checkpoint to
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
arXiv preprint, 2024. [arXiv:2410.03020](https://arxiv.org/abs/2410.03020)

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
