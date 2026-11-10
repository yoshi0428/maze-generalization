"""Train and test one model configuration. Used by notebooks/experiments.ipynb.

    cfg = Config(name="dtnet_s15", train_size=15, test_sizes={63: (200, 50)})
    out = run(cfg, out_dir=Path("results/dtnet_s15"), checkpoint=Path("checkpoints/dtnet_s15.pt"))
    out["results"]["0.0/63"][1000]   # fraction of size-63 test mazes solved at 1000 iterations

Results are keyed "<wall_removal>/<size>" for test sets (unseen generators) and
"extrap/<size>" for held-out larger mazes from the *training* generators, which is a
tuning signal that does not touch the test generators.
"""

import json
import os
import random
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset import MazeDataset, make_test_datasets, make_train_dataset, stratified_split
from models import dt_net, feedforward

TEST_GENERATORS = ("eller", "wilson", "hunt_and_kill", "recursive_division", "sidewinder")

# model name -> (module with train_step / evaluate / solved_by_iters, constructor)
MODELS = {
    "dtnet": (dt_net, lambda cfg: dt_net.DTNetRecall(width=cfg.width, zero_init=cfg.zero_init)),
    "ff": (feedforward, lambda cfg: feedforward.FeedForwardNet(width=cfg.width, depth=cfg.depth)),
}


@dataclass
class Config:
    name: str
    model: str = "dtnet"                    # "dtnet" or "ff" (fixed-depth feedforward baseline)
    width: int = 128
    depth: int = 30                         # ff only: number of residual blocks
    zero_init: bool = False                 # dtnet only: zero-init the last conv of each block
    train_generators: tuple = ("dfs", "prim", "kruskal")
    train_size: int = 15
    n_train: int = 30_000
    train_wr: float | tuple = 0.0           # wall removal in training: fraction or (min, max)
    val_fraction: float = 0.1
    epochs: int = 20
    batch: int = 64
    lr: float = 1e-3
    warmup: float = 2.0                     # epochs of linear warmup before cosine decay
    alpha: float = 0.01                     # progressive-loss weight (dtnet only)
    max_iters: int = 30                     # training iterations (dtnet only)
    clip: float = 2.0
    test_generators: tuple = TEST_GENERATORS
    test_sizes: dict = field(default_factory=dict)    # size -> (n mazes, eval batch)
    test_wr: tuple = (0.0, 0.1)
    loop_test_max_size: int = 127           # tests with walls removed only up to this size (slow, ~0%)
    test_iters: tuple = (30, 100, 300, 1000, 3000)
    extrap_sizes: dict = field(default_factory=dict)  # size -> (n mazes, eval batch)
    amp: bool = True                        # bf16 autocast on CUDA
    init_from: str | None = None            # checkpoint to start from (curriculum)
    seed: int = 0


# ---------------------------------------------------------------- data

def materialize(ds, batch_size: int = 64, progress: str | None = None) -> TensorDataset:
    """Generate every sample once; store as uint8 to keep RAM small."""
    loader = DataLoader(ds, batch_size=batch_size, num_workers=min(8, os.cpu_count() or 1))
    batches = iter(loader)  # fork the workers before a tqdm widget exists (Jupyter breaks otherwise)
    if progress:
        from tqdm.auto import tqdm
        batches = tqdm(batches, total=len(loader), desc=progress, leave=False)
    xs, ys = zip(*((x.to(torch.uint8), y.to(torch.uint8)) for x, y in batches))
    return TensorDataset(torch.cat(xs), torch.cat(ys))


@lru_cache(maxsize=1)  # repeated runs on the same data (e.g. Optuna trials) reuse it
def _training_data(generators, size, n, wall_removal, val_fraction, seed, progress):
    full = make_train_dataset(list(generators), size=size, num_samples=n, seed=seed,
                              wall_removal=wall_removal)
    splits = stratified_split(full, val_fraction, seed=seed)
    data = materialize(full, progress="train mazes" if progress else None)
    return tuple(TensorDataset(*(t[s.indices] for t in data.tensors)) for s in splits)


def training_data(cfg: Config, progress: bool = False):
    """(train, val) TensorDatasets."""
    wr = tuple(cfg.train_wr) if isinstance(cfg.train_wr, (list, tuple)) else cfg.train_wr
    return _training_data(tuple(cfg.train_generators), cfg.train_size, cfg.n_train, wr,
                          cfg.val_fraction, cfg.seed, progress)


def eval_sets(cfg: Config) -> dict:
    """Every evaluation set of cfg: {result key: (make dataset, n mazes, eval batch)}."""
    sets = {f"extrap/{size}": (lambda size=size, n=n: MazeDataset(
                list(cfg.train_generators), size, n, seed=10_000 + cfg.seed), n, batch)
            for size, (n, batch) in cfg.extrap_sizes.items()}
    for wr in cfg.test_wr:
        for size, (n, batch) in cfg.test_sizes.items():
            if wr == 0 or size <= cfg.loop_test_max_size:
                sets[f"{wr}/{size}"] = (lambda size=size, n=n, wr=wr: make_test_datasets(
                    list(cfg.test_generators), size, n, train_generators=list(cfg.train_generators),
                    seed=cfg.seed + 1, wall_removal=wr)[size], n, batch)
    return sets


# ---------------------------------------------------------------- run

class _Recorder:
    """Logs messages and, with an out_dir, writes them and JSON files there."""

    def __init__(self, out_dir, log):
        self.out_dir, self.log = (Path(out_dir) if out_dir else None), log
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)

    def say(self, msg):
        self.log(msg)
        if self.out_dir:
            with open(self.out_dir / "log.txt", "a") as f:
                f.write(msg + "\n")

    def save(self, name, obj):
        if self.out_dir:
            (self.out_dir / name).write_text(json.dumps(obj, indent=2))


def train(model, cfg: Config, device, rec: _Recorder, checkpoint=None, on_epoch=None, progress=False):
    """Train in place; leaves the best epoch's weights (by val solved, then val loss) in model."""
    M = MODELS[cfg.model][0]
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp and device.type == "cuda")
    train_set, val_set = training_data(cfg, progress=progress)
    rec.say(f"data: {len(train_set)} train / {len(val_set)} val, size {cfg.train_size}")
    train_loader = DataLoader(train_set, batch_size=cfg.batch, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch, pin_memory=True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    total, warm = cfg.epochs * len(train_loader), int(cfg.warmup * len(train_loader))
    sched = torch.optim.lr_scheduler.LambdaLR(  # linear warmup, then cosine decay to zero
        opt, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm))))

    history = {"train_loss": [], "val_loss": [], "val_solved": [], "epoch_s": []}
    best, best_state = (-1.0, -float("inf")), None  # (val solved, -val loss): higher is better
    epochs = range(cfg.epochs)
    if progress:
        from tqdm.auto import tqdm
        epochs = tqdm(epochs, desc=cfg.name, leave=False)

    for epoch in epochs:
        t0, losses = time.time(), []
        for x, y in train_loader:
            with autocast:
                losses.append(M.train_step(model, opt, x.to(device).float(), y.to(device).float(),
                                           max_iters=cfg.max_iters, alpha=cfg.alpha, clip=cfg.clip))
            sched.step()

        val_loss, solved = 0.0, []
        with autocast:
            for x, y in val_loader:
                loss, ok = M.evaluate(model, x.to(device).float(), y.to(device).float(), cfg.max_iters)
                val_loss += loss * len(x) / len(val_set)
                solved.append(ok.cpu())
        val_solved = torch.cat(solved).float().mean().item()
        train_loss = float(np.mean(losses)) if np.all(np.isfinite(losses)) else float("nan")
        for k, v in zip(history, (train_loss, val_loss, val_solved, time.time() - t0)):
            history[k].append(v)

        if (val_solved, -val_loss) > best:
            best = (val_solved, -val_loss)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if checkpoint:
                Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint)
        rec.say(f"epoch {epoch + 1:3d}/{cfg.epochs}  train {train_loss:.4f}  val loss {val_loss:.4f}  "
                f"val solved {val_solved:.1%}  lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s")
        rec.save("history.json", history)
        if on_epoch:
            on_epoch(epoch, history)

    rec.say(f"best val solved {best[0]:.1%} (val loss {-best[1]:.4f})")
    model.load_state_dict(best_state)
    return history, (best[0], -best[1])


def run(
    cfg: Config,
    out_dir: Path | None = None,
    checkpoint: Path | None = None,
    skip_train: bool = False,
    on_epoch: Callable[[int, dict], None] | None = None,
    progress: bool = False,
    log: Callable[[str], None] = print,
    results: dict | None = None,
) -> dict:
    """Train (unless skip_train, which loads `checkpoint`), then evaluate.

    Returns {"history", "best", "results"}. on_epoch(epoch, history) is called after every
    validation, e.g. to report to Optuna (it may raise optuna.TrialPruned). With out_dir,
    writes log.txt, config.json, history.json, results.json, curves.png and test.png there.
    Evaluation sets already in `results` (earlier results of this run) are skipped.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    rec = _Recorder(out_dir, log)
    rec.save("config.json", asdict(cfg))
    M, make_model = MODELS[cfg.model]
    model = make_model(cfg)

    if cfg.init_from:
        model.load_state_dict(torch.load(cfg.init_from, map_location="cpu"))

    model.to(device)
    rec.say(f"== {cfg.name}: {cfg.model}, {sum(p.numel() for p in model.parameters()):,} params")

    history, best = {}, None
    if not skip_train:
        history, best = train(model, cfg, device, rec, checkpoint, on_epoch, progress)
        if rec.out_dir:
            plot_history(history, cfg, rec.out_dir / "curves.png")
    elif checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=device))

    results = normalize_results(results or {})
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.amp and device.type == "cuda")
    for key, (make_dataset, n, batch) in eval_sets(cfg).items():
        if key in results:
            continue
        t0, solved = time.time(), {t: [] for t in cfg.test_iters}
        with autocast:
            for x, y in DataLoader(materialize(make_dataset(), max(1, min(64, n // 8))), batch_size=batch):
                for t, ok in M.solved_by_iters(model, x.to(device).float(), y.to(device).float(),
                                               list(cfg.test_iters)).items():
                    solved[t].append(ok.cpu())
        r = results[key] = {t: torch.cat(v).float().mean().item() for t, v in solved.items()}
        rec.say(f"{key} n={n}: " + "  ".join(f"{t}:{v:.1%}" for t, v in r.items())
                + f"   ({time.time() - t0:.0f}s)")
        rec.save("results.json", results)

    if rec.out_dir and cfg.test_sizes:
        plot_results(results, cfg, rec.out_dir / "test.png")

    rec.say("done")
    return {"history": history, "best": best, "results": results}


def normalize_results(results: dict) -> dict:
    """Iteration keys as ints (JSON turns them into strings)."""
    return {k: {int(t): v for t, v in r.items()} for k, r in results.items()}


def run_or_load(cfg: Config, results_dir: Path, checkpoint_dir: Path, **kwargs) -> dict:
    """Load a finished run from results_dir/<name>, finish a partial one, or run from scratch.

    Finished = results.json covers every evaluation set of cfg. Partial = a checkpoint exists:
    only the missing sets are evaluated, without retraining.
    """
    out, ckpt = Path(results_dir) / cfg.name, Path(checkpoint_dir) / f"{cfg.name}.pt"
    load = lambda name: json.loads((out / name).read_text()) if (out / name).exists() else {}
    history, results = load("history.json"), normalize_results(load("results.json"))
    if history and eval_sets(cfg).keys() <= results.keys():
        return {"history": history, "best": None, "results": results, "cached": True}
    if history and ckpt.exists():
        return {**run(cfg, out_dir=out, checkpoint=ckpt, skip_train=True, results=results, **kwargs),
                "history": history}
    return run(cfg, out_dir=out, checkpoint=ckpt, **kwargs)


# ---------------------------------------------------------------- plots

def plot_history(history: dict, cfg: Config, path: Path | None = None):
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    ep = range(1, len(history["train_loss"]) + 1)
    a1.plot(ep, history["train_loss"], label="train")
    a1.plot(ep, history["val_loss"], label="val")
    a1.set(xlabel="epoch", ylabel="masked loss", yscale="log", title=cfg.name)
    a1.legend()
    a2.plot(ep, history["val_solved"])
    a2.set(xlabel="epoch", ylabel="val solved", ylim=(0, 1.02))
    return _finish(fig, path)


def plot_results(results: dict, cfg: Config, path: Path | None = None):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(cfg.test_wr), figsize=(5.5 * len(cfg.test_wr), 3.8), squeeze=False)
    for ax, wr in zip(axes[0], cfg.test_wr):
        for size in cfg.test_sizes:
            if (r := results.get(f"{wr}/{size}")) is not None:
                ax.plot(list(r), list(r.values()), marker="o", label=f"size {size}")
        ax.set(xscale="log", xlabel="test iterations", ylabel="solved", ylim=(0, 1.02),
               title=f"{cfg.name}: wall removal {wr:.0%}")
        ax.legend()
    return _finish(fig, path)


def _finish(fig, path):
    import matplotlib.pyplot as plt

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig
