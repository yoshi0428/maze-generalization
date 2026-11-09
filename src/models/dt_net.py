"""DeepThinking recurrent ResNet with recall and progressive loss (Bansal et al., 2022).

Condensed rewrite of DTNet from https://github.com/aks2203/deep-thinking
(deepthinking/models/dt_net_2d.py and deepthinking/utils/training.py), adapted to
the samples produced by dataset.MazeDataset:

    x: (B, 3, H, W)  channel 0 = passage (1) / wall (0), channel 1 = start, channel 2 = exit
    y: (B, 1, H, W)  1 = on the solution path

The recurrent block is weight-tied, so the same weights run for more iterations
and on larger mazes at test time than in training.
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Residual block without batchnorm (BN statistics break when iterating past training depth)."""

    def __init__(self, width: int):
        super().__init__()
        self.conv1 = nn.Conv2d(width, width, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(width, width, 3, padding=1, bias=False)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)

        return F.relu(out + x)


class DTNetRecall(nn.Module):
    def __init__(self, in_channels: int = 3, width: int = 128, num_blocks: int = 2, zero_init: bool = False):
        """zero_init: start the last conv of every residual block at zero, so each block begins as
        an identity map (as in Fixup / SkipInit); meant to stabilise training without normalisation."""
        super().__init__()
        self.width = width

        # Project the input into feature space (run once).
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1, bias=False), nn.ReLU()
        )

        # Recurrent block; "recall" concatenates the raw input to the features every iteration.
        self.recall_conv = nn.Conv2d(width + in_channels, width, 3, padding=1, bias=False)
        self.blocks = nn.Sequential(*[BasicBlock(width) for _ in range(num_blocks)])
        if zero_init:
            for block in self.blocks:
                nn.init.zeros_(block.conv2.weight)

        # Head: features -> per-pixel logits for (not path, path), shared across iterations.
        self.head = nn.Sequential(
            nn.Conv2d(width, 32, 3, padding=1, bias=False), nn.ReLU(),
            nn.Conv2d(32, 8, 3, padding=1, bias=False), nn.ReLU(),
            nn.Conv2d(8, 2, 3, padding=1, bias=False),
        )

    def forward(self, x, iters: int, interim_thought=None):
        """Run `iters` recurrent steps; pass `interim_thought` to resume from an earlier state.

        Returns (logits, state), logits of shape (B, 2, H, W).
        """
        h = self.projection(x) if interim_thought is None else interim_thought

        for _ in range(iters):
            h = self.blocks(F.relu(self.recall_conv(torch.cat([h, x], dim=1))))

        return self.head(h), h


def _labels(y):
    return y.squeeze(1).long() if y.dim() == 4 else y.long()


def masked_ce(logits, y, x):
    """Cross-entropy on passage pixels only (walls are trivially 'not path')."""
    valid = x[:, 0] > 0.5
    loss = F.cross_entropy(logits, _labels(y), reduction="none")

    return (loss * valid).sum() / valid.sum().clamp(min=1)


def train_step(model, opt, x, y, max_iters: int = 30, alpha: float = 0.01, clip: float = 2.0):
    """One optimisation step with the progressive loss.

    loss = (1 - alpha) * loss after max_iters
         +      alpha  * loss after k grad-tracked steps started from a *detached*
                         state reached after n random steps.

    The detach stops the network from learning "which iteration am I on", so the
    recurrent step becomes an incremental improvement that can be repeated at test time.
    """
    model.train()
    opt.zero_grad()

    logits_max, _ = model(x, max_iters)
    loss_max = masked_ce(logits_max, y, x)

    loss_prog = torch.zeros((), device=x.device)
    if alpha > 0:
        n = random.randint(0, max_iters - 1)
        k = random.randint(1, max_iters - n)

        with torch.no_grad():
            h = model(x, n)[1] if n > 0 else model.projection(x)

        logits_prog, _ = model(x, k, interim_thought=h.detach())
        loss_prog = masked_ce(logits_prog, y, x)

    loss = (1 - alpha) * loss_max + alpha * loss_prog
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()

    return loss.item()


@torch.no_grad()
def solved_by_iters(model, x, y, test_iters):
    """Per-sample "solved" flags after each iteration count in `test_iters`.

    A maze counts as solved only if every passage pixel is labelled correctly.
    Iterations are run incrementally, so memory does not grow with the count.
    Returns {iters: bool tensor of shape (B,)}.
    """
    model.eval()
    labels = _labels(y)
    valid = x[:, 0] > 0.5

    h, done, results = model.projection(x), 0, {}
    for t in sorted(test_iters):
        logits, h = model(x, t - done, interim_thought=h)
        done = t
        wrong = (logits.argmax(1) != labels) & valid
        results[t] = ~wrong.flatten(1).any(1)

    return results


@torch.no_grad()
def evaluate(model, x, y, iters: int):
    """Masked loss (float) and per-sample "solved" flags (B,) after `iters` iterations."""
    model.eval()
    logits, _ = model(x, iters)
    wrong = (logits.argmax(1) != _labels(y)) & (x[:, 0] > 0.5)

    return masked_ce(logits, y, x).item(), ~wrong.flatten(1).any(1)


@torch.no_grad()
def predict(model, x, iters: int):
    """Predicted path mask (B, 1, H, W) after `iters` iterations."""
    model.eval()
    logits, _ = model(x, iters)

    return logits.argmax(1, keepdim=True).float() * (x[:, :1] > 0.5)
