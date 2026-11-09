"""Fixed-depth feedforward baseline: DTNet without weight tying.

Same projection, residual blocks and head as models.dt_net.DTNetRecall, but every block has
its own weights and the depth is fixed, so the network cannot "think longer" at test time.
This is the feedforward baseline of Schwarzschild et al. (2021): with the default depth of 30
blocks it uses about as much compute per maze as DTNet at 30 training iterations, so the
difference between the two comes from weight tying and the ability to run more iterations.

Works on the samples produced by dataset.MazeDataset:

    x: (B, 3, H, W)  channel 0 = passage (1) / wall (0), channel 1 = start, channel 2 = exit
    y: (B, 1, H, W)  1 = on the solution path

The training and evaluation helpers mirror models.dt_net so the same pipeline drives both.
The `iters` arguments are accepted for that reason and ignored.
"""

import torch
import torch.nn as nn

from models.dt_net import BasicBlock, _labels, masked_ce


class FeedForwardNet(nn.Module):
    def __init__(self, in_channels: int = 3, width: int = 128, depth: int = 30):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1, bias=False), nn.ReLU()
        )
        self.blocks = nn.Sequential(*[BasicBlock(width) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.Conv2d(width, 32, 3, padding=1, bias=False), nn.ReLU(),
            nn.Conv2d(32, 8, 3, padding=1, bias=False), nn.ReLU(),
            nn.Conv2d(8, 2, 3, padding=1, bias=False),
        )

    def forward(self, x, iters=None):
        """Logits of shape (B, 2, H, W). `iters` is ignored (fixed depth)."""
        return self.head(self.blocks(self.projection(x)))


def train_step(model, opt, x, y, max_iters=None, alpha=None, clip: float = 2.0):
    """One optimisation step with the masked cross-entropy (no progressive loss)."""
    model.train()
    opt.zero_grad()
    loss = masked_ce(model(x), y, x)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()

    return loss.item()


def _solved(logits, x, y):
    wrong = (logits.argmax(1) != _labels(y)) & (x[:, 0] > 0.5)
    return ~wrong.flatten(1).any(1)


@torch.no_grad()
def evaluate(model, x, y, iters=None):
    """Masked loss (float) and per-sample "solved" flags (B,)."""
    model.eval()
    logits = model(x)

    return masked_ce(logits, y, x).item(), _solved(logits, x, y)


@torch.no_grad()
def solved_by_iters(model, x, y, test_iters):
    """Same interface as models.dt_net; every entry is identical because the depth is fixed."""
    model.eval()
    ok = _solved(model(x), x, y)

    return {t: ok for t in test_iters}


@torch.no_grad()
def predict(model, x, iters=None):
    """Predicted path mask (B, 1, H, W)."""
    model.eval()

    return model(x).argmax(1, keepdim=True).float() * (x[:, :1] > 0.5)
