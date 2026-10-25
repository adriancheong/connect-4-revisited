"""Stage 3: the two network shapes for spotting winning moves.

Both take the 84 switches from stage 2 (`connect4.numpy_nn.winning_moves.encode`) and
give 7 totals, one per column. Squashing a total gives that column's guess: "if the
player to move plays here, do they win right now?".

The last squash is left out of the networks themselves, because the cross-entropy
error (step 2b) works on the totals directly. `winning_moves.guesses()` adds it back.
"""

from __future__ import annotations

import torch
from torch import nn

from connect4.board import COLS, ROWS

SWITCHES = 2 * ROWS * COLS  # 84: "my piece here?" and "opponent's piece here?" for each cell


def make_mlp(hidden: int = 128, squash: str = "sigmoid") -> nn.Sequential:
    """Stage 2's network: 84 switches -> `hidden` neurons -> 7 totals.

    `squash` is the hidden neurons' squash: "sigmoid" (stage 2) or "relu" (step 2b).
    """
    return nn.Sequential(
        nn.Linear(SWITCHES, hidden),
        nn.Sigmoid() if squash == "sigmoid" else nn.ReLU(),
        nn.Linear(hidden, COLS),
    )


class ConnectFourCNN(nn.Module):
    """A convolutional network (step 3): small detectors that slide across the board.

    1. The 84 switches become two 6x7 grids ("planes"): my pieces, my opponent's.
    2. Layer 1: `detectors` small 4x4 detectors, each looking at both planes. Each one
       slides across the whole board, giving a 6x7 map of where its pattern is.
    3. Layer 2: 3x3 detectors that look at all of layer 1's maps at once, so they can
       combine simple patterns into bigger ones.
    4. A final layer reads every map and gives the 7 totals, one per column.

    The edges are padded with zeros (empty cells) so every map stays 6x7.
    """

    def __init__(self, detectors: int = 64) -> None:
        super().__init__()
        # A 4x4 detector needs 3 extra cells across and 3 up to keep the map 6x7: 1 on the
        # left and bottom, 2 on the right and top.
        self.pad = nn.ZeroPad2d((1, 2, 1, 2))
        self.layer1 = nn.Conv2d(2, detectors, kernel_size=4)
        self.layer2 = nn.Conv2d(detectors, detectors, kernel_size=3, padding=1)
        self.out = nn.Linear(detectors * ROWS * COLS, COLS)

    def forward(self, switches: torch.Tensor) -> torch.Tensor:
        planes = switches.reshape(-1, 2, ROWS, COLS)  # row 0 is the bottom row, as in stage 2
        maps = torch.relu(self.layer1(self.pad(planes)))
        maps = torch.relu(self.layer2(maps))
        return self.out(maps.flatten(1))


def dial_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
