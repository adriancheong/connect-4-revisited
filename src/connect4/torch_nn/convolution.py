"""Stage 3, step 3a: how a convolution works, by hand.

Read this alongside docs/stage-3/03a-convolution-by-hand.md.

Run `python -m connect4.torch_nn.convolution` to print the worked example: one small
detector slid across a board by hand, then the same done by PyTorch's convolution.
"""

from __future__ import annotations

import numpy as np
import torch

from connect4.board import COLS, ROWS, Board
from connect4.cli import render_board
from connect4.numpy_nn.winning_moves import encode

EXAMPLE_MOVES = [2, 3, 2, 3, 2, 6]  # red has three stacked in column 3; red to move

# A detector for "three of my pieces stacked in a column". It is 3 cells tall and 1 wide,
# and looks at both planes: weight 1 on each of my pieces, 0 on my opponent's.
STACK_WEIGHTS = np.zeros((2, 3, 1))  # (plane, height, width)
STACK_WEIGHTS[0, :, 0] = 1.0
STACK_BIAS = -2.0  # so the total only gets above 0 when all three are mine


def planes(board: Board) -> np.ndarray:
    """The 84 switches as two 6x7 grids: [0] my pieces, [1] my opponent's. Row 0 is the bottom."""
    return encode(board).reshape(2, ROWS, COLS)


def slide(board_planes: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """Slide one detector across the board, by hand.

    At each position: multiply every switch under the detector by its weight, add them
    all up, add the bias, then ReLU (anything below 0 becomes 0). The result is a map
    with one number per position: how strongly the pattern is there.
    """
    _, height, width = weights.shape
    out = np.zeros((ROWS - height + 1, COLS - width + 1))
    for row in range(out.shape[0]):
        for col in range(out.shape[1]):
            window = board_planes[:, row : row + height, col : col + width]
            out[row, col] = max(0.0, float((window * weights).sum() + bias))
    return out


def slide_with_torch(board_planes: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """The same thing, done by PyTorch's convolution."""
    x = torch.as_tensor(board_planes[None], dtype=torch.float64)  # a batch of one board
    w = torch.as_tensor(weights[None], dtype=torch.float64)  # one detector
    b = torch.tensor([bias], dtype=torch.float64)
    return torch.relu(torch.nn.functional.conv2d(x, w, b))[0, 0].numpy()


def main() -> None:
    board = Board.from_moves(EXAMPLE_MOVES)
    p = planes(board)
    print(render_board(board))
    print("Red to move.\n")
    print("My pieces (red), as switches, top row first:")
    print(p[0][::-1].astype(int))
    print()
    by_hand = slide(p, STACK_WEIGHTS, STACK_BIAS)
    print("The 'three of mine stacked' detector at every position, top row first.")
    print("Each number is one position; its row is the detector's bottom cell:")
    print(by_hand[::-1])
    print()
    for row, col in ((0, 2), (1, 2), (0, 3)):
        mine = p[0, row : row + 3, col].astype(int).tolist()
        total = sum(mine) + STACK_BIAS
        added = " + ".join(map(str, mine))
        print(f"  column {col + 1}, rows {row + 1}-{row + 3}: my pieces {mine} -> "
              f"{added} - 2 = {total:+.0f} -> {max(total, 0):.0f}")  # fmt: skip
    same = np.array_equal(by_hand, slide_with_torch(p, STACK_WEIGHTS, STACK_BIAS))
    print(f"\nPyTorch's conv2d gives the same map: {same}")


if __name__ == "__main__":
    main()
