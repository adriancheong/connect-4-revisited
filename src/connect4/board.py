"""Connect 4 rules: the board, legal moves and win detection.

This module only knows the rules. It does no input or output, so humans,
simple bots and (later) neural-network bots can all share it.

Conventions:
- The grid has 6 rows and 7 columns. Row 0 is the bottom row.
- Columns are numbered 0-6 in code. The terminal shows them as 1-7.
- Cells hold 0 (empty), 1 (player 1) or 2 (player 2). Player 1 always moves first.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

ROWS = 6
COLS = 7
EMPTY = 0
PLAYER_1 = 1
PLAYER_2 = 2
SYMBOLS = {EMPTY: ".", PLAYER_1: "X", PLAYER_2: "O"}  # how each cell is shown

# The four line directions as (row step, column step): horizontal, vertical, and both diagonals.
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def other_player(player: int) -> int:
    """Return the opponent of `player`."""
    return PLAYER_2 if player == PLAYER_1 else PLAYER_1


class Board:
    """An immutable Connect 4 position.

    `play()` returns a new Board and leaves this one unchanged, so a board can be
    stored in a game history or explored by a bot without being modified by accident.
    """

    __slots__ = ("_grid", "_heights", "_moves", "_winner")

    def __init__(self) -> None:
        self._grid = np.zeros((ROWS, COLS), dtype=np.int8)
        self._heights = [0] * COLS  # number of pieces in each column
        self._moves: tuple[int, ...] = ()
        self._winner: int | None = None

    @classmethod
    def from_moves(cls, moves: Iterable[int]) -> Board:
        """Build a board by playing a sequence of columns (0-6) from an empty board."""
        board = cls()
        for col in moves:
            board = board.play(col)
        return board

    # --- Reading the position -------------------------------------------------

    @property
    def grid(self) -> np.ndarray:
        """Read-only 6x7 array of cell values. Row 0 is the bottom."""
        view = self._grid.view()
        view.flags.writeable = False
        return view

    @property
    def moves(self) -> tuple[int, ...]:
        """Columns played so far, in order."""
        return self._moves

    @property
    def current_player(self) -> int:
        """The player whose turn it is (1 or 2)."""
        return PLAYER_1 if len(self._moves) % 2 == 0 else PLAYER_2

    @property
    def winner(self) -> int | None:
        """1 or 2 if that player has four in a row, otherwise None."""
        return self._winner

    def is_full(self) -> bool:
        return len(self._moves) == ROWS * COLS

    def is_terminal(self) -> bool:
        """True when the game is over: someone has won or the board is full (a draw)."""
        return self._winner is not None or self.is_full()

    def legal_moves(self) -> list[int]:
        """Columns that can still be played. Empty once the game is over."""
        if self._winner is not None:
            return []
        return [col for col in range(COLS) if self._heights[col] < ROWS]

    def winning_moves(self, player: int | None = None) -> list[int]:
        """Columns where `player` (default: the player to move) would complete four now."""
        if player is None:
            player = self.current_player
        wins = []
        for col in self.legal_moves():
            row = self._heights[col]
            self._grid[row, col] = player  # try the move, then undo it
            if _makes_four(self._grid, row, col):
                wins.append(col)
            self._grid[row, col] = EMPTY
        return wins

    # --- Making a move --------------------------------------------------------

    def play(self, col: int) -> Board:
        """Drop the current player's piece into `col` and return the new board."""
        if self.is_terminal():
            raise ValueError("the game is already over")
        if not 0 <= col < COLS:
            raise ValueError(f"column must be between 0 and {COLS - 1}, got {col}")
        row = self._heights[col]
        if row == ROWS:
            raise ValueError(f"column {col} is full")

        player = self.current_player
        new = Board.__new__(Board)
        new._grid = self._grid.copy()
        new._grid[row, col] = player
        new._heights = self._heights.copy()
        new._heights[col] += 1
        new._moves = self._moves + (col,)
        # A new line of four must pass through the piece just placed,
        # so only those lines need checking.
        new._winner = player if _makes_four(new._grid, row, col) else None
        return new

    def __repr__(self) -> str:
        return f"Board(moves={list(self._moves)})"


def _makes_four(grid: np.ndarray, row: int, col: int) -> bool:
    """True if the piece at (row, col) is part of a line of four or more."""
    player = grid[row, col]
    for d_row, d_col in _DIRECTIONS:
        count = 1  # the piece itself
        for sign in (1, -1):  # walk outwards in both directions along the line
            r, c = row + sign * d_row, col + sign * d_col
            while 0 <= r < ROWS and 0 <= c < COLS and grid[r, c] == player:
                count += 1
                r += sign * d_row
                c += sign * d_col
        if count >= 4:
            return True
    return False
