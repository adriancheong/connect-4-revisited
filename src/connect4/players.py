"""Players: anything that can look at a board and pick a column.

Humans and bots share one interface, `choose_move(board) -> column`, so the game
loop never needs to know which kind it is talking to. The neural-network bots in
later stages plug in the same way.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Protocol

from connect4.board import COLS, SYMBOLS, Board, other_player


class Player(Protocol):
    name: str

    def choose_move(self, board: Board) -> int:
        """Return a legal column (0-6) for the player to move on `board`."""
        ...


class QuitGame(Exception):
    """Raised when a human chooses to stop the game."""


class HumanPlayer:
    """Asks a person at the terminal for a column."""

    def __init__(
        self,
        name: str = "Human",
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        self.name = name
        self._input = input_fn  # replaceable so tests can script the answers
        self._output = output_fn

    def choose_move(self, board: Board) -> int:
        legal = board.legal_moves()
        symbol = SYMBOLS[board.current_player]
        while True:
            try:
                answer = self._input(f"{symbol} to move, choose a column (1-{COLS}, q to quit): ")
            except EOFError:  # Ctrl-D or end of piped input
                raise QuitGame from None
            answer = answer.strip().lower()
            if answer in ("q", "quit", "exit"):
                raise QuitGame
            if not answer.isdecimal() or not 1 <= int(answer) <= COLS:
                self._output(f"Please enter a number from 1 to {COLS}.")
                continue
            col = int(answer) - 1  # people count columns from 1, the code from 0
            if col not in legal:
                self._output(f"Column {col + 1} is full, pick another.")
                continue
            return col


class RandomPlayer:
    """Picks any legal column at random."""

    def __init__(self, seed: int | None = None, name: str = "Random") -> None:
        self.name = name
        self._rng = random.Random(seed)

    def choose_move(self, board: Board) -> int:
        return self._rng.choice(board.legal_moves())


class HeuristicPlayer:
    """Wins if it can, blocks if it must, otherwise plays randomly.

    It only looks one move ahead, so it is easy to beat with a bit of planning.
    It is the first baseline the neural-network bot has to beat.
    """

    def __init__(self, seed: int | None = None, name: str = "Heuristic") -> None:
        self.name = name
        self._rng = random.Random(seed)

    def choose_move(self, board: Board) -> int:
        wins = board.winning_moves()
        if wins:
            return self._rng.choice(wins)
        threats = board.winning_moves(other_player(board.current_player))
        if threats:
            return self._rng.choice(threats)
        return self._rng.choice(board.legal_moves())
