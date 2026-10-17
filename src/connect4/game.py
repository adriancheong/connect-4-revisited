"""The game loop: lets two players take turns until the game ends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from connect4.board import PLAYER_1, Board
from connect4.players import Player


@dataclass(frozen=True)
class GameResult:
    winner: int | None  # 1 or 2, or None for a draw
    moves: tuple[int, ...]  # columns played, in order
    # Every position from the empty board to the final one (one more than the number of moves).
    # The self-play training in stage 4 learns from this history.
    boards: tuple[Board, ...]

    @property
    def is_draw(self) -> bool:
        return self.winner is None


def play_game(
    player1: Player,
    player2: Player,
    on_move: Callable[[Board], None] | None = None,
) -> GameResult:
    """Play one game. Player 1 moves first. `on_move` is called with the board after each move."""
    board = Board()
    boards = [board]
    while not board.is_terminal():
        player = player1 if board.current_player == PLAYER_1 else player2
        board = board.play(player.choose_move(board))
        boards.append(board)
        if on_move is not None:
            on_move(board)
    return GameResult(winner=board.winner, moves=board.moves, boards=tuple(boards))
