"""Terminal front end: draws the board and runs games from the command line.

Examples:
    python -m connect4                                   # two humans
    python -m connect4 --p2 heuristic                    # you against a bot
    python -m connect4 --p1 random --p2 heuristic --games 1000
    python -m connect4 --p2 neural                       # against the trained network (stage 2)
    python -m connect4 --p2 neural --model models/winning-moves-cnn.pt   # a stage 3 network
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

from connect4.board import COLS, EMPTY, PLAYER_1, PLAYER_2, ROWS, SYMBOLS, Board, other_player
from connect4.game import play_game
from connect4.players import HeuristicPlayer, HumanPlayer, Player, QuitGame, RandomPlayer

PLAYER_TYPES = ("human", "random", "heuristic", "neural")
DEFAULT_MODEL = Path("models") / "winning-moves.npz"

_COLORS = {PLAYER_1: "\033[1;31m", PLAYER_2: "\033[1;33m"}  # bold red X, bold yellow O
_RESET = "\033[0m"


def render_board(board: Board, color: bool = False) -> str:
    """Draw the board as text, top row first, with column numbers 1-7 above."""
    lines = [" " + " ".join(str(col + 1) for col in range(COLS))]
    grid = board.grid
    for row in reversed(range(ROWS)):  # row 0 is the bottom, so print it last
        cells = []
        for col in range(COLS):
            value = int(grid[row, col])
            text = SYMBOLS[value]
            if color and value != EMPTY:
                text = f"{_COLORS[value]}{text}{_RESET}"
            cells.append(text)
        lines.append("|" + " ".join(cells) + "|")
    return "\n".join(lines)


def make_player(kind: str, seed: int | None = None, model: Path = DEFAULT_MODEL) -> Player:
    if kind == "neural":
        return _load_neural_player(model, seed)
    if kind == "human":
        return HumanPlayer()
    if kind == "random":
        return RandomPlayer(seed)
    if kind == "heuristic":
        return HeuristicPlayer(seed)
    raise ValueError(f"unknown player type: {kind}")


def _load_neural_player(model: Path, seed: int | None) -> Player:
    # Imported here so the plain game never needs the neural network code.
    pytorch = Path(model).suffix == ".pt"  # stage 3 networks; stage 2's are .npz
    if not Path(model).exists():
        trainer = f"connect4.{'torch_nn' if pytorch else 'numpy_nn'}.winning_moves"
        raise SystemExit(
            f"No trained network at {model}. Train one first (it takes a few minutes) with:\n"
            f"    python -m {trainer}"
        )
    if pytorch:
        from connect4.torch_nn.winning_moves import load_player

        return load_player(Path(model), seed)
    from connect4.numpy_nn import network
    from connect4.numpy_nn.winning_moves import NeuralPlayer

    return NeuralPlayer(network.load(model), seed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="connect4", description="Play Connect 4 in the terminal.")
    parser.add_argument(
        "--p1", choices=PLAYER_TYPES, default="human", help="player 1 (X), moves first"
    )
    parser.add_argument("--p2", choices=PLAYER_TYPES, default="human", help="player 2 (O)")
    parser.add_argument(
        "--games",
        type=int,
        default=1,
        help="number of games to play (bots only); more than 1 prints a results tally",
    )
    parser.add_argument("--seed", type=int, help="random seed, to make bot games repeatable")
    parser.add_argument("--no-color", action="store_true", help="draw the board without colours")
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_MODEL, help="trained network for the neural player"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.games < 1:
        parser.error("--games must be at least 1")
    if args.games > 1 and "human" in (args.p1, args.p2):
        parser.error("--games above 1 needs both players to be bots")

    seeds = random.Random(args.seed)
    player1 = make_player(args.p1, seeds.randrange(2**32), args.model)
    player2 = make_player(args.p2, seeds.randrange(2**32), args.model)
    labels = {PLAYER_1: f"X ({args.p1})", PLAYER_2: f"O ({args.p2})"}
    color = not args.no_color and sys.stdout.isatty() and "NO_COLOR" not in os.environ

    try:
        if args.games == 1:
            _show_game(player1, player2, labels, color)
        else:
            _play_many(player1, player2, args.games, labels)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    return 0


def _show_game(player1: Player, player2: Player, labels: dict[int, str], color: bool) -> None:
    """Play one game, drawing the board after every move."""
    print(f"Connect 4: {labels[PLAYER_1]} vs {labels[PLAYER_2]}\n")
    print(render_board(Board(), color))

    def on_move(board: Board) -> None:
        mover = other_player(board.current_player)  # the player who just moved
        print(f"\n{labels[mover]} played column {board.moves[-1] + 1}")
        print(render_board(board, color))

    try:
        result = play_game(player1, player2, on_move)
    except QuitGame:
        print("Game abandoned.")
        return
    if result.is_draw:
        print("\nIt's a draw!")
    else:
        print(f"\n{labels[result.winner]} wins!")


def _play_many(player1: Player, player2: Player, games: int, labels: dict[int, str]) -> None:
    """Play many bot-vs-bot games and print how often each side won."""
    counts: dict[int | None, int] = {PLAYER_1: 0, PLAYER_2: 0, None: 0}
    for _ in range(games):
        counts[play_game(player1, player2).winner] += 1

    rows = [
        (f"Player 1, {labels[PLAYER_1]}, wins:", counts[PLAYER_1]),
        (f"Player 2, {labels[PLAYER_2]}, wins:", counts[PLAYER_2]),
        ("Draws:", counts[None]),
    ]
    label_width = max(len(label) for label, _ in rows)
    count_width = len(str(games))
    print(f"Played {games} games: {labels[PLAYER_1]} vs {labels[PLAYER_2]}")
    for label, count in rows:
        print(f"  {label:<{label_width}}  {count:>{count_width}}  {100 * count / games:5.1f}%")
