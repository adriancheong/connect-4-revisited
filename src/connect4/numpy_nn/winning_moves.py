"""Stage 2, step 4: teaching the network to spot winning moves in Connect 4.

Read this alongside:
- docs/stage-2/04a-boards-into-numbers.md   (inputs, flashcards, test cards, measuring success)
- docs/stage-2/04b-training-on-connect-4.md (training, results, looking inside, playing)

Train and save a network (about 2 minutes):
    python -m connect4.numpy_nn.winning_moves

Then play against it:
    connect4 --p2 neural

The question the network answers, for any board:
    "For each of the 7 columns: if the player to move plays here, do they win right now?"
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from connect4.board import COLS, ROWS, Board, other_player
from connect4.game import play_game
from connect4.numpy_nn import network as nw
from connect4.players import RandomPlayer

INPUTS = 2 * ROWS * COLS  # 84: two switches per cell
DEFAULT_MODEL = Path("models") / "winning-moves.npz"

# ---------------------------------------------------------------------------
# Boards into numbers
# ---------------------------------------------------------------------------


def encode(board: Board) -> np.ndarray:
    """84 switches (0 or 1), from the point of view of the player to move.

    Numbers 0-41:  "is MY piece in this cell?"         (one per cell)
    Numbers 42-83: "is my OPPONENT's piece in this cell?"
    An empty cell has both switches off.

    Cells are numbered row by row from the bottom: cell = row * 7 + column.
    """
    grid = board.grid
    me = board.current_player
    mine = (grid == me).ravel()
    theirs = (grid == other_player(me)).ravel()
    return np.concatenate([mine, theirs]).astype(float)


def encode_simple(board: Board) -> np.ndarray:
    """The other way to do it, for comparison: 42 numbers, +1 mine, -1 theirs, 0 empty."""
    grid = board.grid
    me = board.current_player
    return ((grid == me).astype(float) - (grid == other_player(me)).astype(float)).ravel()


def swap_sides(inputs: np.ndarray) -> np.ndarray:
    """Swap "mine" and "opponent's", to look at a board from the other player's side."""
    half = inputs.shape[-1] // 2
    return np.concatenate([inputs[..., half:], inputs[..., :half]], axis=-1)


def answers(board: Board, player: int | None = None) -> np.ndarray:
    """The right answers: 7 numbers, 1 for each column where `player` wins right now.

    `player` defaults to the player to move. The game engine works these out exactly.
    """
    result = np.zeros(COLS)
    result[board.winning_moves(player)] = 1.0
    return result


# ---------------------------------------------------------------------------
# Making the flashcards
# ---------------------------------------------------------------------------


def make_positions(count: int, seed: int = 1) -> list[Board]:
    """`count` different positions from random games, none of them finished.

    Each position appears once, even if several games passed through it.
    """
    player1, player2 = RandomPlayer(seed=seed), RandomPlayer(seed=seed + 1)
    seen: dict[tuple, Board] = {}
    while len(seen) < count:
        for board in play_game(player1, player2).boards:
            if not board.is_terminal():
                seen.setdefault((board.grid.tobytes(), board.current_player), board)
    return list(seen.values())[:count]


@dataclass(frozen=True)
class Dataset:
    practice_inputs: np.ndarray
    practice_answers: np.ndarray
    test_inputs: np.ndarray
    test_answers: np.ndarray
    test_boards: list[Board]


def make_dataset(practice: int = 300_000, test: int = 30_000, seed: int = 1, encoder=encode):
    """Practice cards and test cards. Every position is different, so the test
    positions are ones the network never practises on."""
    boards = make_positions(practice + test, seed)
    practice_boards, test_boards = boards[:practice], boards[practice:]
    return Dataset(
        practice_inputs=np.array([encoder(b) for b in practice_boards]),
        practice_answers=np.array([answers(b) for b in practice_boards]),
        test_inputs=np.array([encoder(b) for b in test_boards]),
        test_answers=np.array([answers(b) for b in test_boards]),
        test_boards=test_boards,
    )


# ---------------------------------------------------------------------------
# Measuring success
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scores:
    error: float  # average (guess - answer) squared, over every card and column
    found: float  # of positions WITH a win: how often its top guess says "win" and really wins
    false_alarms: float  # of positions WITHOUT one: how often any guess says "win" (above 0.5)
    all_right: float  # how often all 7 guesses are on the right side of 0.5


def score(guesses: np.ndarray, right: np.ndarray) -> Scores:
    has_win = right.sum(axis=1) > 0
    top_pick = guesses.argmax(axis=1)
    claims_win = guesses.max(axis=1) > 0.5  # it says "win" for its top column...
    top_pick_wins = right[np.arange(len(right)), top_pick] == 1  # ...and that column really wins
    return Scores(
        error=float(np.mean((guesses - right) ** 2)),
        found=float((claims_win & top_pick_wins)[has_win].mean()),
        false_alarms=float((guesses[~has_win] > 0.5).any(axis=1).mean()),
        all_right=float((np.round(guesses) == right).all(axis=1).mean()),
    )


def evaluate(net: nw.Network, inputs: np.ndarray, right: np.ndarray) -> Scores:
    return score(nw.predict(net, inputs), right)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    practice: Scores
    test: Scores


def train(
    data: Dataset,
    hidden: int = 128,
    learning_rate: float = 10.0,
    epochs: int = 50,
    batch_size: int = 64,
    seed: int = 0,
    report=None,
) -> tuple[nw.Network, list[EpochRecord]]:
    """Train a network on the practice cards, scoring practice and test after every epoch.

    `report(record)` is called after every epoch, for printing progress.
    """
    net = nw.new_network(data.practice_inputs.shape[1], hidden, COLS, seed)
    history = []

    def after_each_epoch(epoch, net):
        record = EpochRecord(
            epoch,
            evaluate(net, data.practice_inputs, data.practice_answers),
            evaluate(net, data.test_inputs, data.test_answers),
        )
        history.append(record)
        if report is not None:
            report(record)

    net = nw.train_in_batches(
        net, data.practice_inputs, data.practice_answers, learning_rate, epochs,
        batch_size, seed, after_each_epoch,
    )  # fmt: skip
    return net, history


# ---------------------------------------------------------------------------
# A player that uses the network
# ---------------------------------------------------------------------------


class NeuralPlayer:
    """Plays like HeuristicPlayer, but sees wins and threats through the network.

    1. If the network thinks one of my moves wins (guess above 0.5), play it.
    2. Otherwise, look at the board from my opponent's side. If the network thinks
       they have a winning move, block that column.
    3. Otherwise, play randomly.
    """

    def __init__(self, net: nw.Network, seed: int | None = None, name: str = "Neural") -> None:
        self.net = net
        self.name = name
        self._rng = random.Random(seed)

    def guesses(self, board: Board) -> tuple[np.ndarray, np.ndarray]:
        """The network's 7 guesses for me, and for my opponent."""
        x = encode(board)
        both = nw.predict(self.net, np.array([x, swap_sides(x)]))
        return both[0], both[1]

    def choose_move(self, board: Board) -> int:
        legal = board.legal_moves()
        mine, theirs = self.guesses(board)
        for guesses in (mine, theirs):  # winning comes first, then blocking
            best = max(legal, key=lambda col: guesses[col])
            if guesses[best] > 0.5:
                return best
        return self._rng.choice(legal)


# ---------------------------------------------------------------------------
# Command line: train and save
# ---------------------------------------------------------------------------


def _print_record(record: EpochRecord) -> None:
    p, t = record.practice, record.test
    print(
        f"  epoch {record.epoch:>3}: practice error {p.error:.4f} | test error {t.error:.4f}"
        f" | finds the win {t.found:6.1%} | false alarms {t.false_alarms:5.1%}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the winning-move network and save it.")
    parser.add_argument("--hidden", type=int, default=128, help="hidden neurons (default 128)")
    parser.add_argument("--epochs", type=int, default=50, help="passes through the data")
    parser.add_argument("--learning-rate", type=float, default=10.0, help="step size")
    parser.add_argument("--positions", type=int, default=300_000, help="practice positions")
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL, help="where to save it")
    args = parser.parse_args(argv)

    started = time.time()
    print(f"Making {args.positions:,} practice and 30,000 test positions from random games...")
    data = make_dataset(practice=args.positions, test=30_000)
    print(f"Training {args.hidden} hidden neurons for {args.epochs} epochs:")
    net, history = train(
        data, args.hidden, args.learning_rate, args.epochs, report=_print_record
    )  # fmt: skip
    args.out.parent.mkdir(parents=True, exist_ok=True)
    nw.save(net, args.out)
    final = history[-1].test
    print(
        f"Done in {time.time() - started:.0f}s. On test positions it finds the win "
        f"{final.found:.1%} of the time, with {final.false_alarms:.1%} false alarms."
    )
    play = "connect4 --p2 neural" + ("" if args.out == DEFAULT_MODEL else f" --model {args.out}")
    print(f"Saved to {args.out}. Play against it with: {play}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
