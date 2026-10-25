"""Draw the charts for docs/stage-2/04a and 04b, and print every number the notes quote.

Run from the repository root (takes about 15 minutes, because it trains several networks):
    python scripts/make_stage2_step4_charts.py

Everything is seeded, so the same computer gives the same numbers every time.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_stage2_charts as base  # noqa: E402  (shared colours, styles and helpers)
from make_stage2_charts import (  # noqa: E402
    AXIS,
    DIVERGING,
    INK,
    INK_2,
    MUTED,
    RED_PIECE,
    SERIES,
    SURFACE,
    YELLOW_PIECE,
    plt,
    save,
)

from connect4.board import COLS, ROWS, Board  # noqa: E402
from connect4.game import play_game  # noqa: E402
from connect4.numpy_nn import network as nw  # noqa: E402
from connect4.numpy_nn import winning_moves as wm  # noqa: E402
from connect4.players import HeuristicPlayer, RandomPlayer  # noqa: E402

EXAMPLE_MOVES = [3, 3, 2, 2, 1, 4]  # red to move; column 1 wins
TWO_QUESTIONS_MOVES = [0, 6, 1, 6, 2, 6]  # red to move; red wins in column 4, yellow in 7
PIECE = {1: RED_PIECE, 2: YELLOW_PIECE}
NAME = {1: "red", 2: "yellow"}


def say(text: str) -> None:
    print(text, flush=True)


# ---------------------------------------------------------------------------
# 04a: boards into numbers
# ---------------------------------------------------------------------------


def draw_board(ax, board: Board, left: float, bottom: float, cell: float = 1.0) -> None:
    grid = board.grid
    for row in range(ROWS):
        for col in range(COLS):
            value = int(grid[row, col])
            ax.add_patch(
                plt.Circle(
                    (left + col * cell, bottom + row * cell), cell * 0.4,
                    facecolor=PIECE.get(value, "#f0efec"), edgecolor=AXIS, linewidth=0.6,
                )
            )  # fmt: skip


def draw_switches(ax, values: np.ndarray, left: float, bottom: float, color: str) -> None:
    """A 6x7 grid of 0/1 switches, bottom row first."""
    grid = values.reshape(ROWS, COLS)
    for row in range(ROWS):
        for col in range(COLS):
            on = grid[row, col] == 1
            ax.add_patch(
                plt.Rectangle(
                    (left + col - 0.45, bottom + row - 0.45), 0.9, 0.9,
                    facecolor=color if on else "#f0efec", edgecolor=AXIS, linewidth=0.5,
                )
            )  # fmt: skip
            ax.text(left + col, bottom + row, "1" if on else "0", ha="center", va="center",
                    fontsize=6.5, color=SURFACE if on else MUTED)  # fmt: skip


def diagram_encoding() -> None:
    board = Board.from_moves(EXAMPLE_MOVES)
    x = wm.encode(board)
    fig, ax = base.blank_canvas(10.5, 3.6, (-1.0, 30.5), (-2.8, 7.4))
    draw_board(ax, board, 0, 0)
    ax.text(3, 6.4, "The board (red to move)", ha="center", fontsize=10, weight="bold")
    base.arrow(ax, (7.0, 2.5), (8.4, 2.5))
    draw_switches(ax, x[:42], 9.5, 0, SERIES[0])
    ax.text(12.5, 6.4, "Is it MY piece? (red)", ha="center", fontsize=10, weight="bold")
    ax.text(12.5, -1.1, "inputs 1–42", ha="center", va="top", fontsize=9, color=INK_2)
    ax.text(17.2, 2.5, "+", ha="center", va="center", fontsize=16, color=INK_2)
    draw_switches(ax, x[42:], 18.5, 0, SERIES[1])
    ax.text(21.5, 6.4, "Is it my OPPONENT's? (yellow)", ha="center", fontsize=10, weight="bold")
    ax.text(21.5, -1.1, "inputs 43–84", ha="center", va="top", fontsize=9, color=INK_2)
    base.arrow(ax, (25.6, 2.5), (27.0, 2.5))
    ax.text(28.6, 2.5, "84\nswitches", ha="center", va="center", fontsize=10, weight="bold")
    save(fig, "04-encoding.png")


# ---------------------------------------------------------------------------
# 04b: training and results
# ---------------------------------------------------------------------------


def chart_training(history) -> None:
    epochs = [r.epoch for r in history]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
    left.plot(epochs, [r.practice.error for r in history], color=SERIES[0], label="practice cards")
    left.plot(epochs, [r.test.error for r in history], color=SERIES[1], label="test cards")
    left.set_title("Average error")
    left.set_xlabel("epoch (passes through the practice cards)")
    left.set_ylabel("average error (lower is better)")
    left.set_ylim(bottom=0)
    left.legend(loc="upper right")
    found = [100 * r.test.found for r in history]
    right.plot(epochs, found, color=SERIES[0])
    right.set_title("Test cards: finds the winning column")
    right.set_xlabel("epoch (passes through the practice cards)")
    right.set_ylabel("% of positions that have a win")
    right.set_ylim(0, 100)
    base.label_line_end(right, epochs[-1], found[-1], f"{found[-1]:.1f}%")
    save(fig, "04-training.png")


def pick_examples(net, data):
    guesses = nw.predict(net, data.test_inputs)
    right = data.test_answers
    wins = right.sum(axis=1)
    top = guesses.argmax(axis=1)
    top_right = right[np.arange(len(right)), top] == 1
    rules = [
        ("The network found it", (wins == 1) & top_right & (guesses.max(1) > 0.9)),
        ("The network found both", (wins == 2) & ((guesses > 0.5) == (right == 1)).all(1)),
        ("The network correctly said no to all 7", (wins == 0) & (guesses.max(1) < 0.1)
         & (np.array([len(b.moves) for b in data.test_boards]) > 12)),
        ("A mistake: the network missed it",
         (wins == 1) & ~top_right & (guesses.max(1) < 0.5)),
    ]  # fmt: skip
    return [(title, int(np.flatnonzero(mask)[0])) for title, mask in rules], guesses


def what_wins(answers: np.ndarray) -> str:
    """'no column wins', 'column 5 wins' or 'columns 5 and 7 win'."""
    cols = [str(c + 1) for c in np.flatnonzero(answers)]
    if not cols:
        return "no column wins"
    if len(cols) == 1:
        return f"column {cols[0]} wins"
    return f"columns {', '.join(cols[:-1])} and {cols[-1]} win"


def chart_examples(net, data) -> None:
    examples, guesses = pick_examples(net, data)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.6), constrained_layout=True)
    for ax, (title, i) in zip(axes.ravel(), examples):
        board, g, right = data.test_boards[i], guesses[i], data.test_answers[i]
        ax.set_xlim(-0.8, 6.8)
        ax.set_ylim(-1.6, 8.6)
        ax.set_aspect("equal")
        ax.axis("off")
        draw_board(ax, board, 0, 0)
        for col in range(COLS):
            height = 1.7 * g[col]
            ax.add_patch(plt.Rectangle((col - 0.32, 6.0), 0.64, max(height, 0.02),
                                       facecolor=SERIES[0], edgecolor="none"))  # fmt: skip
            ax.text(col, 6.05 + height, f"{g[col]:.2f}", ha="center", va="bottom", fontsize=8,
                    color=INK)  # fmt: skip
            if right[col] == 1:
                ax.text(col, -0.9, "wins", ha="center", va="top", fontsize=8.5, weight="bold")
        ax.plot([-0.5, 6.5], [6.0, 6.0], color=AXIS, linewidth=0.8)
        mover = NAME[board.current_player].capitalize()
        ax.set_title(f"{mover} to move: {what_wins(right)}\n{title}", fontsize=10)
    save(fig, "04-examples.png")
    return examples


def chart_two_questions(net, mine: wm.Scores, theirs: wm.Scores) -> None:
    """One board asked both questions, and how well each is answered on the test cards."""
    board = Board.from_moves(TWO_QUESTIONS_MOVES)
    x = wm.encode(board)
    guesses = nw.predict(net, np.array([x, wm.swap_sides(x)]))
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(9.4, 5.4), constrained_layout=True, width_ratios=[1.45, 1]
    )  # fmt: skip

    left.set_xlim(-0.8, 6.8)
    left.set_ylim(-1.6, 11.4)
    left.set_aspect("equal")
    left.axis("off")
    draw_board(left, board, 0, 0)
    strips = [
        (8.9, "Where can I (red) win?", SERIES[0], guesses[0]),
        (6.1, "Where could my opponent (yellow) win?", SERIES[1], guesses[1]),
    ]
    for base_y, label, color, g in strips:
        left.add_patch(plt.Rectangle((-0.5, base_y + 1.95), 0.3, 0.3, facecolor=color,
                                     edgecolor="none"))  # fmt: skip
        left.text(-0.05, base_y + 2.1, label, va="center", fontsize=9, color=INK)
        for col in range(COLS):
            height = 1.4 * g[col]
            left.add_patch(plt.Rectangle((col - 0.32, base_y), 0.64, max(height, 0.02),
                                         facecolor=color, edgecolor="none"))  # fmt: skip
            left.text(col, base_y + 0.05 + height, f"{g[col]:.2f}", ha="center", va="bottom",
                      fontsize=8, color=INK)  # fmt: skip
        left.plot([-0.5, 6.5], [base_y, base_y], color=AXIS, linewidth=0.8)
    for player in (board.current_player, 3 - board.current_player):
        for col in board.winning_moves(player):
            left.text(col, -0.9, f"{NAME[player]} wins", ha="center", va="top", fontsize=8.5,
                      weight="bold")  # fmt: skip
    left.set_title("One board, both questions (red to move)")

    questions = [("Question 1\nwhere can I win?", mine),
                 ("Question 2\nwhere could my\nopponent win?", theirs)]  # fmt: skip
    values = [100 * s.found for _, s in questions]
    right.bar([0, 1], values, width=0.55, color=SERIES[:2])
    for xpos, value in enumerate(values):
        right.text(xpos, value + 1.5, f"{value:.1f}%", ha="center", fontsize=9, color=INK)
    right.set_xticks([0, 1], [f"{name}\n(false alarms {s.false_alarms:.1%})"
                              for name, s in questions])  # fmt: skip
    right.set_xlim(-0.7, 1.7)
    right.set_ylim(0, 108)
    right.set_ylabel("% of positions that have a win")
    right.set_title("Test positions: finds the winning column")
    right.grid(axis="x")
    save(fig, "04-two-questions.png")


def chart_hidden_neurons(net) -> list[tuple[int, int, float]]:
    """The six hidden neurons with the strongest positive vote for some column."""
    w, v = net.hidden_weights, net.output_weights
    best_col = v.argmax(axis=1)
    strength = v.max(axis=1)
    chosen = np.argsort(-strength)[:6]
    fig, axes = plt.subplots(3, 4, figsize=(9.0, 7.4), constrained_layout=True)
    picked = []
    for k, j in enumerate(chosen):
        row, pair = divmod(k, 2)
        limit = np.abs(w[:, j]).max()
        for half, label in enumerate(["my pieces", "opponent's pieces"]):
            ax = axes[row, 2 * pair + half]
            image = ax.imshow(
                w[half * 42 : (half + 1) * 42, j].reshape(ROWS, COLS), origin="lower",
                cmap=DIVERGING, vmin=-limit, vmax=limit,
            )  # fmt: skip
            ax.set_xticks(range(COLS), [str(c + 1) for c in range(COLS)], fontsize=7)
            ax.set_yticks(range(ROWS), [str(r + 1) for r in range(ROWS)], fontsize=7)
            ax.grid(False)
            ax.set_title(f"Neuron {j}: {label}", fontsize=9)
        axes[row, 2 * pair].set_ylabel(f"votes for column {best_col[j] + 1}", fontsize=8.5)
        picked.append((int(j), int(best_col[j]) + 1, float(strength[j])))
    bar = fig.colorbar(image, ax=axes, shrink=0.5, location="bottom", aspect=40)
    bar.set_ticks([])
    bar.set_label(
        "weights, not a board  ·  blue = a piece here pushes the neuron to fire  ·  red = pushes"
        " it not to  ·  rows counted from the bottom",
        color=INK_2,
    )  # fmt: skip
    bar.outline.set_visible(False)
    save(fig, "04-hidden-neurons.png")
    return picked


def describe_neuron(net, data, j: int = 108, col: int = 5) -> None:
    """Print the numbers section 8 quotes for one neuron: its weights in one column, its bias
    and votes, and how often it fires for different column contents on the test positions."""
    w = net.hidden_weights[:, j].reshape(2, ROWS, COLS)
    say(f"neuron {j}, column {col + 1} weights by row, bottom first: "
        f"mine {np.round(w[0, :, col], 1)}, opponent's {np.round(w[1, :, col], 1)}")  # fmt: skip
    rest = np.delete(w, col, axis=2)
    say(f"  other weights {rest.min():.1f} to {rest.max():.1f}, bias {net.hidden_biases[j]:.2f}, "
        f"votes for columns 1-7 {np.round(net.output_weights[j], 1)}")  # fmt: skip
    fires = nw.forward(net, data.test_inputs).hidden[:, j] > 0.5
    says_win = nw.predict(net, data.test_inputs)[:, col] > 0.5
    stacks: dict[str, list[int]] = {}
    for i, board in enumerate(data.test_boards):
        me = board.current_player
        stack = "".join("M" if v == me else "O" for v in board.grid[:, col] if v)  # bottom first
        stacks.setdefault(stack, []).append(i)
    for stack in ("MMM", "OMMM", "OOMMM", "MMMO", "OMM"):
        idx = stacks[stack]
        say(f"  column {col + 1} = {stack:5} ({len(idx):>3} positions)"
            f": fires {fires[idx].mean():.0%}, really wins {data.test_answers[idx, col].mean():.0%}"
            f", network says win {says_win[idx].mean():.0%}")  # fmt: skip
    blocked = [i for stack, idx in stacks.items() if stack.endswith("O") for i in idx]
    say(f"  opponent's piece on top ({len(blocked):,} positions)"
        f": fires {fires[blocked].mean():.1%}")  # fmt: skip


DIRECTIONS = {"vertical": [(1, 0)], "horizontal": [(0, 1)], "diagonal": [(1, 1), (1, -1)]}


def line_directions(board: Board, col: int) -> list[str]:
    """Which directions of four-in-a-row the winning move `col` completes."""
    me, row = board.current_player, int(np.count_nonzero(board.grid[:, col]))
    grid = board.play(col).grid
    found = []
    for name, steps in DIRECTIONS.items():
        for dr, dc in steps:
            in_line = 1
            for sign in (1, -1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < ROWS and 0 <= c < COLS and grid[r, c] == me:
                    in_line += 1
                    r, c = r + sign * dr, c + sign * dc
            if in_line >= 4 and name not in found:
                found.append(name)
    return found


def wins_spotted_by_direction(net, data) -> dict[str, tuple[int, int]]:
    """For winning moves that complete one line: (spotted, total) for each direction.

    "Spotted" means the network's guess for that column is above 0.5.
    """
    guesses = nw.predict(net, data.test_inputs)
    counts = {name: [0, 0] for name in DIRECTIONS}
    for i, board in enumerate(data.test_boards):
        for col in np.flatnonzero(data.test_answers[i]):
            found = line_directions(board, int(col))
            if len(found) == 1:
                counts[found[0]][0] += int(guesses[i, col] > 0.5)
                counts[found[0]][1] += 1
    return {name: (spotted, total) for name, (spotted, total) in counts.items()}


def chart_bars(name, title, ylabel, groups, series, decimals=0, xlabel=None) -> None:
    """Grouped bar chart. `series` is a list of (label, colour, values)."""
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    width = 0.8 / len(series)
    positions = np.arange(len(groups))
    for k, (label, color, values) in enumerate(series):
        xs = positions - 0.4 + width * (k + 0.5)
        ax.bar(xs, values, width=width * 0.92, color=color, label=label)
        for x, value in zip(xs, values):
            ax.text(x, value + 1.5, f"{value:.{decimals}f}%", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(positions, groups)
    ax.set_ylim(0, 108)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(axis="x")
    if len(series) > 1:
        ax.legend(loc="upper left")
    save(fig, name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def train_and_score(data, batch, rate, epochs, checks, practice=False) -> None:
    """Train a fresh 128-neuron network; print its test (and practice) scores at `checks`."""

    def report(epoch, net):
        if epoch in checks:
            say(f"  batch {batch}, step {rate:g}, epoch {epoch}: "
                f"test {wm.evaluate(net, data.test_inputs, data.test_answers)}")  # fmt: skip
            if practice:
                score = wm.evaluate(net, data.practice_inputs, data.practice_answers)
                say(f"  {'':>{len(str(batch)) + len(f'{rate:g}') + 20}}practice {score}")

    net = nw.new_network(data.practice_inputs.shape[1], 128, COLS, seed=0)
    nw.train_in_batches(
        net, data.practice_inputs, data.practice_answers, rate, epochs, batch, 0, report
    )  # fmt: skip


def tally(player1, player2, games=1000) -> str:
    wins = {1: 0, 2: 0, None: 0}
    for _ in range(games):
        wins[play_game(player1, player2).winner] += 1
    return f"P1 {wins[1] / games:.1%}  P2 {wins[2] / games:.1%}  draws {wins[None] / games:.1%}"


def main() -> None:
    started = time.time()
    diagram_encoding()

    data = wm.make_dataset()
    has_win = data.test_answers.sum(axis=1) > 0
    say(f"test positions with a winning move: {has_win.mean():.1%}")
    say(f"per-column 'yes' rate (practice): {np.round(data.practice_answers.mean(0), 3)}")
    no_guess = np.zeros_like(data.test_answers)
    say(f"'always no' scores: {wm.score(no_guess, data.test_answers)}")

    say("\nMain run: 128 hidden neurons, step size 10, 50 epochs")
    net, history = wm.train(data, hidden=128, learning_rate=10.0, epochs=50)
    for r in history:
        if r.epoch in (1, 2, 5, 10, 20, 30, 40, 50):
            say(f"  epoch {r.epoch:>2}: practice {r.practice}\n            test     {r.test}")
    chart_training(history)
    for title, i in chart_examples(net, data):
        say(f"example '{title}': test position {i}, moves {list(data.test_boards[i].moves)}")
    say(f"hidden neurons drawn (neuron, column, vote): {chart_hidden_neurons(net)}")
    describe_neuron(net, data)
    by_direction = wins_spotted_by_direction(net, data)
    say(f"winning moves spotted, by direction (spotted, total): {by_direction}")
    chart_bars(
        "04-directions.png", "Which wins it spots (test positions)",
        "winning moves spotted (%)",
        [f"{name}\n({total:,} moves)" for name, (_, total) in by_direction.items()],
        [("spotted", SERIES[0], [100 * s / t for s, t in by_direction.values()])], decimals=1,
    )  # fmt: skip

    their_view = wm.swap_sides(data.test_inputs)
    their_answers = np.array([wm.answers(b, 3 - b.current_player) for b in data.test_boards])
    theirs = wm.score(nw.predict(net, their_view), their_answers)
    say(f"opponent's view, never practised: {theirs}")
    chart_two_questions(net, history[-1].test, theirs)

    neural = wm.NeuralPlayer(net, seed=1)
    say("\nGames (1,000 each):")
    say(f"  neural vs random:       {tally(neural, RandomPlayer(2))}")
    say(f"  random vs neural:       {tally(RandomPlayer(3), neural)}")
    say(f"  neural vs heuristic:    {tally(neural, HeuristicPlayer(4))}")
    say(f"  heuristic vs neural:    {tally(HeuristicPlayer(5), neural)}")
    right_moves = total = 0
    p1, p2 = RandomPlayer(7), RandomPlayer(8)
    for _ in range(300):
        for board in play_game(p1, p2).boards:
            wins = board.winning_moves()
            threats = board.winning_moves(3 - board.current_player) if board.moves else []
            if board.is_terminal() or not (wins or threats):
                continue
            move = neural.choose_move(board)
            total += 1
            right_moves += move in (wins or threats)
    say(f"  win-or-block moments handled correctly: {right_moves / total:.1%} of {total}")

    say("\nHidden layer sizes (step size 10, 30 epochs):")
    found_by_size = {}
    for hidden in (8, 16, 32, 64, 128):
        sized, h = wm.train(data, hidden=hidden, learning_rate=10.0, epochs=30)
        found_by_size[hidden] = h[-1]
        say(f"  {hidden:>3}: practice {h[-1].practice}\n       test     {h[-1].test}")
    chart_bars(
        "04-hidden-sizes.png", "Bigger hidden layers find more wins (test positions)",
        "finds the winning column (%)", [str(k) for k in found_by_size],
        [("test", SERIES[0], [100 * r.test.found for r in found_by_size.values()])],
        xlabel="hidden neurons",
    )  # fmt: skip

    say("\nMemorising: 128 hidden neurons, less practice data:")
    memo = {}
    for count, epochs in ((2_000, 450), (20_000, 45)):
        small = wm.Dataset(
            data.practice_inputs[:count], data.practice_answers[:count],
            data.test_inputs, data.test_answers, data.test_boards,
        )  # fmt: skip
        _, h = wm.train(small, hidden=128, learning_rate=10.0, epochs=epochs)
        memo[f"{count:,}"] = h[-1]
        say(f"  {count:>7,} positions: practice {h[-1].practice}")
        say(f"  {count:>7,} positions: test     {h[-1].test}")
    memo["300,000"] = found_by_size[128]
    chart_bars(
        "04-memorising.png", "Less practice data: memorising instead of learning",
        "finds the winning column (%)", list(memo),
        [("positions it practised on", SERIES[0], [100 * r.practice.found for r in memo.values()]),
         ("test positions (never seen)", SERIES[1], [100 * r.test.found for r in memo.values()])],
        xlabel="how many practice positions",
    )  # fmt: skip

    say("\nBoard encodings (64 hidden neurons, 30 epochs):")
    simple = wm.make_dataset(encoder=wm.encode_simple)
    for lr in (10.0, 20.0):
        _, h42 = wm.train(simple, hidden=64, learning_rate=lr, epochs=30)
        _, h84 = wm.train(data, hidden=64, learning_rate=lr, epochs=30)
        say(f"  step {lr:g}: 42 numbers {h42[-1].test}\n           84 switches {h84[-1].test}")

    say("\nTraining longer (128 hidden neurons, step size 10, batches of 64):")
    train_and_score(data, 64, 10.0, 500, {50, 100, 200, 300, 500}, practice=True)
    say("\nBatch sizes (128 hidden neurons, step size 10, 10 epochs; batch 16 also to 50):")
    for size in (4, 16, 64, 256, 1024, len(data.practice_inputs)):
        train_and_score(data, size, 10.0, 50 if size == 16 else 10, {10, 50})
    say("Bigger batches with bigger steps (10 epochs):")
    for size, rate in ((64, 40.0), (256, 40.0), (256, 160.0)):
        train_and_score(data, size, rate, 10, {10})

    say(f"\nDone in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
