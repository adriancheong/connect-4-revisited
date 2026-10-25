"""Draw the charts for docs/stage-3, and print every number the notes quote.

Run from the repository root (takes about 30 minutes, because it trains several networks):
    python scripts/make_stage3_charts.py

Everything is seeded. Results on the graphics card can differ very slightly from run to
run, because it adds numbers up in a different order each time.
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_stage2_charts as base  # noqa: E402  (shared colours, styles and helpers)
import make_stage2_step4_charts as s4  # noqa: E402  (games, directions, win-or-block)
from make_stage2_charts import (  # noqa: E402
    AXIS,
    DIVERGING,
    INK,
    INK_2,
    MUTED,
    SERIES,
    SURFACE,
    plt,
    save,
)

from connect4.board import COLS, ROWS, Board  # noqa: E402
from connect4.numpy_nn import winning_moves as wm  # noqa: E402
from connect4.players import HeuristicPlayer, RandomPlayer  # noqa: E402
from connect4.torch_nn import convolution as conv  # noqa: E402
from connect4.torch_nn import winning_moves as tw  # noqa: E402
from connect4.torch_nn.models import dial_count  # noqa: E402

base.OUT = Path(__file__).resolve().parent.parent / "docs" / "stage-3" / "images"
CPU = torch.device("cpu")
GPU = torch.device("cuda") if torch.cuda.is_available() else CPU
LABELS = {
    "stage2": "stage 2's recipe",
    "cross_entropy": "+ cross-entropy",
    "relu": "+ ReLU",
    "adam": "+ Adam",
    "cnn": "CNN",
}
COLORS = {"stage2": MUTED, "cross_entropy": SERIES[2], "relu": SERIES[0], "adam": SERIES[1],
          "cnn": SERIES[1]}  # fmt: skip


def say(text: str) -> None:
    print(text, flush=True)


def short(s: wm.Scores) -> str:
    return (f"finds {s.found:.1%}, false alarms {s.false_alarms:.1%}, all right {s.all_right:.1%},"
            f" error {s.error:.4f}")  # fmt: skip


def seconds_per_epoch(recipe: tw.Recipe, data: wm.Dataset, device: torch.device) -> float:
    """Time for one epoch of training (no scoring), after a short warm-up."""
    small = wm.Dataset(data.practice_inputs[:2_000], data.practice_answers[:2_000],
                       data.test_inputs, data.test_answers, data.test_boards)  # fmt: skip
    tw.train(replace(recipe, epochs=1), small, device, practice=False)  # warm-up
    started = time.perf_counter()
    model = tw.build(recipe).to(device)
    error, optimizer = tw.error_measure(recipe), tw.make_optimizer(recipe, model)
    x = torch.as_tensor(data.practice_inputs, dtype=torch.float32, device=device)
    y = torch.as_tensor(data.practice_answers, dtype=torch.float32, device=device)
    for start in range(0, len(x), recipe.batch_size):
        batch = slice(start, start + recipe.batch_size)
        loss = error(model(x[batch]), y[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter() - started


def report_player(name: str, model) -> None:
    player = tw.TorchNeuralPlayer(model, seed=1)
    right, total = s4.win_or_block_moments(player)
    say(f"  {name}: win-or-block moments handled correctly {right / total:.1%} of {total}")
    say(f"    vs random:      {s4.tally(player, RandomPlayer(2))}")
    say(f"    random vs:      {s4.tally(RandomPlayer(3), player)}")
    say(f"    vs heuristic:   {s4.tally(player, HeuristicPlayer(4))}")
    say(f"    heuristic vs:   {s4.tally(HeuristicPlayer(5), player)}")


def directions(model, data) -> dict[str, tuple[int, int]]:
    """Winning moves spotted, by direction: the stage 2 helper, with this model's guesses."""
    guesses = tw.guesses(model, data.test_inputs)
    counts = {name: [0, 0] for name in s4.DIRECTIONS}
    for i, board in enumerate(data.test_boards):
        for col in np.flatnonzero(data.test_answers[i]):
            found = s4.line_directions(board, int(col))
            if len(found) == 1:
                counts[found[0]][0] += int(guesses[i, col] > 0.5)
                counts[found[0]][1] += 1
    return {name: (spotted, total) for name, (spotted, total) in counts.items()}


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------


def chart_upgrades(histories: dict[str, list]) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for name, history in histories.items():
        epochs = [r.epoch for r in history]
        found = [100 * r.test.found for r in history]
        ax.plot(epochs, found, color=COLORS[name],
                label=f"{LABELS[name]}: {found[-1]:.1f}% after 50")  # fmt: skip
    ax.set_title("Each upgrade, added one at a time (test positions)")
    ax.set_xlabel("epoch (passes through the practice cards)")
    ax.set_ylabel("finds the winning column (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right")
    save(fig, "02-upgrades.png")


def chart_convolution() -> None:
    """The step 3a worked example: the board's 'mine' plane, the detector, and its map."""
    board = Board.from_moves(conv.EXAMPLE_MOVES)
    mine = conv.planes(board)[0]
    out = conv.slide(conv.planes(board), conv.STACK_WEIGHTS, conv.STACK_BIAS)
    fig, ax = base.blank_canvas(9.4, 4.0, (-1.0, 23.5), (-1.6, 7.6))
    ax.text(3, 6.9, "My pieces (red), as switches", ha="center", fontsize=10, weight="bold")
    for row in range(ROWS):
        for col in range(COLS):
            on = mine[row, col] == 1
            ax.add_patch(plt.Rectangle((col - 0.45, row - 0.45), 0.9, 0.9, edgecolor=AXIS,
                                       facecolor=SERIES[0] if on else "#f0efec", linewidth=0.5))
            ax.text(col, row, "1" if on else "0", ha="center", va="center", fontsize=8,
                    color=SURFACE if on else MUTED)  # fmt: skip
    for r0, color in ((0, SERIES[1]), (1, MUTED)):  # the window at column 3, two positions
        ax.add_patch(plt.Rectangle((2 - 0.5, r0 - 0.5), 1.0, 3.0, fill=False, edgecolor=color,
                                   linewidth=2.2, linestyle="-" if r0 == 0 else "--"))
    for c in range(COLS):
        ax.text(c, -0.95, str(c + 1), ha="center", va="center", fontsize=8, color=INK_2)
    ax.text(3, -1.5, "column", ha="center", fontsize=8, color=MUTED)

    ax.text(10.0, 6.9, "The detector", ha="center", fontsize=10, weight="bold")
    for i in range(3):
        ax.add_patch(plt.Rectangle((9.55, 1.55 + i), 0.9, 0.9, facecolor=SERIES[1], alpha=0.25,
                                   edgecolor=SERIES[1]))  # fmt: skip
        ax.text(10.0, 2.0 + i, "1", ha="center", va="center", fontsize=10)
    ax.text(10.0, 0.9, "bias −2", ha="center", fontsize=9, color=INK_2)
    ax.text(10.0, 0.2, "(weights on my pieces;\n0 on the opponent's)", ha="center", va="top",
            fontsize=8, color=MUTED)  # fmt: skip
    base.arrow(ax, (11.3, 3.0), (13.6, 3.0))
    ax.text(12.45, 3.5, "slide to every\nposition", ha="center", fontsize=8, color=INK_2)

    ax.text(18.5, 6.9, "Its map: is the pattern here?", ha="center", fontsize=10, weight="bold")
    for row in range(out.shape[0]):
        for col in range(out.shape[1]):
            hit = out[row, col] > 0
            ax.add_patch(plt.Rectangle((15.5 + col - 0.45, 1 + row - 0.45), 0.9, 0.9,
                                       edgecolor=AXIS, linewidth=0.5,
                                       facecolor=SERIES[1] if hit else "#f0efec"))  # fmt: skip
            ax.text(15.5 + col, 1 + row, f"{out[row, col]:.0f}", ha="center", va="center",
                    fontsize=8, color=SURFACE if hit else MUTED)  # fmt: skip
    ax.text(18.5, 0.0, "row = the detector's bottom cell (rows 1–4)", ha="center", fontsize=8,
            color=MUTED)  # fmt: skip
    save(fig, "03-convolution.png")


def chart_cnn_detectors(model) -> list[int]:
    """Eight first-layer detectors: the ones with the biggest weights."""
    w = model.layer1.weight.detach().cpu().numpy()  # (detectors, 2 planes, 4, 4)
    chosen = list(np.argsort(-np.abs(w).reshape(len(w), -1).sum(1))[:8])
    fig, axes = plt.subplots(4, 4, figsize=(8.0, 8.2), constrained_layout=True)
    for k, d in enumerate(chosen):
        limit = np.abs(w[d]).max()
        for plane, label in enumerate(["mine", "opponent's"]):
            ax = axes[k // 2, 2 * (k % 2) + plane]
            image = ax.imshow(w[d, plane], origin="lower", cmap=DIVERGING, vmin=-limit,
                              vmax=limit)  # fmt: skip
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_title(f"Detector {d}: {label}", fontsize=9)
    bar = fig.colorbar(image, ax=axes, shrink=0.5, location="bottom", aspect=40)
    bar.set_ticks([])
    bar.set_label("weights, 4×4 cells, bottom row at the bottom  ·  blue = a piece here counts"
                  " for  ·  red = against", color=INK_2)  # fmt: skip
    bar.outline.set_visible(False)
    save(fig, "03-cnn-detectors.png")
    return [int(d) for d in chosen]


def chart_directions(by_net: dict[str, dict[str, tuple[int, int]]]) -> None:
    names = list(s4.DIRECTIONS)
    labels = {"stage2": "stage 2's recipe", "relu": "best 84 → 128 → 7 (step 2b)", "cnn": "CNN"}
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    width = 0.8 / len(by_net)
    for k, (net, counts) in enumerate(by_net.items()):
        values = [100 * counts[n][0] / counts[n][1] for n in names]
        xs = np.arange(len(names)) - 0.4 + width * (k + 0.5)
        ax.bar(xs, values, width=width * 0.92, color=COLORS[net], label=labels[net])
        for x, value in zip(xs, values):
            ax.text(x, value + 1.5, f"{value:.1f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(names)), names)
    ax.set_ylim(0, 110)
    ax.set_ylabel("winning moves spotted (%)")
    ax.set_title("Which wins each network spots (test positions)")
    ax.grid(axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    save(fig, "03-directions.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    started = time.time()
    data = wm.make_dataset()
    say(f"Devices: processor, and {torch.cuda.get_device_name(0) if GPU.type == 'cuda' else '-'}")

    say("\n== Step 2a: stage 2's recipe in PyTorch (processor) ==")
    histories, models = {}, {}
    models["stage2"], histories["stage2"] = tw.train(tw.RECIPES["stage2"], data, CPU)
    for r in histories["stage2"]:
        if r.epoch in (1, 2, 5, 10, 20, 30, 40, 50):
            say(f"  epoch {r.epoch:>2}: practice {short(r.practice)}\n            test     "
                f"{short(r.test)}")  # fmt: skip
    for device in (CPU, GPU):
        secs = seconds_per_epoch(tw.RECIPES["stage2"], data, device)
        say(f"  one epoch on the {device.type}: {secs:.1f}s")
    report_player("stage 2's recipe", models["stage2"])

    say("\n== Step 2b: step sizes for plain SGD (10 epochs, chosen on the practice cards) ==")
    for name in ("cross_entropy", "relu"):
        for lr in (1.0, 3.0, 10.0, 30.0):
            _, h = tw.train(replace(tw.RECIPES[name], learning_rate=lr, epochs=10), data, CPU)
            say(f"  {name:<13} step {lr:>4g}: practice {short(h[-1].practice)}")

    say("\n== Step 2b: the upgrades, one at a time (50 epochs, processor) ==")
    for name in ("cross_entropy", "relu", "adam"):
        models[name], histories[name] = tw.train(tw.RECIPES[name], data, CPU)
    for name, h in histories.items():
        at = {r.epoch: r.test.found for r in h}
        say(f"  {name:<13} {tw.RECIPES[name]}")
        say("     test after 1/5/10/20/50 epochs: "
            + " / ".join(f"{at[e]:.1%}" for e in (1, 5, 10, 20, 50)))  # fmt: skip
        say(f"     final practice {short(h[-1].practice)}\n     final test     {short(h[-1].test)}")
    chart_upgrades(histories)
    report_player("best 84-128-7 (relu)", models["relu"])
    right, total = s4.win_or_block_moments(tw.TorchNeuralPlayer(models["adam"], seed=1))
    say(f"  adam: win-or-block moments handled correctly {right / total:.1%} of {total}")

    say("\n== Step 3a: the worked example ==")
    board = Board.from_moves(conv.EXAMPLE_MOVES)
    say(f"  map:\n{conv.slide(conv.planes(board), conv.STACK_WEIGHTS, conv.STACK_BIAS)[::-1]}")
    chart_convolution()

    say("\n== Step 3b: the CNN ==")
    cnn_recipe = tw.RECIPES["cnn"]
    say(f"  dials: CNN {dial_count(tw.build(cnn_recipe)):,}, "
        f"84-128-7 network {dial_count(tw.build(tw.RECIPES['relu'])):,}")  # fmt: skip
    for batch_size in (64, 512):
        for device in (CPU, GPU):
            secs = seconds_per_epoch(replace(cnn_recipe, batch_size=batch_size), data, device)
            say(f"  one epoch, batches of {batch_size}, on the {device.type}: {secs:.1f}s")
    models["cnn"], histories["cnn"] = tw.train(cnn_recipe, data, GPU)
    for r in histories["cnn"]:
        if r.epoch in (1, 2, 5, 10, 20, 30, 40, 50):
            say(f"  epoch {r.epoch:>2}: practice {short(r.practice)}\n            test     "
                f"{short(r.test)}")  # fmt: skip
    final = histories["cnn"][-1].test
    with_win = int((data.test_answers.sum(axis=1) > 0).sum())
    without = len(data.test_answers) - with_win
    say(f"  cnn exactly: finds the win in {round(final.found * with_win)} of {with_win}, "
        f"false alarms on {round(final.false_alarms * without)} of {without}")  # fmt: skip
    say(f"  detectors drawn: {chart_cnn_detectors(models['cnn'])}")
    by_net = {name: directions(models[name], data) for name in ("stage2", "relu", "cnn")}
    for name, counts in by_net.items():
        say(f"  directions, {name}: {counts}")
    chart_directions(by_net)
    report_player("cnn", models["cnn"])
    their_view = wm.swap_sides(data.test_inputs)
    their_answers = np.array([wm.answers(b, 3 - b.current_player) for b in data.test_boards])
    theirs = wm.score(tw.guesses(models["cnn"], their_view), their_answers)
    say(f"  cnn, opponent's view: {short(theirs)}")

    say(f"\nDone in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
