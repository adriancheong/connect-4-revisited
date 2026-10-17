"""Draw the diagrams and charts used in docs/stage-2/.

Run from the repository root:
    uv sync --locked --all-extras   (see the README; this includes matplotlib)
    python scripts/make_stage2_charts.py

Every chart is generated from the real code in src/connect4/numpy_nn, so the
pictures always match what the code does.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

from connect4.numpy_nn import neuron  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "stage-2" / "images"

# Colours (light theme).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua
ORDINAL_BLUES = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # light -> dark
NEURON_FILL = "#cde2fb"
RED_PIECE = "#e34948"
YELLOW_PIECE = "#eda100"
DIVERGING = LinearSegmentedColormap.from_list("red_gray_blue", ["#e34948", "#f0efec", "#2a78d6"])

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "lines.linewidth": 1.8,
        "legend.frameon": False,
    }
)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {OUT / name}")


def label_line_end(ax, x, y, text, dy=0.0):
    """Write a label just past the right end of a line."""
    ax.annotate(
        text, xy=(x, y), xytext=(6, dy), textcoords="offset points",
        va="center", fontsize=9, color=INK_2,
    )  # fmt: skip


def blank_canvas(width, height, xlim, ylim):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def node(ax, x, y, radius=0.32, fill=NEURON_FILL, edge=SERIES[0], text=None, size=9):
    ax.add_patch(Circle((x, y), radius, facecolor=fill, edgecolor=edge, linewidth=1.4, zorder=3))
    if text:
        ax.text(x, y, text, ha="center", va="center", fontsize=size, color=INK, zorder=4)


def arrow(ax, start, end, color=INK_2, width=1.2):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=12, color=color, linewidth=width, zorder=2
        )
    )


def column_y(count, spacing=1.0):
    """Evenly spaced heights for `count` circles, centred on 0."""
    return [spacing * (i - (count - 1) / 2) for i in range(count)][::-1]


# ---------------------------------------------------------------------------
# Start here: what a network looks like
# ---------------------------------------------------------------------------


def diagram_network() -> None:
    fig, ax = blank_canvas(7.2, 5.0, (-1.6, 8.6), (-4.2, 3.6))
    layers = [(0.0, column_y(3, 1.3)), (3.5, column_y(4, 1.2)), (7.0, column_y(1))]

    # Connections: every circle in one layer to every circle in the next.
    for (x1, ys1), (x2, ys2) in zip(layers, layers[1:]):
        for y1 in ys1:
            for y2 in ys2:
                ax.plot([x1, x2], [y1, y2], color=AXIS, linewidth=1.0, zorder=1)
    # Highlight one connection and one neuron, to label them.
    hx1, hy1, hx2, hy2 = 0.0, layers[0][1][0], 3.5, layers[1][1][0]
    ax.plot([hx1, hx2], [hy1, hy2], color=SERIES[1], linewidth=2.2, zorder=2)

    for x, ys in layers:
        for y in ys:
            node(ax, x, y)
    node(ax, 3.5, layers[1][1][-1], fill=SERIES[0], edge=SERIES[0])

    ax.text(1.75, 2.0, "a connection:\nits weight is a dial", ha="center", fontsize=9,
            color=SERIES[1], weight="bold")  # fmt: skip
    ax.annotate(
        "a neuron", xy=(3.85, layers[1][1][-1]), xytext=(5.0, -2.75), fontsize=9,
        color=SERIES[0], weight="bold",
        arrowprops={"arrowstyle": "-", "color": SERIES[0], "linewidth": 0.9},
    )  # fmt: skip

    for x, title, note in [
        (0.0, "Inputs", "the facts\n(e.g. the board)"),
        (3.5, "Hidden layer", "working notes"),
        (7.0, "Output", "the answer\n(e.g. how good\nis this board?)"),
    ]:
        ax.text(x, 3.2, title, ha="center", fontsize=10, weight="bold", color=INK)
        ax.text(x, {0.0: -1.9, 3.5: -2.45, 7.0: -0.6}[x], note, ha="center", va="top",
                fontsize=9, color=INK_2)  # fmt: skip

    arrow(ax, (0.6, -4.0), (6.4, -4.0), color=MUTED)
    ax.text(3.5, -3.9, "information flows this way", ha="center", va="bottom", fontsize=8,
            color=MUTED)  # fmt: skip
    save(fig, "00-network.png")


def diagram_connect4_network() -> None:
    fig, ax = blank_canvas(7.6, 4.4, (-0.4, 13.2), (-4.3, 3.6))

    # A small Connect 4 board.
    board = np.zeros((6, 7), dtype=int)
    for col, row, player in [(3, 0, 1), (3, 1, 2), (2, 0, 1), (4, 0, 2), (1, 0, 1), (2, 1, 2)]:
        board[row, col] = player
    cell = 0.42
    left, bottom = 0.0, -1.3
    for row in range(6):
        for col in range(7):
            color = {0: "#f0efec", 1: RED_PIECE, 2: YELLOW_PIECE}[board[row, col]]
            ax.add_patch(
                Circle((left + col * cell, bottom + row * cell), cell * 0.4, facecolor=color,
                       edgecolor=AXIS, linewidth=0.6)
            )  # fmt: skip
    ax.text(left + 3 * cell, 2.75, "The board", ha="center", fontsize=10, weight="bold")
    ax.text(left + 3 * cell, -1.85, "6 rows × 7 columns\n= 42 cells", ha="center", va="top",
            fontsize=9, color=INK_2)  # fmt: skip
    arrow(ax, (3.2, 0.0), (4.3, 0.0))

    # Inputs (42, drawn as a few plus dots), hidden layer, 7 outputs.
    x_in, x_hidden, x_out = 5.0, 8.3, 11.6
    inputs = [2.1, 1.5, 0.9, -0.9, -1.5, -2.1]
    hidden = [1.8, 1.0, 0.2, -1.4, -2.2]
    outputs = column_y(7, 0.62)
    for ys1, ys2, x1, x2 in [(inputs, hidden, x_in, x_hidden), (hidden, outputs, x_hidden, x_out)]:
        for y1 in ys1:
            for y2 in ys2:
                ax.plot([x1, x2], [y1, y2], color=AXIS, linewidth=0.6, zorder=1)
    for y in inputs:
        node(ax, x_in, y, radius=0.22)
    for y in hidden:
        node(ax, x_hidden, y, radius=0.22)
    for i, y in enumerate(outputs):
        node(ax, x_out, y, radius=0.22)
        ax.text(x_out + 0.4, y, f"column {i + 1}", va="center", fontsize=8, color=INK_2)
    for x, y in [(x_in, 0.0), (x_hidden, -0.6)]:
        ax.text(x, y, "⋮", ha="center", va="center", fontsize=14, color=MUTED)

    ax.text(x_in, 2.75, "42 inputs", ha="center", fontsize=10, weight="bold")
    ax.text(x_hidden, 2.75, "Hidden layer", ha="center", fontsize=10, weight="bold")
    ax.text(x_out, 2.75, "7 outputs", ha="center", fontsize=10, weight="bold")
    ax.text(x_in, -2.6, "one per cell", ha="center", va="top", fontsize=9, color=INK_2)
    ax.text(x_hidden, -2.6, "patterns it finds\nfor itself", ha="center", va="top", fontsize=9,
            color=INK_2)  # fmt: skip
    ax.text(x_out + 0.6, -2.6, "does playing this\ncolumn win right now?\n(0 = no, 1 = yes)",
            ha="center", va="top", fontsize=9, color=INK_2)  # fmt: skip
    save(fig, "00-connect4-network.png")


# ---------------------------------------------------------------------------
# Step 1: one neuron, one example
# ---------------------------------------------------------------------------


def diagram_neuron() -> None:
    fig, ax = blank_canvas(7.4, 3.4, (-1.2, 11.4), (-2.6, 2.8))
    x1, x2 = neuron.EXAMPLE_X
    w1, w2 = neuron.EXAMPLE_WEIGHTS

    node(ax, 0, 1.2, radius=0.5, fill="#f0efec", edge=AXIS, text=f"x1\n{x1:.1f}")
    node(ax, 0, -1.2, radius=0.5, fill="#f0efec", edge=AXIS, text=f"x2\n{x2:.1f}")
    ax.text(0, 2.2, "inputs", ha="center", fontsize=10, weight="bold")

    centre = (4.2, 0.0)
    arrow(ax, (0.5, 1.1), (3.25, 0.25), color=SERIES[1], width=1.6)
    arrow(ax, (0.5, -1.1), (3.25, -0.25), color=SERIES[1], width=1.6)
    ax.text(1.9, 1.15, f"× {w1:g}  (dial w1)", fontsize=9, color=SERIES[1], weight="bold",
            ha="center", rotation=-17)  # fmt: skip
    ax.text(1.9, -1.15, f"× {w2:g}  (dial w2)", fontsize=9, color=SERIES[1], weight="bold",
            ha="center", va="top", rotation=17)  # fmt: skip

    ax.add_patch(Circle(centre, 0.95, facecolor=NEURON_FILL, edgecolor=SERIES[0], linewidth=1.6))
    ax.text(centre[0], centre[1] + 0.25, "add up", ha="center", fontsize=9, weight="bold")
    ax.text(centre[0], centre[1] - 0.3, "0.2 − 0.2 + 0\n= 0", ha="center", va="center",
            fontsize=8, color=INK_2)  # fmt: skip
    arrow(ax, (centre[0], 2.25), (centre[0], 1.0), color=SERIES[2], width=1.6)
    ax.text(centre[0], 2.35, "bias b = 0\n(starting lean)", ha="center", fontsize=9,
            color=INK_2)  # fmt: skip

    arrow(ax, (5.2, 0), (6.3, 0))
    ax.add_patch(
        FancyBboxPatch((6.35, -0.5), 1.9, 1.0, boxstyle="round,pad=0.05,rounding_size=0.2",
                       facecolor=NEURON_FILL, edgecolor=SERIES[0], linewidth=1.6)
    )  # fmt: skip
    ax.text(7.3, 0.12, "squash", ha="center", fontsize=9, weight="bold")
    ax.text(7.3, -0.25, "into 0 to 1", ha="center", fontsize=8, color=INK_2)
    arrow(ax, (8.3, 0), (9.3, 0))
    ax.text(10.1, 0.15, "guess", ha="center", fontsize=10, weight="bold")
    ax.text(10.1, -0.35, "0.5", ha="center", fontsize=12, weight="bold", color=SERIES[0])

    ax.add_patch(
        FancyBboxPatch((3.1, -2.45), 5.3, 3.7, boxstyle="round,pad=0.05,rounding_size=0.3",
                       facecolor="none", edgecolor=AXIS, linewidth=1.0, linestyle=(0, (3, 3)))
    )  # fmt: skip
    ax.text(5.75, -2.2, "the neuron", ha="center", fontsize=9, color=MUTED)
    save(fig, "01-neuron.png")


def chart_error_landscape() -> None:
    """The error for every setting of dial w1, with w2 and b held at their starting values."""
    x, answer = neuron.EXAMPLE_X, neuron.EXAMPLE_ANSWER
    w2, b = neuron.EXAMPLE_WEIGHTS[1], neuron.EXAMPLE_BIAS
    start_w1 = neuron.EXAMPLE_WEIGHTS[0]

    def error_at(w1):
        return neuron.error(neuron.predict(np.array([w1, w2]), b, x), answer)

    w1_values = np.linspace(-3, 4, 300)
    start_error = error_at(start_w1)
    slope, _ = neuron.slopes(neuron.EXAMPLE_WEIGHTS, b, x, answer)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(w1_values, [error_at(w) for w in w1_values], color=SERIES[0])

    # The tilt of the curve at the starting point, drawn as a short straight line.
    span = np.array([-1.0, 1.0])
    ax.plot(start_w1 + span, start_error + slope[0] * span, color=SERIES[1], linewidth=1.4)
    ax.plot(
        [start_w1], [start_error], "o", color=SERIES[1], markersize=8,
        markeredgecolor=SURFACE, markeredgewidth=2, zorder=5,
    )  # fmt: skip
    ax.annotate(
        f"you are here: w1 = {start_w1}, error = {start_error:.2f}",
        xy=(start_w1, start_error), xytext=(0.75, 0.62), textcoords="data",
        fontsize=9, color=INK, arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8},
    )  # fmt: skip
    ax.text(-0.95, 0.52, f"orange line: the tilt here\n(slope = {slope[0]:.2f})", ha="right",
            fontsize=9, color=SERIES[1])  # fmt: skip
    arrow(ax, (start_w1 + 0.6, 0.22), (start_w1 + 2.1, 0.22))
    ax.text(start_w1 + 1.35, 0.24, "downhill: turn w1 up", ha="center", va="bottom",
            fontsize=9, color=INK_2)  # fmt: skip

    ax.set_title("The error for every setting of dial w1")
    ax.set_xlabel("setting of dial w1  (w2 and b stay as they are)")
    ax.set_ylabel("error (lower is better)")
    ax.set_ylim(0, 0.75)
    save(fig, "01-error-landscape.png")


# ---------------------------------------------------------------------------
# Step 2: one neuron, a whole set of examples
# ---------------------------------------------------------------------------


def chart_or_training() -> None:
    _, _, history = neuron.train(neuron.INPUTS, neuron.OR_ANSWERS, learning_rate=1.0, rounds=1000)
    rounds = [s.round for s in history]

    # Learning curve: average error over training.
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(rounds, [s.error for s in history], color=SERIES[0])
    ax.set_title("Learning curve: the error falls as the neuron practises")
    ax.set_xlabel("training round (practice)")
    ax.set_ylabel("average error (lower is better)")
    ax.set_ylim(bottom=0)
    start, end = history[0], history[-1]
    ax.annotate(
        f"start: {start.error:.2f}\n(guessing about 0.5 for everything)",
        xy=(0, start.error), xytext=(40, -8), textcoords="offset points", fontsize=9, color=INK_2,
    )  # fmt: skip
    ax.annotate(
        f"after 1,000 rounds: {end.error:.3f}", xy=(end.round, end.error), xytext=(-150, 22),
        textcoords="offset points", fontsize=9, color=INK_2,
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8},
    )  # fmt: skip
    save(fig, "02-learning-curve.png")

    # The dials over training.
    w1 = [s.weights[0] for s in history]
    w2 = [s.weights[1] for s in history]
    bias = [s.bias for s in history]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(rounds, w1, color=SERIES[0], label="dial w1")
    ax.plot(rounds, w2, color=SERIES[1], label="dial w2")
    ax.plot(rounds, bias, color=SERIES[2], label="bias b")
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    label_line_end(ax, rounds[-1], (w1[-1] + w2[-1]) / 2, f"w1 and w2 → {w1[-1]:.2f}")
    label_line_end(ax, rounds[-1], bias[-1], f"b → {bias[-1]:.2f}")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.62))
    ax.set_title("The dials during training")
    ax.set_xlabel("training round (practice)")
    ax.set_ylabel("dial setting")
    save(fig, "02-or-dials.png")


def chart_step_sizes() -> None:
    rates = [0.1, 1.0, 10.0, 100.0]
    notes = {0.1: "slow", 1.0: "", 10.0: "fast", 100.0: "jumps too far, then stuck"}
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for rate, color in zip(rates, ORDINAL_BLUES):
        _, _, history = neuron.train(neuron.INPUTS, neuron.OR_ANSWERS, learning_rate=rate)
        rounds = [s.round for s in history]
        errors = [s.error for s in history]
        ax.plot(rounds, errors, color=color, label=f"step size {rate:g}")
        text = f"{rate:g}" + (f": {notes[rate]}" if notes[rate] else "")
        if rate in (0.1, 100.0):
            label_line_end(ax, rounds[-1], errors[-1], text)
        else:  # these two end almost on top of each other, so label them along the way
            at = 230 if rate == 1.0 else 30
            ax.annotate(
                text, xy=(at, errors[at]), xytext=(4, 6), textcoords="offset points",
                fontsize=9, color=INK_2,
            )  # fmt: skip
    ax.set_title("Same training, four different step sizes")
    ax.set_xlabel("training round (practice)")
    ax.set_ylabel("average error (lower is better)")
    ax.set_ylim(0, 0.27)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.55))
    save(fig, "02-step-sizes.png")


def chart_or_vs_xor() -> None:
    """The neuron's answer for every point in the input square, after training on OR and XOR."""
    grid = np.linspace(-0.25, 1.25, 121)
    xx, yy = np.meshgrid(grid, grid)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8), constrained_layout=True)
    for ax, (name, answers) in zip(axes, [("OR", neuron.OR_ANSWERS), ("XOR", neuron.XOR_ANSWERS)]):
        weights, bias, _ = neuron.train(neuron.INPUTS, answers, learning_rate=1.0)
        output = neuron.sigmoid(points @ weights + bias).reshape(xx.shape)
        image = ax.imshow(
            output, origin="lower", extent=(-0.25, 1.25, -0.25, 1.25),
            cmap=DIVERGING, vmin=0, vmax=1,
        )  # fmt: skip
        correct = 0
        for (x1, x2), answer in zip(neuron.INPUTS, answers):
            guess = neuron.predict(weights, bias, np.array([x1, x2]))
            correct += round(guess) == answer
            ax.plot(
                x1, x2, "o", markersize=9, color=INK if answer else SURFACE,
                markeredgecolor=INK, markeredgewidth=1.2,
            )  # fmt: skip
            ax.annotate(
                f"want {answer:g}\ngot {guess:.2f}", xy=(x1, x2),
                xytext=(0, 10 if x2 == 0 else -26), textcoords="offset points",
                ha="center", fontsize=8, color=INK,
            )  # fmt: skip
        ax.set_title(f"{name}: {'learned it' if correct == 4 else 'can’t learn it'}")
        ax.set_xlabel("input 1")
        ax.set_ylabel("input 2")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.grid(False)
    bar = fig.colorbar(image, ax=axes, shrink=0.8)
    bar.set_label("neuron's answer (0 = no, 1 = yes)", color=INK_2)
    bar.outline.set_visible(False)
    save(fig, "02-or-vs-xor.png")


def main() -> None:
    diagram_network()
    diagram_connect4_network()
    diagram_neuron()
    chart_error_landscape()
    chart_or_training()
    chart_step_sizes()
    chart_or_vs_xor()


if __name__ == "__main__":
    main()
