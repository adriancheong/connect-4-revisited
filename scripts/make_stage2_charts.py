"""Draw the charts used in docs/stage-2/.

Run from the repository root:
    pip install -e ".[plots]"
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


def label_line_end(ax, x, y, text, color_ink=INK_2, dy=0.0):
    """Write a label just past the right end of a line."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(6, dy),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=color_ink,
    )


# ---------------------------------------------------------------------------
# Step 1: one neuron, one example
# ---------------------------------------------------------------------------


def chart_loss_vs_w1() -> None:
    """How the loss changes as w1 changes, with w2 and b held at their starting values."""
    x, target = neuron.EXAMPLE_X, neuron.EXAMPLE_TARGET
    w2, b = neuron.EXAMPLE_WEIGHTS[1], neuron.EXAMPLE_BIAS
    start_w1 = neuron.EXAMPLE_WEIGHTS[0]

    def loss_at(w1):
        return neuron.loss(neuron.predict(np.array([w1, w2]), b, x), target)

    w1_values = np.linspace(-3, 4, 300)
    losses = [loss_at(w) for w in w1_values]
    start_loss = loss_at(start_w1)
    slope, _ = neuron.gradients(neuron.EXAMPLE_WEIGHTS, b, x, target)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(w1_values, losses, color=SERIES[0])

    # The slope at the starting point, drawn as a short straight line touching the curve.
    span = np.array([-1.0, 1.0])
    ax.plot(start_w1 + span, start_loss + slope[0] * span, color=SERIES[1], linewidth=1.4)
    ax.plot(
        [start_w1], [start_loss], "o", color=SERIES[1], markersize=8,
        markeredgecolor=SURFACE, markeredgewidth=2, zorder=5,
    )  # fmt: skip
    ax.annotate(
        f"start: w1 = {start_w1}, loss = {start_loss:.2f}\nslope here = {slope[0]:.2f}",
        xy=(start_w1, start_loss),
        xytext=(-150, -38),
        textcoords="offset points",
        fontsize=9,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8},
    )
    ax.annotate(
        "",
        xy=(start_w1 + 2.0, 0.40),
        xytext=(start_w1 + 0.5, 0.40),
        arrowprops={"arrowstyle": "->", "color": INK_2, "linewidth": 1.2},
    )
    ax.text(
        start_w1 + 1.25, 0.42, "downhill: increase w1",
        ha="center", va="bottom", fontsize=9, color=INK_2,
    )  # fmt: skip

    ax.set_title("Loss as w1 changes (w2 and b held fixed)")
    ax.set_xlabel("w1")
    ax.set_ylabel("loss")
    ax.set_ylim(0, 0.75)
    save(fig, "01-loss-vs-w1.png")


# ---------------------------------------------------------------------------
# Step 2: one neuron, a whole dataset
# ---------------------------------------------------------------------------


def chart_or_training() -> None:
    _, _, history = neuron.train(neuron.INPUTS, neuron.OR_TARGETS, learning_rate=1.0, steps=1000)
    steps = [s.step for s in history]

    # Loss over training.
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(steps, [s.loss for s in history], color=SERIES[0])
    ax.set_title("Learning OR: average loss over training")
    ax.set_xlabel("training step")
    ax.set_ylabel("average loss")
    ax.set_ylim(bottom=0)
    end = history[-1]
    label_line_end(ax, end.step, end.loss, f"{end.loss:.4f}")
    save(fig, "02-or-loss.png")

    # Weights over training.
    w1 = [s.weights[0] for s in history]
    w2 = [s.weights[1] for s in history]
    bias = [s.bias for s in history]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(steps, w1, color=SERIES[0], label="w1")
    ax.plot(steps, w2, color=SERIES[1], label="w2")
    ax.plot(steps, bias, color=SERIES[2], label="b (bias)")
    ax.axhline(0, color=AXIS, linewidth=0.8, zorder=1)
    label_line_end(ax, steps[-1], (w1[-1] + w2[-1]) / 2, f"w1 and w2 → {w1[-1]:.2f}")
    label_line_end(ax, steps[-1], bias[-1], f"b → {bias[-1]:.2f}")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.62))
    ax.set_title("Learning OR: how the weights change")
    ax.set_xlabel("training step")
    ax.set_ylabel("value")
    save(fig, "02-or-weights.png")


def chart_learning_rates() -> None:
    rates = [0.1, 1.0, 10.0, 100.0]
    notes = {0.1: "slow", 1.0: "", 10.0: "fast", 100.0: "overshoots, then stuck"}
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for rate, color in zip(rates, ORDINAL_BLUES):
        _, _, history = neuron.train(
            neuron.INPUTS, neuron.OR_TARGETS, learning_rate=rate, steps=1000
        )
        steps = [s.step for s in history]
        losses = [s.loss for s in history]
        ax.plot(steps, losses, color=color, label=f"learning rate {rate:g}")
        text = f"{rate:g}" + (f": {notes[rate]}" if notes[rate] else "")
        if rate in (0.1, 100.0):
            label_line_end(ax, steps[-1], losses[-1], text)
        else:  # these two end almost on top of each other, so label them along the way
            at = 230 if rate == 1.0 else 30
            ax.annotate(
                text, xy=(at, losses[at]), xytext=(4, 6), textcoords="offset points",
                fontsize=9, color=INK_2,
            )  # fmt: skip
    ax.set_title("Learning OR with different learning rates")
    ax.set_xlabel("training step")
    ax.set_ylabel("average loss")
    ax.set_ylim(0, 0.27)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.55))
    save(fig, "02-learning-rates.png")


def chart_or_vs_xor() -> None:
    """The neuron's output over the whole input square, after training on OR and on XOR."""
    grid = np.linspace(-0.25, 1.25, 121)
    xx, yy = np.meshgrid(grid, grid)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.7), constrained_layout=True)
    for ax, (name, targets) in zip(axes, [("OR", neuron.OR_TARGETS), ("XOR", neuron.XOR_TARGETS)]):
        weights, bias, history = neuron.train(neuron.INPUTS, targets, learning_rate=1.0)
        output = neuron.sigmoid(points @ weights + bias).reshape(xx.shape)
        image = ax.imshow(
            output, origin="lower", extent=(-0.25, 1.25, -0.25, 1.25),
            cmap=DIVERGING, vmin=0, vmax=1,
        )  # fmt: skip
        correct = 0
        for (x1, x2), target in zip(neuron.INPUTS, targets):
            prediction = neuron.predict(weights, bias, np.array([x1, x2]))
            correct += round(prediction) == target
            ax.plot(
                x1, x2, "o", markersize=9, color=INK if target else SURFACE,
                markeredgecolor=INK, markeredgewidth=1.2,
            )  # fmt: skip
            ax.annotate(
                f"want {target:g}\ngot {prediction:.2f}", xy=(x1, x2),
                xytext=(0, 10 if x2 == 0 else -26), textcoords="offset points",
                ha="center", fontsize=8, color=INK,
            )  # fmt: skip
        verdict = "all 4 right" if correct == 4 else "stuck at 0.5"
        ax.set_title(f"{name}: {verdict}")
        ax.set_xlabel("input 1")
        ax.set_ylabel("input 2")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.grid(False)
    bar = fig.colorbar(image, ax=axes, shrink=0.8)
    bar.set_label("neuron output", color=INK_2)
    bar.outline.set_visible(False)
    save(fig, "02-or-vs-xor.png")


def main() -> None:
    chart_loss_vs_w1()
    chart_or_training()
    chart_learning_rates()
    chart_or_vs_xor()


if __name__ == "__main__":
    main()
