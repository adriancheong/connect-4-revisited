"""Stage 2, steps 1 and 2: one artificial neuron, and how it learns.

Read this alongside:
- docs/stage-2/00-start-here.md          (the big picture and the words used here)
- docs/stage-2/01-one-neuron-by-hand.md
- docs/stage-2/02-training-a-neuron.md

Run `python -m connect4.numpy_nn.neuron` to print the worked example from step 1.

Words used in this file (textbook names in brackets):
- weight: a dial on one input, saying how much the neuron listens to it
- bias: the neuron's starting lean, before it looks at any input
- error (loss): how far off a guess was. Nothing to do with losing a game.
- slope (gradient): which way, and how strongly, the error changes when a dial is turned up a little
- learning rate: the step size, i.e. how far the dials are turned each time
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# The neuron: make a guess
# ---------------------------------------------------------------------------


def sigmoid(z):
    """The squash function: turn any number into a confidence between 0 and 1.

    Big positive numbers come out near 1 ("yes"), big negative numbers near 0 ("no"),
    and 0 comes out as 0.5 ("no idea").
    """
    return 1.0 / (1.0 + np.exp(-z))


def predict(weights, bias, x):
    """The neuron's guess for the inputs `x`.

    Multiply each input by its weight, add them up, add the bias, then squash.
    """
    total = np.dot(weights, x) + bias  # w1*x1 + w2*x2 + ... + b
    return sigmoid(total)


def error(guess, answer):
    """How wrong a guess was: the difference, squared. 0 means exactly right.

    Squaring makes it positive whether the guess was too high or too low,
    and makes big misses count much more than small ones.
    """
    return (guess - answer) ** 2


# ---------------------------------------------------------------------------
# Which way to turn each dial: the slopes
#
# A slope answers: "if I turn this dial up a little, does the error go up or down,
# and by how much?" Training turns every dial a small step in the downhill direction.
# ---------------------------------------------------------------------------


def slopes_by_nudging(weights, bias, x, answer, nudge=1e-6):
    """Find each slope by experiment: turn one dial up a tiny bit and re-measure the error.

        slope = (error after the nudge - error before) / size of the nudge

    Easy to understand, but it needs one extra guess for every dial.
    """
    error_before = error(predict(weights, bias, x), answer)

    weight_slopes = np.zeros_like(weights)
    for i in range(len(weights)):
        nudged = weights.copy()
        nudged[i] += nudge
        weight_slopes[i] = (error(predict(nudged, bias, x), answer) - error_before) / nudge

    bias_slope = (error(predict(weights, bias + nudge, x), answer) - error_before) / nudge
    return weight_slopes, bias_slope


def slopes(weights, bias, x, answer):
    """Find the same slopes exactly, in one go, using the chain rule ("gear ratios").

    Turning a weight turns the total, which turns the guess, which turns the error:

        weight -> total -> guess -> error

    Each link has a simple ratio. Multiply the ratios to get the slope of the whole chain.
    """
    total = np.dot(weights, x) + bias
    guess = sigmoid(total)

    # Ratio 1: how much the error moves when the guess moves. Says how wrong, and which way.
    error_per_guess = 2 * (guess - answer)
    # Ratio 2: how much the guess moves when the total moves. Says how responsive the squash is.
    guess_per_total = guess * (1 - guess)
    error_per_total = error_per_guess * guess_per_total

    # Ratio 3: total = w1*x1 + w2*x2 + b, so turning weight i moves the total by x_i,
    # and turning the bias moves it by 1.
    weight_slopes = error_per_total * np.asarray(x, dtype=float)
    bias_slope = error_per_total
    return weight_slopes, bias_slope


def train_step(weights, bias, x, answer, learning_rate):
    """One training step on one example: turn every dial a small step downhill.

    The slope points uphill (towards more error), so we go the opposite way: minus.
    """
    weight_slopes, bias_slope = slopes(weights, bias, x, answer)
    new_weights = weights - learning_rate * weight_slopes
    new_bias = bias - learning_rate * bias_slope
    return new_weights, new_bias


# ---------------------------------------------------------------------------
# Step 2: training on a whole set of examples
# ---------------------------------------------------------------------------

# Every combination of two on/off inputs.
INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
OR_ANSWERS = np.array([0, 1, 1, 1], dtype=float)  # 1 if either input is on
XOR_ANSWERS = np.array([0, 1, 1, 0], dtype=float)  # 1 if exactly one input is on


@dataclass(frozen=True)
class Snapshot:
    """The neuron at one point during training."""

    round: int
    error: float  # average error over all the examples
    weights: np.ndarray
    bias: float


def average_error(weights, bias, inputs, answers):
    return float(np.mean([error(predict(weights, bias, x), a) for x, a in zip(inputs, answers)]))


def train(inputs, answers, learning_rate=1.0, rounds=1000, seed=0):
    """Train on a whole set of examples.

    Each round works out the slopes for every example, averages them, and takes one step.
    Averaging gives every example a vote, so a change that helps one example but hurts
    the others doesn't win.

    Returns the final weights and bias, plus a snapshot from before every round
    (and one at the end).
    """
    rng = np.random.default_rng(seed)
    weights = rng.normal(0, 0.5, size=inputs.shape[1])  # small random starting dials
    bias = 0.0

    history = []
    for round_number in range(rounds):
        history.append(
            Snapshot(round_number, average_error(weights, bias, inputs, answers), weights, bias)
        )

        total_weight_slopes = np.zeros_like(weights)
        total_bias_slope = 0.0
        for x, answer in zip(inputs, answers):
            weight_slopes, bias_slope = slopes(weights, bias, x, answer)
            total_weight_slopes += weight_slopes
            total_bias_slope += bias_slope

        weights = weights - learning_rate * total_weight_slopes / len(inputs)
        bias = bias - learning_rate * total_bias_slope / len(inputs)

    history.append(Snapshot(rounds, average_error(weights, bias, inputs, answers), weights, bias))
    return weights, bias, history


# ---------------------------------------------------------------------------
# The worked example from docs/stage-2/01-one-neuron-by-hand.md
# ---------------------------------------------------------------------------

EXAMPLE_X = np.array([1.0, 0.5])
EXAMPLE_ANSWER = 1.0
EXAMPLE_WEIGHTS = np.array([0.2, -0.4])
EXAMPLE_BIAS = 0.0


def main() -> None:
    x, answer = EXAMPLE_X, EXAMPLE_ANSWER
    weights, bias = EXAMPLE_WEIGHTS, EXAMPLE_BIAS

    print("Slopes at the start, two ways:")
    nudged_w, nudged_b = slopes_by_nudging(weights, bias, x, answer, nudge=0.001)
    exact_w, exact_b = slopes(weights, bias, x, answer)
    print(f"  by nudging 0.001:  w1 {nudged_w[0]:+.4f}  w2 {nudged_w[1]:+.4f}  b {nudged_b:+.4f}")
    print(f"  by gear ratios:    w1 {exact_w[0]:+.4f}  w2 {exact_w[1]:+.4f}  b {exact_b:+.4f}")
    print()

    print("Training on the one example, step size 1:")
    print(" round    w1      w2      b      guess    error")
    for round_number in range(11):
        guess = predict(weights, bias, x)
        print(
            f" {round_number:>4}  {weights[0]:6.3f}  {weights[1]:6.3f}  {bias:6.3f}"
            f"    {guess:.3f}    {error(guess, answer):.4f}"
        )
        weights, bias = train_step(weights, bias, x, answer, learning_rate=1.0)


if __name__ == "__main__":
    main()
