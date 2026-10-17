"""Stage 2, steps 1 and 2: one artificial neuron, trained by gradient descent.

Read this alongside:
- docs/stage-2/01-one-neuron-by-hand.md
- docs/stage-2/02-training-a-neuron.md

Run `python -m connect4.numpy_nn.neuron` to print the worked example from step 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# The neuron
# ---------------------------------------------------------------------------


def sigmoid(z):
    """Squash any number into the range 0 to 1.

    Big positive numbers come out near 1, big negative numbers near 0, and 0 comes out as 0.5.
    """
    return 1.0 / (1.0 + np.exp(-z))


def predict(weights, bias, x):
    """The neuron's output for one input `x`: a weighted sum, then a squash."""
    z = np.dot(weights, x) + bias  # w1*x1 + w2*x2 + ... + b
    return sigmoid(z)


def loss(prediction, target):
    """Squared error: how far the prediction is from the target, squared. 0 means perfect."""
    return (prediction - target) ** 2


# ---------------------------------------------------------------------------
# The slope ("gradient") of the loss for each weight
#
# A slope answers: "if I nudge this weight up a little, does the loss go up or down,
# and by how much?" Training moves every weight a small step downhill.
# ---------------------------------------------------------------------------


def gradients_by_nudging(weights, bias, x, target, nudge=1e-6):
    """Find each slope by experiment: nudge one weight up a tiny bit and re-measure the loss.

        slope = (loss after the nudge - loss before) / size of the nudge

    Easy to understand, but it needs one extra prediction for every weight.
    """
    loss_before = loss(predict(weights, bias, x), target)

    weight_slopes = np.zeros_like(weights)
    for i in range(len(weights)):
        nudged = weights.copy()
        nudged[i] += nudge
        weight_slopes[i] = (loss(predict(nudged, bias, x), target) - loss_before) / nudge

    bias_slope = (loss(predict(weights, bias + nudge, x), target) - loss_before) / nudge
    return weight_slopes, bias_slope


def gradients(weights, bias, x, target):
    """Find the same slopes exactly, with the chain rule from calculus.

    The loss depends on a weight through a chain of simple steps:

        weight -> z (weighted sum) -> prediction (sigmoid) -> loss

    The slope of the whole chain is the slopes of the individual links multiplied together.
    """
    z = np.dot(weights, x) + bias
    prediction = sigmoid(z)

    dloss_dprediction = 2 * (prediction - target)  # how wrong, and in which direction
    dprediction_dz = prediction * (1 - prediction)  # how strongly the sigmoid responds here
    dloss_dz = dloss_dprediction * dprediction_dz

    # z = w1*x1 + w2*x2 + b, so nudging weight i changes z by x_i, and nudging b changes z by 1.
    weight_slopes = dloss_dz * np.asarray(x, dtype=float)
    bias_slope = dloss_dz
    return weight_slopes, bias_slope


def train_step(weights, bias, x, target, learning_rate):
    """One step of gradient descent on one example: move every weight a little way downhill."""
    weight_slopes, bias_slope = gradients(weights, bias, x, target)
    new_weights = weights - learning_rate * weight_slopes
    new_bias = bias - learning_rate * bias_slope
    return new_weights, new_bias


# ---------------------------------------------------------------------------
# Step 2: training on a whole dataset
# ---------------------------------------------------------------------------

# Every combination of two on/off inputs.
INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
OR_TARGETS = np.array([0, 1, 1, 1], dtype=float)  # 1 if either input is on
XOR_TARGETS = np.array([0, 1, 1, 0], dtype=float)  # 1 if exactly one input is on


@dataclass(frozen=True)
class Snapshot:
    """The state of the neuron at one point during training."""

    step: int
    loss: float  # average loss over the whole dataset
    weights: np.ndarray
    bias: float


def average_loss(weights, bias, inputs, targets):
    return float(np.mean([loss(predict(weights, bias, x), t) for x, t in zip(inputs, targets)]))


def train(inputs, targets, learning_rate=1.0, steps=1000, seed=0):
    """Gradient descent over a whole dataset.

    Each step works out the slopes for every example, averages them, and takes one step.
    Averaging gives every example a vote, so a change that helps one example but hurts
    the others doesn't win.

    Returns the final weights and bias, plus a snapshot from before every step (and one at the end).
    """
    rng = np.random.default_rng(seed)
    weights = rng.normal(0, 0.5, size=inputs.shape[1])  # small random starting weights
    bias = 0.0

    history = []
    for step in range(steps):
        history.append(Snapshot(step, average_loss(weights, bias, inputs, targets), weights, bias))

        total_weight_slopes = np.zeros_like(weights)
        total_bias_slope = 0.0
        for x, target in zip(inputs, targets):
            weight_slopes, bias_slope = gradients(weights, bias, x, target)
            total_weight_slopes += weight_slopes
            total_bias_slope += bias_slope

        weights = weights - learning_rate * total_weight_slopes / len(inputs)
        bias = bias - learning_rate * total_bias_slope / len(inputs)

    history.append(Snapshot(steps, average_loss(weights, bias, inputs, targets), weights, bias))
    return weights, bias, history


# ---------------------------------------------------------------------------
# The worked example from docs/stage-2/01-one-neuron-by-hand.md
# ---------------------------------------------------------------------------

EXAMPLE_X = np.array([1.0, 0.5])
EXAMPLE_TARGET = 1.0
EXAMPLE_WEIGHTS = np.array([0.2, -0.4])
EXAMPLE_BIAS = 0.0


def main() -> None:
    x, target = EXAMPLE_X, EXAMPLE_TARGET
    weights, bias = EXAMPLE_WEIGHTS, EXAMPLE_BIAS

    print("Slopes at the start, two ways:")
    nudged_w, nudged_b = gradients_by_nudging(weights, bias, x, target, nudge=0.001)
    exact_w, exact_b = gradients(weights, bias, x, target)
    print(f"  by nudging 0.001:  w1 {nudged_w[0]:+.4f}  w2 {nudged_w[1]:+.4f}  b {nudged_b:+.4f}")
    print(f"  by chain rule:     w1 {exact_w[0]:+.4f}  w2 {exact_w[1]:+.4f}  b {exact_b:+.4f}")
    print()

    print("Training on the one example, learning rate 1:")
    print(" step     w1      w2      b    prediction   loss")
    for step in range(11):
        prediction = predict(weights, bias, x)
        print(
            f" {step:>3}  {weights[0]:6.3f}  {weights[1]:6.3f}  {bias:6.3f}"
            f"    {prediction:.3f}     {loss(prediction, target):.4f}"
        )
        weights, bias = train_step(weights, bias, x, target, learning_rate=1.0)


if __name__ == "__main__":
    main()
