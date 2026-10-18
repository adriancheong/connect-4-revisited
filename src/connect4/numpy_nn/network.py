"""Stage 2, step 3: a network with one hidden layer, trained with backpropagation.

Read this alongside:
- docs/stage-2/03a-hidden-layer-by-hand.md   (one guess and one round of blame, by hand)
- docs/stage-2/03b-training-a-network.md     (training it on XOR)

Run `python -m connect4.numpy_nn.network` to print the worked example from 03a
and a full training run on XOR.

The network:

    inputs ──► hidden layer ──► output layer ──► guesses

Every hidden neuron gets every input. Every output neuron gets every hidden
neuron's output. Each neuron works exactly like the single neuron in neuron.py:
multiply, add up, add its bias, squash.

How the dials are stored. Instead of naming every dial separately, each layer keeps
its weights in a grid (a "matrix"), with one column per neuron:

    hidden_weights[i, j]  = the dial from input i     into hidden neuron j
    output_weights[j, k]  = the dial from hidden neuron j into output neuron k

So for XOR, with 2 inputs, 2 hidden neurons (A and B) and 1 output:

    hidden_weights = [[wA1, wB1],      hidden_biases = [bA, bB]
                      [wA2, wB2]]

    output_weights = [[wOA],           output_biases = [bO]
                      [wOB]]

Many cards at once. `inputs` holds one card per row, so the whole deck goes
through the network in one go. `inputs @ hidden_weights` (the `@` is matrix
multiplication) works out every hidden neuron's weighted sum for every card at
once: it is just many `w1*x1 + w2*x2` sums done together.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from connect4.numpy_nn.neuron import sigmoid


@dataclass(frozen=True)
class Network:
    """All the dials of a network with one hidden layer."""

    hidden_weights: np.ndarray  # shape (inputs, hidden)
    hidden_biases: np.ndarray  # shape (hidden,)
    output_weights: np.ndarray  # shape (hidden, outputs)
    output_biases: np.ndarray  # shape (outputs,)

    def dials(self) -> list[np.ndarray]:
        return [self.hidden_weights, self.hidden_biases, self.output_weights, self.output_biases]

    def dial_count(self) -> int:
        return sum(d.size for d in self.dials())


def new_network(inputs: int, hidden: int, outputs: int, seed: int = 0) -> Network:
    """A network with small random weights and biases of 0.

    Random, so no two hidden neurons start as identical twins.
    Small, so every neuron starts on the steep middle part of the squash. The more
    inputs a neuron has, the smaller its starting weights, so its total doesn't
    start out huge.
    """
    rng = np.random.default_rng(seed)
    return Network(
        hidden_weights=rng.normal(0, 1 / np.sqrt(inputs), size=(inputs, hidden)),
        hidden_biases=np.zeros(hidden),
        output_weights=rng.normal(0, 1 / np.sqrt(hidden), size=(hidden, outputs)),
        output_biases=np.zeros(outputs),
    )


# ---------------------------------------------------------------------------
# Forward: make the guesses, remembering the numbers along the way
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Forward:
    hidden: np.ndarray  # each hidden neuron's output, for each card: shape (cards, hidden)
    guesses: np.ndarray  # each output neuron's guess, for each card: shape (cards, outputs)


def forward(net: Network, inputs: np.ndarray) -> Forward:
    """Run every card through the network. `inputs` has one card per row."""
    hidden = sigmoid(inputs @ net.hidden_weights + net.hidden_biases)
    guesses = sigmoid(hidden @ net.output_weights + net.output_biases)
    return Forward(hidden, guesses)


def predict(net: Network, inputs: np.ndarray) -> np.ndarray:
    """Just the guesses."""
    return forward(net, inputs).guesses


def error(net: Network, inputs: np.ndarray, answers: np.ndarray) -> float:
    """Average error over every card and every output: (guess - answer) squared."""
    return float(np.mean((predict(net, inputs) - answers) ** 2))


# ---------------------------------------------------------------------------
# Backward: backpropagation, the slope of every dial in one pass
# ---------------------------------------------------------------------------


def slopes(net: Network, inputs: np.ndarray, answers: np.ndarray) -> Network:
    """The slope of the average error for every dial, found by backpropagation.

    Returns a Network-shaped bundle: each number is the slope for the dial in
    the same place.

    It is the gear-ratio chain from neuron.py, carried back one more layer:

        hidden dial -> hidden total -> hidden output -> output total -> guess -> error

    The output layer is handled exactly like the single neuron. Then the blame
    is passed back to each hidden neuron along the weight that connects it to the
    output: that weight is the gear ratio from "hidden output" to "output total".
    """
    fwd = forward(net, inputs)
    hidden, guesses = fwd.hidden, fwd.guesses
    count = guesses.size  # the error is averaged over every card and output

    # Output layer: the same three ratios as the single neuron.
    error_per_guess = 2 * (guesses - answers) / count
    error_per_output_total = error_per_guess * guesses * (1 - guesses)
    output_weight_slopes = hidden.T @ error_per_output_total  # last ratio: each dial's input
    output_bias_slopes = error_per_output_total.sum(axis=0)

    # Pass the blame back to the hidden neurons, along the connecting weights.
    error_per_hidden_output = error_per_output_total @ net.output_weights.T
    error_per_hidden_total = error_per_hidden_output * hidden * (1 - hidden)
    hidden_weight_slopes = inputs.T @ error_per_hidden_total
    hidden_bias_slopes = error_per_hidden_total.sum(axis=0)

    return Network(
        hidden_weight_slopes, hidden_bias_slopes, output_weight_slopes, output_bias_slopes
    )


def slopes_by_nudging(
    net: Network, inputs: np.ndarray, answers: np.ndarray, nudge: float = 1e-5
) -> Network:
    """The same slopes found by experiment, as a check that backpropagation is right.

    Each dial is nudged up and down by a hair, and the change in error is divided by
    the distance between the two nudges. Nudging both ways is a little more accurate
    than nudging one way. It needs two full guesses per dial, so it is far too slow
    for training, but it is a reliable check.
    """
    result = []
    for which, dial in enumerate(net.dials()):
        slope = np.zeros_like(dial)
        for index in np.ndindex(dial.shape):
            errors = []
            for direction in (+1, -1):
                changed = [d.copy() for d in net.dials()]
                changed[which][index] += direction * nudge
                errors.append(error(Network(*changed), inputs, answers))
            slope[index] = (errors[0] - errors[1]) / (2 * nudge)
        result.append(slope)
    return Network(*result)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_step(net: Network, inputs: np.ndarray, answers: np.ndarray, learning_rate: float):
    """One round: work out every dial's slope, then move every dial a step downhill."""
    s = slopes(net, inputs, answers)
    return Network(*(dial - learning_rate * slope for dial, slope in zip(net.dials(), s.dials())))


def train(
    inputs: np.ndarray,
    answers: np.ndarray,
    hidden: int = 2,
    learning_rate: float = 2.0,
    rounds: int = 10_000,
    seed: int = 0,
    start: Network | None = None,
) -> tuple[Network, list[float]]:
    """Train a network on a deck of cards. Returns the trained network and the
    average error before every round (plus one at the end)."""
    answers = answers.reshape(len(inputs), -1)  # one row per card, one column per output
    net = start or new_network(inputs.shape[1], hidden, answers.shape[1], seed)
    history = []
    for _ in range(rounds):
        history.append(error(net, inputs, answers))
        net = train_step(net, inputs, answers, learning_rate)
    history.append(error(net, inputs, answers))
    return net, history


# ---------------------------------------------------------------------------
# The worked example from docs/stage-2/03a-hidden-layer-by-hand.md
# ---------------------------------------------------------------------------

# Hand-picked starting dials, so the arithmetic is easy to follow.
EXAMPLE = Network(
    hidden_weights=np.array([[0.3, -0.2], [-0.1, 0.4]]),  # [[wA1, wB1], [wA2, wB2]]
    hidden_biases=np.array([0.1, 0.0]),  # [bA, bB]
    output_weights=np.array([[0.5], [-0.3]]),  # [[wOA], [wOB]]
    output_biases=np.array([0.1]),  # [bO]
)
EXAMPLE_CARD = np.array([[1.0, 0.0]])  # inputs x1 = 1, x2 = 0
EXAMPLE_ANSWER = np.array([[1.0]])  # XOR of (1, 0) is 1

XOR_INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
XOR_ANSWERS = np.array([0, 1, 1, 0], dtype=float)


def named_dials(net: Network) -> dict[str, float]:
    """The nine dials of a 2-2-1 network, by name."""
    hw, hb, ow, ob = net.dials()
    return {
        "wA1": hw[0, 0], "wA2": hw[1, 0], "bA": hb[0],
        "wB1": hw[0, 1], "wB2": hw[1, 1], "bB": hb[1],
        "wOA": ow[0, 0], "wOB": ow[1, 0], "bO": ob[0],
    }  # fmt: skip


def main() -> None:
    net, card, answer = EXAMPLE, EXAMPLE_CARD, EXAMPLE_ANSWER
    fwd = forward(net, card)
    out_a, out_b = fwd.hidden[0]
    guess = fwd.guesses[0, 0]
    print("Forward pass for the card (1, 0), right answer 1:")
    print(f"  hidden A output {out_a:.4f}   hidden B output {out_b:.4f}")
    print(f"  guess {guess:.4f}   error {(guess - 1) ** 2:.4f}")
    print()

    exact = named_dials(slopes(net, card, answer))
    nudged = named_dials(slopes_by_nudging(net, card, answer))
    print("Slope of every dial, two ways:")
    print("  dial   backpropagation   nudging")
    for name in exact:
        print(f"  {name:<4}  {exact[name]:+.6f}        {nudged[name]:+.6f}")
    print()

    trained, history = train(XOR_INPUTS, XOR_ANSWERS, hidden=2, learning_rate=2.0, seed=1)
    print("Training on all four XOR cards (2 hidden neurons, step size 2, 10,000 rounds, seed 1):")
    print(f"  average error: start {history[0]:.4f}, end {history[-1]:.4f}")
    result = forward(trained, XOR_INPUTS)
    print("  card    answer   hidden A   hidden B   guess")
    for x, a, h, g in zip(XOR_INPUTS, XOR_ANSWERS, result.hidden, result.guesses):
        card_text = f"({x[0]:.0f}, {x[1]:.0f})"
        print(f"  {card_text}    {a:.0f}       {h[0]:.2f}       {h[1]:.2f}      {g[0]:.2f}")


if __name__ == "__main__":
    main()
