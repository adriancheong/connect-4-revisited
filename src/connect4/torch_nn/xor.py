"""Stage 3, step 1: the XOR network from stage 2, rebuilt in PyTorch.

Read this alongside docs/stage-3/01-numpy-to-pytorch.md.

Run `python -m connect4.torch_nn.xor` to print:
- the worked example from stage 2, step 3a, with PyTorch working out the slopes,
- the XOR training run from stage 2, step 3b, done by PyTorch from the same start,
- how long one training round takes on the processor and on the graphics card.

What PyTorch takes over from the stage 2 code:

    stage 2 (NumPy, by hand)                 stage 3 (PyTorch)
    forward(): multiply, add, squash         nn.Linear + nn.Sigmoid, stacked in nn.Sequential
    error(): average of (guess - answer)²    nn.functional.mse_loss
    slopes(): backpropagation by hand        error.backward()
    train_step(): dial -= step × slope       torch.optim.SGD
"""

from __future__ import annotations

import time

import torch
from torch import nn

from connect4.numpy_nn import network as np_nn

# NumPy works in float64 (about 16 significant digits). PyTorch's default is float32
# (about 7). This step uses float64 so its numbers can be checked against stage 2's
# exactly. Later steps use float32, which is plenty for training and much faster on
# graphics cards.
EXACT = torch.float64


def best_device() -> torch.device:
    """The graphics card if PyTorch can use one, otherwise the processor."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_network(inputs: int, hidden: int, outputs: int, dtype=EXACT) -> nn.Sequential:
    """The stage 2 network. Data flows through the four parts in order."""
    return nn.Sequential(
        nn.Linear(inputs, hidden),  # every input into every hidden neuron: the totals
        nn.Sigmoid(),  # squash
        nn.Linear(hidden, outputs),  # every hidden neuron into every output
        nn.Sigmoid(),  # squash: the guesses
    ).to(dtype)


def from_numpy(net: np_nn.Network) -> nn.Sequential:
    """A PyTorch copy of a stage 2 network, with exactly the same dials.

    One difference in bookkeeping: stage 2 kept one *column* per neuron, PyTorch keeps
    one *row* per neuron. So each weight grid is flipped (transposed, `.T`) on the way.
    """
    inputs, hidden = net.hidden_weights.shape
    model = make_network(inputs, hidden, net.output_weights.shape[1])
    with torch.no_grad():  # setting dials by hand, not learning: nothing to record
        for layer, weights, biases in (
            (model[0], net.hidden_weights, net.hidden_biases),
            (model[2], net.output_weights, net.output_biases),
        ):
            layer.weight.copy_(torch.from_numpy(weights.T))
            layer.bias.copy_(torch.from_numpy(biases))
    return model


def to_numpy(model: nn.Sequential, use_slopes: bool = False) -> np_nn.Network:
    """The dials of a PyTorch network (or their slopes), in stage 2's layout."""

    def grab(tensor):
        return (tensor.grad if use_slopes else tensor).detach().cpu().numpy()

    return np_nn.Network(
        grab(model[0].weight).T, grab(model[0].bias), grab(model[2].weight).T, grab(model[2].bias)
    )


def slopes(model: nn.Sequential, inputs, answers) -> np_nn.Network:
    """The slope of the average error for every dial, worked out by PyTorch.

    Stage 2's `slopes()` passed the blame backwards by hand. Here PyTorch keeps a record
    of every sum while the guess is made, then `backward()` runs the chain rule (the
    gear ratios) back through that record and stores each dial's slope in `.grad`.
    """
    model.zero_grad()  # clear any slopes left over from an earlier call
    guesses = model(torch.as_tensor(inputs, dtype=EXACT))
    error = nn.functional.mse_loss(guesses, torch.as_tensor(answers, dtype=EXACT))
    error.backward()
    return to_numpy(model, use_slopes=True)


def train(model: nn.Sequential, inputs, answers, learning_rate: float, rounds: int):
    """Stage 2's training loop in PyTorch: guess, measure the error, find every slope,
    turn every dial a step downhill. All cards every round, as in stage 2, step 3.

    Returns the average error before every round (plus one at the end).
    """
    x = torch.as_tensor(inputs, dtype=EXACT)
    y = torch.as_tensor(answers, dtype=EXACT).reshape(len(x), -1)  # one row per card
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # plain steps downhill
    history = []
    for _ in range(rounds):
        error = nn.functional.mse_loss(model(x), y)
        history.append(error.item())
        optimizer.zero_grad()  # forget last round's slopes
        error.backward()  # work out this round's
        optimizer.step()  # every dial -= step size × its slope
    with torch.no_grad():
        history.append(nn.functional.mse_loss(model(x), y).item())
    return history


def seconds_per_round(device: torch.device, cards: int, inputs: int, hidden: int) -> float:
    """Average time for one training round of a float32 network on `device`.

    The cards and answers are random: only the amount of arithmetic matters here.
    """
    gen = torch.Generator().manual_seed(0)
    x = torch.rand(cards, inputs, generator=gen).to(device)
    y = torch.rand(cards, 7, generator=gen).to(device)
    model = make_network(inputs, hidden, 7, dtype=torch.float32).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def one_round():
        error = nn.functional.mse_loss(model(x), y)
        optimizer.zero_grad()
        error.backward()
        optimizer.step()

    def wait():  # the graphics card works in the background; wait for it to finish
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(20):  # warm-up: the first rounds include one-off setup costs
        one_round()
    wait()
    rounds, started = 200, time.perf_counter()
    for _ in range(rounds):
        one_round()
    wait()
    return (time.perf_counter() - started) / rounds


TIMING_SIZES = [  # (description, cards per round, inputs, hidden neurons)
    ("XOR-sized: 4 cards, 2 inputs, 2 hidden", 4, 2, 2),
    ("step 4-sized: 64 cards, 84 inputs, 128 hidden", 64, 84, 128),
    ("big: 4,096 cards, 84 inputs, 4,096 hidden", 4096, 84, 4096),
]


def main() -> None:
    card, answer = np_nn.EXAMPLE_CARD, np_nn.EXAMPLE_ANSWER
    by_hand = np_nn.named_dials(np_nn.slopes(np_nn.EXAMPLE, card, answer))
    by_torch = np_nn.named_dials(slopes(from_numpy(np_nn.EXAMPLE), card, answer))
    print("The worked example from stage 2, step 3a (card (1, 0), right answer 1).")
    print("Slope of every dial, two ways:")
    print("  dial   stage 2 (by hand)   PyTorch (automatic)")
    for name in by_hand:  # (+ 0.0 turns PyTorch's "minus zero" into a plain zero)
        print(f"  {name:<4}  {by_hand[name] + 0.0:+.6f}           {by_torch[name] + 0.0:+.6f}")
    print()

    start = np_nn.new_network(2, 2, 1, seed=1)
    trained_by_hand, history_by_hand = np_nn.train(
        np_nn.XOR_INPUTS, np_nn.XOR_ANSWERS, learning_rate=2.0, start=start
    )  # fmt: skip
    model = from_numpy(start)
    history_by_torch = train(model, np_nn.XOR_INPUTS, np_nn.XOR_ANSWERS, 2.0, 10_000)
    gap = max(
        abs(a - b).max() for a, b in zip(trained_by_hand.dials(), to_numpy(model).dials())
    )  # fmt: skip
    print("XOR from the same starting dials (stage 2, step 3b: step size 2, 10,000 rounds):")
    print(f"  stage 2: average error {history_by_hand[0]:.4f} -> {history_by_hand[-1]:.6f}")
    print(f"  PyTorch: average error {history_by_torch[0]:.4f} -> {history_by_torch[-1]:.6f}")
    print(f"  biggest difference between the two sets of trained dials: {gap:.1e}")
    with torch.no_grad():
        guesses = model(torch.as_tensor(np_nn.XOR_INPUTS, dtype=EXACT)).numpy()
    for x, a, g in zip(np_nn.XOR_INPUTS, np_nn.XOR_ANSWERS, guesses):
        print(f"  card ({x[0]:.0f}, {x[1]:.0f})  answer {a:.0f}  PyTorch's guess {g[0]:.2f}")
    print()

    devices = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
    if len(devices) == 1:
        print("No graphics card that PyTorch can use, so no timing comparison.")
        return
    print(f"One training round (float32), processor vs {torch.cuda.get_device_name(0)}:")
    for name, cards, inputs, hidden in TIMING_SIZES:
        cpu, gpu = (seconds_per_round(d, cards, inputs, hidden) for d in devices)
        if gpu < cpu:
            faster = f"graphics card {cpu / gpu:.0f}x faster"
        else:
            faster = f"processor {gpu / cpu:.1f}x faster"
        print(f"  {name:<46} processor {cpu * 1e3:7.3f} ms   graphics card {gpu * 1e3:7.3f} ms"
              f"   ({faster})")  # fmt: skip


if __name__ == "__main__":
    main()
