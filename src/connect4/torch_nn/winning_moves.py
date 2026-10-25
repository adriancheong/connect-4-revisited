"""Stage 3, steps 2 and 3: training the Connect 4 winning-move networks in PyTorch.

Read this alongside:
- docs/stage-3/02a-connect-4-in-pytorch.md   (stage 2's network, rebuilt)
- docs/stage-3/02b-three-upgrades.md         (cross-entropy, ReLU and Adam)
- docs/stage-3/03b-training-a-cnn.md         (the convolutional network)

Train and save a network (a few minutes each):
    python -m connect4.torch_nn.winning_moves                     # the CNN (step 3)
    python -m connect4.torch_nn.winning_moves --recipe relu       # step 2b's best 84-128-7
    python -m connect4.torch_nn.winning_moves --recipe stage2     # stage 2's recipe

Then play against it:
    connect4 --p2 neural --model models/winning-moves-cnn.pt

The cards, the right answers and the scores are exactly stage 2's
(`connect4.numpy_nn.winning_moves`). Only the network and its training change.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from connect4.board import Board
from connect4.numpy_nn import network as nw
from connect4.numpy_nn import winning_moves as wm
from connect4.torch_nn.models import SWITCHES, ConnectFourCNN, make_mlp


@dataclass(frozen=True)
class Recipe:
    """Everything about how a network is built and trained."""

    net: str = "mlp"  # "mlp" (stage 2's shape) or "cnn" (step 3)
    squash: str = "sigmoid"  # the hidden neurons' squash: "sigmoid" or "relu" (mlp only)
    error: str = "squared"  # "squared" (stage 2) or "cross_entropy" (step 2b)
    optimizer: str = "sgd"  # "sgd" (stage 2's plain steps) or "adam" (step 2b)
    learning_rate: float = 10.0  # the step size
    epochs: int = 50
    batch_size: int = 64


# Step 2b adds one upgrade at a time. The SGD step sizes were chosen by trying 1, 3, 10
# and 30 for 10 epochs and keeping the best on the practice cards (see the notes). Adam
# uses its usual step size, 0.001, with no search.
RECIPES = {
    "stage2": Recipe(),
    "cross_entropy": Recipe(error="cross_entropy", learning_rate=10.0),
    "relu": Recipe(squash="relu", error="cross_entropy", learning_rate=10.0),
    "adam": Recipe(squash="relu", error="cross_entropy", optimizer="adam", learning_rate=1e-3),
    "cnn": Recipe(net="cnn", error="cross_entropy", optimizer="adam", learning_rate=1e-3),
}


def best_device_for(recipe: Recipe) -> torch.device:
    """Small networks train faster on the processor (step 1); the CNN on the graphics card."""
    if recipe.net == "cnn" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build(recipe: Recipe, seed: int = 0) -> nn.Module:
    """A new, untrained network.

    The 84-128-7 networks all start from exactly the same dials as stage 2's network
    (`nw.new_network(84, 128, 7, seed)`), so each upgrade in step 2b changes one thing.
    """
    if recipe.net == "cnn":
        torch.manual_seed(seed)
        return ConnectFourCNN()
    model = make_mlp(128, recipe.squash)
    start = nw.new_network(SWITCHES, 128, 7, seed)
    with torch.no_grad():  # PyTorch keeps one row per neuron; stage 2 one column (step 1)
        model[0].weight.copy_(torch.from_numpy(start.hidden_weights.T))
        model[0].bias.copy_(torch.from_numpy(start.hidden_biases))
        model[2].weight.copy_(torch.from_numpy(start.output_weights.T))
        model[2].bias.copy_(torch.from_numpy(start.output_biases))
    return model


def error_measure(recipe: Recipe):
    """The error, worked out from the network's totals and the right answers (0 or 1)."""
    if recipe.error == "cross_entropy":
        return nn.functional.binary_cross_entropy_with_logits  # step 2b
    return lambda totals, answers: nn.functional.mse_loss(torch.sigmoid(totals), answers)


def make_optimizer(recipe: Recipe, model: nn.Module) -> torch.optim.Optimizer:
    if recipe.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=recipe.learning_rate)
    return torch.optim.SGD(model.parameters(), lr=recipe.learning_rate)


def guesses(model: nn.Module, inputs: np.ndarray, device=None) -> np.ndarray:
    """The network's guesses (0 to 1) for every card, as a NumPy array."""
    device = device or next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(inputs), 8192):
            x = torch.as_tensor(inputs[start : start + 8192], dtype=torch.float32, device=device)
            out.append(torch.sigmoid(model(x)).cpu().numpy())
    model.train()
    return np.concatenate(out)


def evaluate(model: nn.Module, inputs: np.ndarray, answers: np.ndarray) -> wm.Scores:
    """Stage 2's scores: finds the win, false alarms, all right, and the squared error."""
    return wm.score(guesses(model, inputs), answers)


def train(
    recipe: Recipe, data: wm.Dataset, device=None, seed: int = 0, report=None, practice=True
) -> tuple[nn.Module, list[wm.EpochRecord]]:
    """Train a new network on the practice cards in mini-batches, as in stage 2.

    The deck is shuffled with the same random numbers as stage 2, so with stage 2's
    recipe the cards come in exactly the same order. After every epoch the network is
    scored on the test cards (and on the practice cards, unless `practice` is False).
    """
    device = device or best_device_for(recipe)
    model = build(recipe, seed).to(device)
    error = error_measure(recipe)
    optimizer = make_optimizer(recipe, model)
    x = torch.as_tensor(data.practice_inputs, dtype=torch.float32, device=device)
    y = torch.as_tensor(data.practice_answers, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, recipe.epochs + 1):
        order = torch.as_tensor(rng.permutation(len(x)), device=device)  # shuffle the deck
        for start in range(0, len(x), recipe.batch_size):
            batch = order[start : start + recipe.batch_size]
            loss = error(model(x[batch]), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        on_practice = (
            evaluate(model, data.practice_inputs, data.practice_answers) if practice else None
        )  # fmt: skip
        record = wm.EpochRecord(
            epoch, on_practice, evaluate(model, data.test_inputs, data.test_answers)
        )  # fmt: skip
        history.append(record)
        if report is not None:
            report(record)
    return model, history


# ---------------------------------------------------------------------------
# Playing with it, saving and loading
# ---------------------------------------------------------------------------


class TorchNeuralPlayer(wm.NeuralPlayer):
    """Stage 2's neural player (win, else block, else random), seeing through a PyTorch network."""

    def guesses(self, board: Board) -> tuple[np.ndarray, np.ndarray]:
        x = wm.encode(board)
        both = guesses(self.net, np.array([x, wm.swap_sides(x)]))
        return both[0], both[1]


def save(model: nn.Module, recipe: Recipe, path: Path) -> None:
    """Save the recipe and every dial. The dials are all there is to a trained network."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"recipe": asdict(recipe), "dials": model.state_dict()}, path)


def load(path: Path, device=None) -> tuple[nn.Module, Recipe]:
    # weights_only=True: read numbers and plain settings only, never run code from the file
    saved = torch.load(path, map_location="cpu", weights_only=True)
    recipe = Recipe(**saved["recipe"])
    model = build(recipe)
    model.load_state_dict(saved["dials"])
    return model.to(device or best_device_for(recipe)), recipe


def load_player(path: Path, seed: int | None = None) -> TorchNeuralPlayer:
    model, _ = load(path)
    return TorchNeuralPlayer(model, seed)


def default_model_path(recipe_name: str) -> Path:
    return Path("models") / f"winning-moves-{recipe_name}.pt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a winning-move network in PyTorch.")
    parser.add_argument("--recipe", choices=RECIPES, default="cnn", help="what to train")
    parser.add_argument("--epochs", type=int, help="passes through the data (default 50)")
    parser.add_argument("--device", choices=("cpu", "cuda"), help="where to train")
    parser.add_argument("--out", type=Path, help="where to save it")
    args = parser.parse_args(argv)

    recipe = RECIPES[args.recipe]
    if args.epochs:
        recipe = Recipe(**{**asdict(recipe), "epochs": args.epochs})
    device = torch.device(args.device) if args.device else best_device_for(recipe)
    out = args.out or default_model_path(args.recipe)

    started = time.time()
    print("Making 300,000 practice and 30,000 test positions from random games...")
    data = wm.make_dataset()
    print(f"Training '{args.recipe}' on the {device.type} for {recipe.epochs} epochs: {recipe}")

    def report(record):
        t = record.test
        print(f"  epoch {record.epoch:>3}: test error {t.error:.4f} | finds the win {t.found:6.1%}"
              f" | false alarms {t.false_alarms:5.1%}", flush=True)  # fmt: skip

    model, history = train(recipe, data, device, report=report, practice=False)
    save(model, recipe, out)
    final = history[-1].test
    print(f"Done in {time.time() - started:.0f}s. On test positions it finds the win "
          f"{final.found:.1%} of the time, with {final.false_alarms:.1%} false alarms.")
    print(f"Saved to {out}. Play against it with: connect4 --p2 neural --model {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
