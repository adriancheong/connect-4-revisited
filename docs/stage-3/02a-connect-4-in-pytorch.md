# Stage 3, step 2a: The Connect 4 network in PyTorch

**Read first:** [Step 1: From NumPy to PyTorch](01-numpy-to-pytorch.md), and [stage 2, step 4b: Training on Connect 4](../stage-2/04b-training-on-connect-4.md).

**Goal:** rebuild stage 2's winning-move network (84 → 128 → 7) in PyTorch, train it with exactly stage 2's recipe, check it learns the same, then save it and play against it. This is the starting line for the upgrades in step 2b.

**Code:** [`src/connect4/torch_nn/models.py`](../../src/connect4/torch_nn/models.py) (the network) and [`src/connect4/torch_nn/winning_moves.py`](../../src/connect4/torch_nn/winning_moves.py) (training, saving, playing). The numbers come from [`scripts/make_stage3_charts.py`](../../scripts/make_stage3_charts.py).

---

## 1. What stays the same

Only the network and its training move to PyTorch. Everything else is reused from stage 2, unchanged:

- **The cards:** the same 300,000 practice and 30,000 test positions, made by `make_dataset()`.
- **The switches:** the same 84 per board, "mine" and "opponent's", from `encode()`.
- **The right answers and the scores:** "finds the win", "false alarms", "all right" and the error, from `score()`.
- **The player's strategy:** win if the network sees a win, otherwise block, otherwise random.

So every number in this stage can be compared directly with stage 2's.

## 2. The network

```python
def make_mlp(hidden=128, squash="sigmoid"):
    return nn.Sequential(
        nn.Linear(84, hidden),                                  # 84 switches -> 128 totals
        nn.Sigmoid() if squash == "sigmoid" else nn.ReLU(),     # squash (step 2b tries ReLU)
        nn.Linear(hidden, 7),                                   # 128 outputs -> 7 totals
    )
```

The same shape as stage 2, with the same 11,783 dials. One difference from step 1: **the last squash isn't part of the network.** It gives out the 7 totals, and `guesses()` squashes them into guesses between 0 and 1. That's because the error measure in step 2b, cross-entropy, works on the totals directly. For stage 2's recipe it makes no difference: the squash is applied just before measuring the error instead.

(Textbooks call this shape a *multilayer perceptron*, or **MLP**: layers in which every neuron gets every input. In the code, `mlp` is its short name, to tell it apart from step 3's convolutional network.)

## 3. A recipe: everything about how a network is trained

```python
@dataclass(frozen=True)
class Recipe:
    net: str = "mlp"            # the network's shape: "mlp" or "cnn" (step 3)
    squash: str = "sigmoid"     # the hidden neurons' squash
    error: str = "squared"      # how the error is measured
    optimizer: str = "sgd"      # the rule for turning the dials
    learning_rate: float = 10.0 # the step size
    epochs: int = 50
    batch_size: int = 64
```

A **recipe** gathers every setting in one place: the choices stage 2, step 4b section 12 called *hyperparameters*. The defaults are stage 2's recipe exactly. Step 2b changes one line at a time.

Two more things make this a fair rerun of stage 2:
- **The same starting dials.** `build()` copies them from stage 2's `new_network(84, 128, 7, seed=0)`.
- **The same order of cards.** The deck is shuffled with the same random numbers as stage 2, so the mini-batches are the same 64 cards in the same order.

## 4. Training in mini-batches

```python
x = torch.as_tensor(data.practice_inputs, dtype=torch.float32, device=device)
y = torch.as_tensor(data.practice_answers, dtype=torch.float32, device=device)
for epoch in range(recipe.epochs):
    order = torch.as_tensor(rng.permutation(len(x)), device=device)  # shuffle the deck
    for start in range(0, len(x), recipe.batch_size):
        batch = order[start : start + recipe.batch_size]   # the next 64 cards
        loss = error(model(x[batch]), y[batch])            # guess and measure
        optimizer.zero_grad()
        loss.backward()                                    # every slope
        optimizer.step()                                   # turn every dial
```

It's stage 2's `train_in_batches()` with step 1's three-line PyTorch training round inside. The whole deck is turned into tensors once, at the start; each mini-batch is then just 64 rows picked out of it. (*Loss* is the textbook word for the error, and the usual name for it in PyTorch code.)

## 5. Does it learn the same?

Stage 2's recipe, same start, same cards in the same order, trained on the processor. "Finds the win" on the 30,000 test positions:

| Epoch | Stage 2 (NumPy) | PyTorch |
|---|---|---|
| 1 | 15.7% | 15.7% |
| 2 | 36.5% | 36.5% |
| 5 | 68.2% | 68.2% |
| 10 | 76.9% | 76.9% |
| 20 | 84.7% | 84.7% |
| 50 | **92.9%** | **92.9%** |

The same, all the way. The other scores after 50 epochs match too: 0.9% false alarms, 95.0% all right, and 94.8% on the practice cards.

That's slightly lucky. PyTorch works in float32 here, about 7 significant digits, while stage 2 used float64, about 16 (step 1). So the two runs round differently in the last few digits, and over 234,000 dial turns those differences could have grown into a visible gap. Here they never got big enough to change a score's first decimal place.

**Processor or graphics card?** One epoch takes **1.4 seconds** on the processor and **2.8** on the graphics card. That's what step 1 predicted: with only 64 cards per round and 11,783 dials, handing each round to the graphics card costs more than the arithmetic saves. So the 84 → 128 → 7 networks in this step and the next train on the processor.

## 6. Saving it

```python
torch.save({"recipe": asdict(recipe), "dials": model.state_dict()}, path)
```

`model.state_dict()` is every dial in the network, by name. Saved together with its recipe, that's all there is to a trained network: loading rebuilds the network from the recipe, then fills in the dials. PyTorch's files usually end in `.pt`.

**Loading safely.** `torch.load` can, in general, run code stored in the file, so a `.pt` file from a stranger could do anything a program can. `load()` uses `torch.load(path, weights_only=True)`, which only reads numbers and plain settings and refuses anything else. Treat `.pt` files you didn't make like programs, and only load them this way.

## 7. Playing against it

Train and save a network with stage 2's recipe, then play it:

```bash
python -m connect4.torch_nn.winning_moves --recipe stage2      # saves models/winning-moves-stage2.pt
connect4 --p2 neural --model models/winning-moves-stage2.pt
```

The game sees the `.pt` at the end of the file name and loads a PyTorch network instead of a stage 2 one.

1,000 games each, with the PyTorch network playing stage 2's strategy:

| Player 1 (X) | Player 2 (O) | Neural wins | Opponent wins | Draws |
|---|---|---|---|---|
| neural | random | **93.2%** | 6.2% | 0.6% |
| random | neural | **90.2%** | 9.5% | 0.3% |
| neural | heuristic | 36.5% | **62.1%** | 1.4% |
| heuristic | neural | 30.3% | **68.2%** | 1.5% |

Close to stage 2's results (section 11 of step 4b), but not identical. The guesses differ in the last digits, and in a game one slightly different guess on one move can change every move after it. It handles 94.5% of "win or block" moments correctly (stage 2: 94.1%), and still loses to the heuristic bot for the same reason as before: small error rates add up over a game.

## Summary

- **Only the network and its training moved to PyTorch.** The cards, answers, scores and the player's strategy are stage 2's, so the results compare directly.
- **A recipe** holds every setting in one place. Stage 2's recipe, from stage 2's starting dials with the cards in the same order, gives the same results at every epoch: **92.9%** after 50.
- **The network gives out totals,** and the squash is added afterwards, ready for step 2b's error measure.
- **Small networks train faster on the processor** (1.4 s per epoch against 2.8 s), as step 1 predicted.
- **Save the dials with their recipe,** and load with `weights_only=True`, so a file can never run code.

## New words in this step

| Word | Meaning |
|---|---|
| **MLP** (multilayer perceptron) | A network made of layers where every neuron gets every input, like stage 2's. |
| **Recipe** | Every setting for building and training a network, in one place. Its parts are the *hyperparameters*. |
| **Loss** | The textbook word for the error, and its usual name in PyTorch code. |
| **`state_dict`** | Every dial in a PyTorch network, by name: what gets saved. |
| **`.pt` file** | A saved PyTorch network. Load with `weights_only=True`, so it can only contain numbers and settings. |

## Try it yourself

1. **Time it yourself.** Train stage 2's recipe on each device and compare the time per epoch:
   ```bash
   python -m connect4.torch_nn.winning_moves --recipe stage2 --epochs 3 --device cpu
   python -m connect4.torch_nn.winning_moves --recipe stage2 --epochs 3 --device cuda
   ```
2. **File size.** After training, compare `models/winning-moves-stage2.pt` with stage 2's `models/winning-moves.npz` (about 93 KB). Guess first: bigger or smaller, and why? (Hint: float32.)
3. **Change one thing.** In Python, train stage 2's recipe for only 10 epochs, using `replace()` to change one setting of a recipe:
   ```python
   from dataclasses import replace
   from connect4.numpy_nn import winning_moves as wm
   from connect4.torch_nn import winning_moves as tw
   data = wm.make_dataset()
   model, history = tw.train(replace(tw.RECIPES["stage2"], epochs=10), data)
   print(history[-1].test)
   ```
   Guess first: section 5 says what it should print. Then try a different `learning_rate`, such as 3 or 30.

---

**Previous:** [Step 1: From NumPy to PyTorch](01-numpy-to-pytorch.md) · **Next:** [Step 2b: Three standard upgrades](02b-three-upgrades.md)
