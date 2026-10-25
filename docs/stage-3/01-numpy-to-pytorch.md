# Stage 3, step 1: From NumPy to PyTorch

**Read first:** [Stage 2, step 3a: A network with a hidden layer, by hand](../stage-2/03a-hidden-layer-by-hand.md) and [step 3b: Training a network on XOR](../stage-2/03b-training-a-network.md).

**Goal:** rebuild the stage 2 XOR network in PyTorch, check that it does exactly what you built by hand, learn the handful of new words, and find out when your graphics card actually helps.

**Code:** [`src/connect4/torch_nn/xor.py`](../../src/connect4/torch_nn/xor.py). Run `python -m connect4.torch_nn.xor` to print everything on this page.

---

## 1. Why PyTorch?

In stage 2 you wrote four things by hand:

| Job | Stage 2 (NumPy, by hand) | PyTorch |
|---|---|---|
| Make a guess | `forward()`: multiply, add up, add the bias, squash | `nn.Linear` and `nn.Sigmoid`, stacked in `nn.Sequential` |
| Measure the error | `error()`: the average of (guess − answer)² | `nn.functional.mse_loss` |
| Find every dial's slope | `slopes()`: backpropagation, worked out by hand on a whole page (step 3a) | `error.backward()` |
| Turn every dial | dial − step size × slope | `torch.optim.SGD` |

For one hidden layer, writing the slopes by hand was manageable. For the networks in this stage, with more layers and a convolutional network in step 3, it would be long and easy to get wrong. PyTorch is a library that does all four jobs for any network you build out of its parts, and can run them on a graphics card. It's one of the most widely used tools for neural networks, in industry and in research.

This step checks that it does nothing magic: given the same dials, it gives the same numbers as your stage 2 code.

## 2. Tensors: arrays that can live on the graphics card

A **tensor** is PyTorch's version of a NumPy array: a grid of numbers.

```python
import torch
card = torch.tensor([[1.0, 0.0]])  # one card, two inputs: the same as np.array([[1.0, 0.0]])
```

It adds two things a NumPy array can't do:
- **It can live on the graphics card.** `card.to("cuda")` copies it into the graphics card's own memory, and any arithmetic on it then runs there. (`cuda` is NVIDIA's name for running general maths on their graphics cards.) Section 6 shows when that pays off.
- **It can remember how it was calculated,** so PyTorch can work out slopes afterwards. Section 4.

**Precision.** NumPy normally stores each number in 64 bits (*float64*), about 16 significant digits. PyTorch normally uses 32 bits (*float32*), about 7. This step uses float64 so its numbers can be checked against stage 2's to the last digit. Later steps use float32: it's plenty for training, and graphics cards are much faster at it.

## 3. Building the network out of parts

```python
def make_network(inputs, hidden, outputs):
    return nn.Sequential(
        nn.Linear(inputs, hidden),   # every input into every hidden neuron: the totals
        nn.Sigmoid(),                # squash
        nn.Linear(hidden, outputs),  # every hidden neuron into every output
        nn.Sigmoid(),                # squash: the guesses
    )
```

- **`nn.Linear(inputs, hidden)`** is a whole layer's worth of weights and biases. For each neuron, it multiplies every input by its weight, adds them up and adds the bias: the neuron's *total* from stage 2. (Textbooks call this a *linear* or *fully connected* layer.)
- **`nn.Sigmoid()`** is the squash from stage 2.
- **`nn.Sequential`** runs the parts in order, so a card goes in at the top and guesses come out at the bottom.

`make_network(2, 2, 1)` is exactly the XOR network from step 3: 2 inputs, 2 hidden neurons, 1 output, 9 dials.

**One bookkeeping difference.** Stage 2 stored each layer's weights with one *column* per neuron. PyTorch stores one *row* per neuron. So `from_numpy()`, which copies a stage 2 network's dials into a PyTorch one, flips each weight grid on the way (`.T`, short for *transpose*). The numbers are the same; only the layout differs.

## 4. Automatic slopes

```python
def slopes(model, inputs, answers):
    model.zero_grad()                                # clear any slopes left from before
    guesses = model(torch.as_tensor(inputs))         # make the guesses
    error = nn.functional.mse_loss(guesses, answers) # average of (guess - answer)²
    error.backward()                                 # work out every dial's slope
    ...                                              # each slope is now in dial.grad
```

While the guess is made, PyTorch keeps a record of every step: this weight times that input, plus that bias, squashed. `error.backward()` then walks back through the record and applies the gear ratios, the chain rule, at every step: exactly what you did by hand in step 3a. Each dial's slope ends up next to it, in `.grad`. (Textbooks call the record a *computation graph*, and this way of finding slopes *automatic differentiation*, or *autograd*.)

Here is the worked example from step 3a: the same dials, the card (1, 0), right answer 1.

| Dial | Stage 2 (by hand, step 3a) | PyTorch (automatic) |
|---|---|---|
| wA1 | −0.025637 | −0.025637 |
| wA2 | 0.000000 | 0.000000 |
| bA | −0.025637 | −0.025637 |
| wB1 | +0.015846 | +0.015846 |
| wB2 | 0.000000 | 0.000000 |
| bB | +0.015846 | +0.015846 |
| wOA | −0.127763 | −0.127763 |
| wOB | −0.096068 | −0.096068 |
| bO | −0.213406 | −0.213406 |

Identical. The tests check this on random networks of other sizes too, and the two agree to at least 12 decimal places.

## 5. Training: the same run, step for step

```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
for _ in range(rounds):
    error = nn.functional.mse_loss(model(x), y)  # guess and measure
    optimizer.zero_grad()                        # forget last round's slopes
    error.backward()                             # work out this round's
    optimizer.step()                             # every dial -= step size × its slope
```

- **`model.parameters()`** hands the optimizer every dial in the network.
- **`torch.optim.SGD`** is the plain rule from stage 2: dial − step size × slope. (*SGD* stands for *stochastic gradient descent*: gradient descent, done on a random handful of cards at a time. Here it gets all four cards every round.) Step 2 will swap in a smarter rule.
- **`optimizer.zero_grad()`** matters. PyTorch *adds* new slopes to whatever is already in `.grad`. Forget to clear them and each round's slopes pile on top of the last round's.

Starting from the same dials as the stage 2 run in step 3b (seed 1, step size 2, 10,000 rounds):

| | Average error at the start | After 10,000 rounds |
|---|---|---|
| Stage 2 (NumPy) | 0.2651 | 0.0001 |
| PyTorch | 0.2651 | 0.0001 |

The biggest difference between any two of the trained dials is 0.0000000000000018: rounding in the last digits, because the two libraries add things up in slightly different orders. Both networks give the same guesses: 0.01, 0.99, 0.99 and 0.01 for the four cards.

So PyTorch is the same arithmetic you did by hand, done for you.

## 6. Processor or graphics card?

In PyTorch, where the work happens is called the **device**: `cpu` for the processor, `cuda` for an NVIDIA graphics card. Move the network and the cards there with `.to(device)`, and everything after that runs there.

The graphics card doesn't do anything the processor can't; it does the same arithmetic on thousands of numbers at once. Whether that's faster depends on how much arithmetic there is. Here is one training round (guess, error, slopes, turn the dials) at three sizes, in float32, on the processor and the graphics card:

| One training round | Processor | Graphics card | Faster |
|---|---|---|---|
| XOR-sized: 4 cards, 2 inputs, 2 hidden neurons | 0.18 ms | 0.61 ms | processor, 3.4× |
| Stage 2, step 4-sized: 64 cards, 84 inputs, 128 hidden neurons | 0.26 ms | 0.69 ms | processor, 2.7× |
| Big: 4,096 cards, 84 inputs, 4,096 hidden neurons | 47.9 ms | 2.5 ms | **graphics card, 19×** |

(Measured on a Ryzen 7 7700 processor and an RTX 5060 Ti graphics card. The numbers vary a little from run to run, and more from PC to PC.)

- **Small networks: the processor wins.** Handing work to the graphics card has a fixed cost every round, about half a millisecond here, however little arithmetic there is. In a small round that cost is far bigger than the arithmetic itself. So the stage 2 Connect 4 network trains faster on the processor: an epoch's 4,688 rounds take about 1.2 seconds there, against about 3.2 on the graphics card.
- **Big networks: the graphics card wins easily.** The big round has about 2,000 times as much arithmetic as the step 4-sized one. The processor takes about 190 times as long; the graphics card only about 4 times as long, because it works on thousands of numbers at once.
- **For this course:** step 2's network is the stage 2 size, so expect the processor to be the quicker one there. The graphics card earns its keep in step 3's convolutional network, and in stages 4 and 5.

## Summary

- **PyTorch does the four jobs you wrote by hand in stage 2:** guess, measure the error, find every slope, turn the dials.
- **It's the same arithmetic.** Same dials in, same slopes and the same training run out, to 12 decimal places or better.
- **Tensors** are arrays that can live on the graphics card and remember how they were calculated. **`error.backward()`** uses that record to find every slope: backpropagation, done automatically.
- **The graphics card** only pays off when each step has plenty of arithmetic in it. On this PC the processor is about 3 times faster for stage 2-sized networks, and the graphics card about 19 times faster for a big one.

## New words in this step

| Word | Meaning |
|---|---|
| **PyTorch** | A library that builds networks out of parts, works out slopes automatically and can run on a graphics card. |
| **Tensor** | PyTorch's grid of numbers, like a NumPy array, that can live on the graphics card and remember how it was calculated. |
| **Device** | Where the arithmetic happens: `cpu` (the processor) or `cuda` (an NVIDIA graphics card). |
| **float32 / float64** | Numbers stored in 32 bits (about 7 significant digits) or 64 bits (about 16). Training uses float32. |
| **Linear layer** (`nn.Linear`) | One layer's weights and biases: every input into every neuron, giving each neuron's total. |
| **Autograd** (automatic differentiation) | PyTorch's automatic backpropagation: it records how the guess was made, then `backward()` works out every slope. |
| **Optimizer** | The rule for turning the dials. `SGD` is stage 2's plain rule; step 2 meets a smarter one. |

## Try it yourself

1. **Change a dial.** In Python, copy `EXAMPLE` from `connect4.numpy_nn.network` with one weight changed (for example `wOA` from 0.5 to 2.0). Do PyTorch's slopes still match stage 2's? Which slopes change most?
2. **Forget to clear the slopes.** Copy `train()` from `xor.py`, delete `optimizer.zero_grad()`, and train XOR again. What happens to the error, and why?
3. **Find the crossover.** In `TIMING_SIZES`, add sizes between "step 4-sized" and "big". Roughly where does the graphics card start to win on your PC?

---

**Previous:** [Stage 2, step 4b: Training on Connect 4](../stage-2/04b-training-on-connect-4.md) · **Next:** Step 2: the Connect 4 network in PyTorch (coming)
