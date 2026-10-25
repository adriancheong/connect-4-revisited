# Stage 3, step 3b: Training a CNN on Connect 4

**Read first:** [Step 3a: How a convolution works, by hand](03a-convolution-by-hand.md).

**Goal:** train the convolutional network from step 3a on the same cards, compare it with the 84 → 128 → 7 network, look at what its detectors learned, and finally put the graphics card to work.

**Code:** `ConnectFourCNN` in [`src/connect4/torch_nn/models.py`](../../src/connect4/torch_nn/models.py), and the `cnn` recipe in [`src/connect4/torch_nn/winning_moves.py`](../../src/connect4/torch_nn/winning_moves.py). The numbers and charts come from [`scripts/make_stage3_charts.py`](../../scripts/make_stage3_charts.py).

---

## 1. The network and its recipe

`ConnectFourCNN`, from step 3a:

```
84 switches -> 2 grids of 6x7 (mine, opponent's)
  -> layer 1: 64 detectors, 4x4, over both grids  -> 64 maps of 6x7, ReLU
  -> layer 2: 64 detectors, 3x3, over all 64 maps -> 64 maps of 6x7, ReLU
  -> final layer: every number in every map       -> 7 totals, one per column
```

That's 57,863 dials. The recipe is step 2b's upgrades: cross-entropy for the error, ReLU for the squash, and **Adam** to turn the dials. Adam, rather than SGD with a searched step size, because this is a new network and nobody knows a good SGD step size for it yet; Adam's usual setting is a sensible first try. It worked first time, so no search was needed.

Everything else is unchanged: the same cards, batches of 64, 50 epochs, the same scores.

## 2. Finally, a job for the graphics card

Time for one epoch:

| | Processor | Graphics card | Faster |
|---|---|---|---|
| 84 → 128 → 7 network (step 2a), batches of 64 | 1.4 s | 2.8 s | processor, 2× |
| CNN, batches of 64 | 11.0 s | 6.7 s | graphics card, 1.6× |
| CNN, batches of 512 | 6.5 s | 1.1 s | **graphics card, 6×** |

The CNN does far more arithmetic per card than the 84 → 128 → 7 network: every detector is worked out at all 42 positions. That's enough for the graphics card to win even with batches of 64. With bigger batches there's more arithmetic per round to share among its thousands of cores, so it pulls well ahead. (These notes keep batches of 64 so every network is trained the same way.) The CNN trains on the graphics card; `best_device_for()` picks it automatically.

## 3. Results

"Finds the win" on the 30,000 test positions:

| Network | After 1 epoch | 2 | 5 | 10 | **50** | False alarms |
|---|---|---|---|---|---|---|
| Stage 2's recipe | 15.7% | 36.5% | 68.2% | 76.9% | **92.9%** | 0.9% |
| Best 84 → 128 → 7 (step 2b) | 88.3% | | 94.8% | 96.9% | **98.2%** | 0.9% |
| **CNN** | 97.4% | 99.4% | 98.2% | 100.0% | **100.0%** | **0.0%** |

After a single pass through the cards, the CNN is already ahead of everything else in this course. At every check from epoch 10 on it shows 100.0%. Exactly, after 50 epochs:

- Of the 10,649 test positions with a winning move, it finds the win in **10,647**.
- Of the 19,351 with none, it claims a win in **0**.

Its practice score is 100.0% too, and the test score matches it, so it isn't memorising: it found the rule. Training doesn't always improve smoothly: there's a small dip at epoch 5 (98.2%), and by the next check it had recovered.

## 4. The diagonal weak spot is gone

![Grouped bar chart of winning moves spotted on the test positions, by direction, for three networks. Vertical: 99.6% for stage 2's recipe, 100.0% for the best 84-128-7 network, 100.0% for the CNN. Horizontal: 88.0%, 97.3%, 100.0%. Diagonal: 76.5%, 91.2%, 99.9%](images/03-directions.png)

| | Vertical | Horizontal | Diagonal |
|---|---|---|---|
| Stage 2's recipe | 99.6% | 88.0% | 76.5% |
| Best 84 → 128 → 7 | 100.0% | 97.3% | 91.2% |
| **CNN** | **4,830 of 4,830** | **5,003 of 5,005** | **2,941 of 2,943** |

The 84 → 128 → 7 network had to learn each of the 24 diagonal lines separately. The CNN learns "four along a diagonal" once, in a 4 × 4 detector, and slides it everywhere.

## 5. What its detectors learned

These are 8 of layer 1's 64 detectors: the ones with the biggest weights. Each pair of grids is one 4 × 4 detector, read the same way as stage 2's hidden neurons (section 8 of step 4b): its weights on "my piece here?" on the left, on "my opponent's piece here?" on the right, blue counting for, red against.

![Eight first-layer detectors, each drawn as two 4-by-4 grids of weights, one for my pieces and one for my opponent's. Detector 6 has a blue column of my pieces and a red column of opponent's pieces in the same place. Detectors 21 and 12 have blue diagonals of my pieces, one falling and one rising, with red on the same diagonals for the opponent. Detectors 51 and 14 have three blue cells of my pieces in a row, red in the gap, and a cell below the gap that is blue in both grids](images/03-cnn-detectors.png)

- **Detector 6: a stack.** A blue column of my pieces, and red on the same column for my opponent's: "a column of mine with none of theirs in it". The same idea as stage 2's neuron 108, but this one works in every column at once.
- **Detectors 21 and 12: the two diagonals.** Blue along a falling diagonal (21) and a rising one (12) for my pieces, red along the same diagonal for my opponent's. The patterns stage 2's network found hardest are now two detectors.
- **Detectors 51 and 14: across, with gravity.** Three of my pieces in a row (blue on the left), with the gap red in both grids: it must be empty. And the cell *under* the gap is blue in **both** grids: it must be filled, by anyone, or a piece dropped there would fall straight past. It's stage 2's neuron 127 trick again, "it worked out gravity", now learned once for the whole board.
- **The others** (44, 31, 24) are harder to read. Layer 2 combines all 64 maps, so not every detector has to mean something on its own.

The graphics card adds numbers up in a slightly different order on each run, so if you rerun the script the detectors may come out in a different order, with different numbers. The same kinds of pattern turn up.

## 6. Blocking

Swapping the halves of the input (stage 2, step 4b, section 10), the CNN finds the *opponent's* winning column in **100.0%** of test positions, with **0.0%** false alarms. It handles **100.0%** of the 2,442 "win or block" moments from stage 2.

## 7. In play: a perfect spotter, and no better than the heuristic bot

1,000 games each:

| Player 1 (X) | Player 2 (O) | CNN wins | Opponent wins | Draws |
|---|---|---|---|---|
| CNN | random | **97.5%** | 2.4% | 0.1% |
| random | CNN | **95.3%** | 4.5% | 0.2% |
| CNN | heuristic | 50.3% | 45.6% | 4.1% |
| heuristic | CNN | 44.3% | 52.2% | 3.5% |
| *heuristic* | *heuristic* | *≈ 51%* | *≈ 45%* | *≈ 4%* |

The CNN finally draws level with the heuristic bot: the same results as two heuristic bots playing each other (the last row, from the [docs index](../README.md)). It never misses a win or a block, so it plays exactly the heuristic bot's strategy.

And that's as far as this approach can go. When there's nothing to win and nothing to block, which is most moves, the player still picks a **random** column. The network was only ever asked "can I win right now?", never "which of these quiet moves is good?". Perfect eyesight with the same strategy gives the same strength.

## 8. What stage 3 built, and what stage 4 changes

**Stage 3 in one table** ("finds the win" on test positions, after 50 epochs):

| | Network | Dials | Finds the win | Against the heuristic bot (as player 1) |
|---|---|---|---|---|
| Stage 2 | 84 → 128 → 7, by hand in NumPy | 11,783 | 92.9% | 38.3% wins, 59.2% losses |
| Step 2b | same shape, cross-entropy + ReLU | 11,783 | 98.2% | 45.8% wins, 51.2% losses |
| Step 3b | CNN | 57,863 | **100.0%** | 50.3% wins, 45.6% losses: level |

**Stage 4 changes the question on the back of the card,** from "can I win right now?" to "how good is this position?", the question your FYP answered. Nobody can label 300,000 positions with that, so the answers come from **self-play**: the network plays games against itself, and how each game ends becomes the answer for its positions. A CNN like this one is the kind of network it'll use, the graphics card will do the heavy lifting, and the hand-written "win, else block, else random" strategy can finally go: a network that knows how good every move is can simply pick the best.

## Summary

- **The CNN** slides 64 learned 4 × 4 detectors over both grids, then 3 × 3 detectors over their maps: 57,863 dials, trained with cross-entropy, ReLU and Adam.
- **The graphics card pays off at last:** 1.6× faster than the processor with batches of 64, 6× with batches of 512.
- **It's essentially perfect:** after 50 epochs it finds 10,647 of 10,649 test wins with no false alarms, including 2,941 of 2,943 diagonals. It passes 97% after a single epoch.
- **Its detectors** include stacks, both diagonals, and rows that check the empty cell is supported: each learned once for the whole board.
- **In play** it draws level with the heuristic bot, because it plays the same strategy perfectly. Beating it needs a network that judges quiet moves too: stage 4.

## New words in this step

| Word | Meaning |
|---|---|
| **Weight sharing** (again) | The reason the CNN learns each pattern once: every detector's weights are used at every position. |
| **Self-play** (stage 4) | Training on games a network plays against itself, with each game's result as the answer for its positions. |

## Try it yourself

1. **Play it.** Train it (`python -m connect4.torch_nn.winning_moves`, a few minutes on the graphics card) and play: `connect4 --p2 neural --model models/winning-moves-cnn.pt`. Diagonals won't work any more. What does? (Hint: it can only block one threat at a time.)
2. **How few epochs?** `--epochs 1` takes well under a minute on the graphics card. How close to 100% is it after one pass?
3. **Fewer detectors.** In `src/connect4/torch_nn/models.py`, change `detectors: int = 64` to `8`, then train with `python -m connect4.torch_nn.winning_moves --epochs 5 --out models/small-cnn.pt`. Guess first: how many dials does it have, and does it still reach 100%? Undo the change afterwards with `git checkout src/connect4/torch_nn/models.py`.
