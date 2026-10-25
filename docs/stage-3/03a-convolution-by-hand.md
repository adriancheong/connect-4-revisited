# Stage 3, step 3a: How a convolution works, by hand

**Read first:** [Step 2b: Three standard upgrades](02b-three-upgrades.md), and section 8 of [stage 2, step 4b](../stage-2/04b-training-on-connect-4.md) (what the hidden neurons learned).

**Goal:** understand the idea behind a convolutional network (CNN): one small pattern detector, slid across the whole board. Work one example through by hand, then check that PyTorch gives the same numbers.

**Code:** [`src/connect4/torch_nn/convolution.py`](../../src/connect4/torch_nn/convolution.py). Run `python -m connect4.torch_nn.convolution` to print the worked example.

---

## 1. The problem: the same lesson, learned again for every column

Section 8 of stage 2, step 4b looked inside the 84 → 128 → 7 network. Its strongest hidden neurons were stack detectors: neuron 108 looked for "three of mine on top of column 6", neuron 34 for the same in column 1, neuron 59 in column 3, and so on. Each one is a separate neuron with its own 84 weights, and each had to be learned separately, from cards where that particular column had a stack.

That's wasteful. A stack of three is the same pattern wherever it is. Every one of the board's 69 lines of four is one of just four shapes (across, up, and the two diagonals) in a different place. A network that learned "three in a row" once and could look for it everywhere would need far fewer examples to learn it, and would spot it equally well in every part of the board.

That is what a **convolutional network** does.

## 2. The idea: one small detector, slid everywhere

A **detector** is a small grid of weights, say 3 cells tall and 1 wide, plus a bias. To use it:

1. Place it over one spot on the board.
2. Multiply each switch under it by the weight on top, add them up, add the bias, and squash. That's exactly a neuron, but one that only looks at 3 cells.
3. Slide it one cell along and do the same again, until it has visited every position on the board.

The result is a **map**: one number for every position, saying how strongly the detector's pattern is there. The same few weights are used at every position, so whatever the detector has learned, it looks for everywhere.

## 3. A worked example

The board, with red to move, and red's pieces as switches (the "my pieces" grid from stage 2):

```
 1 2 3 4 5 6 7           my pieces (red), top row first
|. . . . . . .|          0 0 0 0 0 0 0
|. . . . . . .|          0 0 0 0 0 0 0
|. . . . . . .|          0 0 0 0 0 0 0
|. . X . . . .|          0 0 1 0 0 0 0
|. . X O . . .|          0 0 1 0 0 0 0
|. . X O . . O|          0 0 1 0 0 0 0
```

**The detector:** "three of my pieces stacked in a column". It's 3 cells tall and 1 wide, with weight 1 on each of my pieces and 0 on my opponent's, and a bias of −2. Its squash is ReLU from step 2b: anything below 0 becomes 0. So it gives 1 when all three cells are mine, and 0 otherwise.

![Left: red's pieces as a 6-by-7 grid of switches, with 1s in column 3, rows 1 to 3. An orange box marks the detector's position over those three cells; a dashed box marks the position one row higher. Middle: the detector, three cells each with weight 1, and bias −2. Right: its map, a 4-by-7 grid that is 0 everywhere except a single 1 in column 3, row 1](images/03-convolution.png)

Sliding it over three of its 28 positions:

| Position | My pieces under it | Total | After ReLU |
|---|---|---|---|
| Column 3, rows 1–3 (orange box) | 1, 1, 1 | 1 + 1 + 1 − 2 = **+1** | **1** |
| Column 3, rows 2–4 (dashed box) | 1, 1, 0 | 1 + 1 + 0 − 2 = **0** | 0 |
| Column 4, rows 1–3 | 0, 0, 0 (they're yellow's) | 0 + 0 + 0 − 2 = **−2** | 0 |

A 3-tall detector fits in 4 positions up each column (rows 1–3, 2–4, 3–5 and 4–6) and 7 across, so its map is 4 × 7. It's 0 everywhere except one 1: column 3, rows 1–3. The detector found the stack, and nothing else.

`python -m connect4.torch_nn.convolution` does this sum by hand in a loop, then asks PyTorch's convolution (`nn.functional.conv2d`) to do it. The two maps are identical; the tests check the same on random boards and random detectors of other sizes.

## 4. From one detector to a convolutional layer

The worked example used one hand-made detector. A real convolutional layer is the same idea, scaled up, with the network learning the numbers:

- **Bigger detectors, on both grids.** Step 3b's detectors are 4 × 4 cells, big enough to hold a whole line of four in any direction, and each one looks at both grids: my pieces and my opponent's. That's 2 × 4 × 4 = 32 weights plus a bias: **33 dials**, used at every position on the board.
- **Many detectors.** Layer 1 has 64 of them, each starting from random weights and learning its own pattern. So layer 1 turns one board into 64 maps.
- **Padding.** The board is surrounded with a border of empty cells before sliding, so every map stays 6 × 7, the size of the board. Without it, a 4 × 4 detector's map would only be 3 × 4, and lines along the edges would be harder to see.
- **A second layer on top.** Layer 2's detectors are 3 × 3, and each looks at all 64 of layer 1's maps at once. So it can combine simple patterns into bigger ones, like "three of mine in a row" *and* "the cell at the end is empty and supported".
- **A final layer** reads every map and gives the 7 totals, one per column, just like the output layer of stage 2's network.

| | Stage 2's network (84 → 128 → 7) | The CNN (step 3b) |
|---|---|---|
| Dials | 11,783 | 57,863 |
| A pattern learned in one place | works only there | works everywhere |

The CNN has about five times as many dials: 36,928 in layer 2 (64 detectors, each 3 × 3 × 64 weights plus a bias) and 18,823 in the final layer. Layer 1's 64 detectors have just 2,112 dials between them, and each detector's 33 dials are reused at all 42 positions on the board.

## Summary

- **A convolution slides one small detector across the board,** doing a neuron's sum at every position, and makes a map of where its pattern is.
- **The same weights are used everywhere,** so a pattern learned in one place is spotted in every place. Stage 2's network had to learn each column's stack detector separately.
- **The worked example** found the one stack of three on the board, by hand and with PyTorch's `conv2d`, with identical results.
- **A convolutional layer** has many detectors (64 in step 3b), each looking at both grids, padded so each map stays board-sized. A second layer combines their maps into bigger patterns.

## New words in this step

| Word | Meaning |
|---|---|
| **Convolutional network** (CNN) | A network built from layers of small detectors, each slid across the whole input. Built for grids: images, and boards. |
| **Detector** (textbook: **kernel** or **filter**) | A small grid of weights, plus a bias, that looks for one pattern. |
| **Map** (textbook: **feature map** or **activation map**) | A detector's output at every position: where its pattern is, and how strongly. |
| **Channel** | One of the grids going into or out of a layer: the 2 input grids (mine, opponent's), or the 64 maps from layer 1. |
| **Padding** | A border of empty cells added around the board so the maps stay board-sized. |
| **Weight sharing** | Using the same detector's weights at every position. It's why a CNN learns a pattern once for the whole board. |

(A footnote for the curious: strictly speaking, PyTorch's `conv2d` doesn't flip the detector before sliding it, which mathematicians call *cross-correlation* rather than convolution. Everyone in machine learning calls it convolution anyway, and for learned detectors it makes no difference.)

## Try it yourself

1. **A different detector.** In Python, make a detector for "three of my pieces in a row, across": `weights = np.zeros((2, 1, 3)); weights[0, 0, :] = 1`, bias −2. Slide it over the example board with `convolution.slide(...)`. What's the map's shape, and why? Where are its 1s?
2. **Block it.** Change the detector to give 0 when the opponent's piece is on top of the stack. Hint: make it 4 tall, and give the top cell a negative weight on the opponent's grid. Check it on `Board.from_moves([2, 3, 2, 3, 2, 2])`.
3. **Count the positions.** How many positions does a 4 × 4 detector have on a 6 × 7 board without padding? With padding?

---

**Previous:** [Step 2b: Three standard upgrades](02b-three-upgrades.md) · **Next:** [Step 3b: Training a CNN on Connect 4](03b-training-a-cnn.md)
