# Stage 3, step 2b: Three standard upgrades

**Read first:** [Step 2a: The Connect 4 network in PyTorch](02a-connect-4-in-pytorch.md).

**Goal:** replace the three hand-built parts that hold stage 2's network back: the error measure, the hidden neurons' squash, and the rule for turning the dials. Add them one at a time, measure each, and see how close to 100% the same 84 → 128 → 7 network can get.

**Code:** `RECIPES` in [`src/connect4/torch_nn/winning_moves.py`](../../src/connect4/torch_nn/winning_moves.py). The numbers and charts come from [`scripts/make_stage3_charts.py`](../../scripts/make_stage3_charts.py).

---

## 1. Upgrade 1: an error measure built for yes/no answers (cross-entropy)

Every answer on our cards is a yes or no: 1 or 0. Stage 2 measured the error as (guess − answer)², and that has a weakness you met as the *flat plains* in stage 2, step 2.

Take a column that really wins (answer 1), where the network is confidently wrong: its total is about −4.6, which squashes to a guess of 0.01. Training needs a big push to fix that. But the slope that reaches the total is

> (how the error changes with the guess) × (the squash's steepness) = 2 × (0.01 − 1) × 0.0099 ≈ **−0.02**

The steepness, guess × (1 − guess), is almost 0 out on the flat part of the squash, so the slope is tiny. The more confidently wrong the network is, the less it learns.

**Cross-entropy** measures the error differently: −log(guess) when the answer is 1, and −log(1 − guess) when it's 0. It's 0 for a perfect guess, and it shoots up as a guess heads confidently the wrong way: −log(0.01) ≈ 4.6. Its real advantage is the slope. Its log exactly cancels the squash's steepness, so the slope reaching the total is simply

> guess − answer = 0.01 − 1 = **−0.99**

That's 50 times bigger than squared error's, for the case that most needs fixing.

| Guess (answer 1) | Squared error's slope at the total | Cross-entropy's slope at the total |
|---|---|---|
| 0.01 (confidently wrong) | −0.02 | **−0.99** |
| 0.5 (unsure) | −0.25 | −0.5 |
| 0.99 (right) | −0.0002 | −0.01 |

In PyTorch it's `nn.functional.binary_cross_entropy_with_logits(totals, answers)`. It takes the network's totals, before the squash, and does the squash itself in a way that stays accurate even for huge totals. That's why the networks in this stage give out totals, and `guesses()` adds the squash afterwards.

(*Binary* because each answer is one of two values. The *with logits* part: "logit" is the textbook word for a total that's about to be squashed.)

## 2. Upgrade 2: a squash that doesn't go flat (ReLU)

The hidden neurons' squash has the same problem one layer further back. In stage 2, step 3a, the blame passed back to each hidden neuron was multiplied by that neuron's steepness, which is at most 0.25 for the S-shaped squash, and close to 0 when the neuron is firing strongly or not at all. So hidden neurons that settle at either end learn slowly.

**ReLU** (short for *rectified linear unit*) is a much simpler squash:

> output = the total if it's above 0, otherwise 0

Its steepness is exactly 1 whenever the total is above 0, so blame passes back undiminished, and it's 0 otherwise. It's also quicker to work out than the S-curve. ReLU is the standard choice for hidden layers today.

Two things to know:
- **ReLU is only for hidden neurons.** Its outputs can be any size, so it can't be a yes/no guess. The outputs still use the S-shaped squash (inside cross-entropy).
- **A neuron can "die".** If its total is below 0 for every card, its steepness is always 0, it gets no blame and never learns again. Starting from small random weights mostly avoids this.

## 3. Upgrade 3: a smarter way to turn the dials (Adam)

Plain gradient descent, `SGD`, turns every dial by the same rule: step size × slope. One step size for 11,783 dials is a compromise. Dials with big, jumpy slopes would like small steps, dials with small, steady slopes would like big ones, and finding a step size that suits them all took trial and error in stage 2 (section 2 of step 4b).

**Adam** (from *adaptive moment estimation*) keeps two running averages for every dial:
- **The average slope.** A dial that keeps being pushed the same way builds up speed, like a ball rolling downhill (textbook: *momentum*). Slopes that flip back and forth cancel out, so the dial doesn't jitter.
- **The average size of the slope.** Each dial's step is divided by this, so every dial moves at a similar, sensible rate, whatever the scale of its slopes.

The result is a separate, self-adjusting step size for every dial. Its usual setting, 0.001, works for most networks without searching, which is why it's the most common choice today. In PyTorch it's one line: `torch.optim.Adam(model.parameters(), lr=0.001)`.

## 4. Choosing the step sizes

Each upgrade is added on top of the previous ones, from the same starting dials and with the same cards, so only one thing changes at a time. The recipes are in `RECIPES`:

| Recipe | Error | Hidden squash | Dial rule | Step size |
|---|---|---|---|---|
| `stage2` | squared | S-curve | SGD | 10 |
| `cross_entropy` | **cross-entropy** | S-curve | SGD | 10 |
| `relu` | cross-entropy | **ReLU** | SGD | 10 |
| `adam` | cross-entropy | ReLU | **Adam** | 0.001 |

A new error measure or squash changes the size of the slopes, so plain SGD needs its step size found again. Each was tried at 1, 3, 10 and 30 for 10 epochs, and the best kept. "Finds the win" on the **practice** cards:

| Step size | 1 | 3 | 10 | 30 |
|---|---|---|---|---|
| + cross-entropy | 72.5% | 88.5% | **95.8%** | 93.9% |
| + ReLU | 91.7% | 96.5% | **97.5%** | 78.2% |

The choice is made on the practice cards, not the test cards. If the test cards were used to pick settings, they'd stop being a fair exam: the settings would be tuned to those particular 30,000 positions. (Textbooks keep a third set of cards for this, the *validation set*.)

Adam gets its usual step size, 0.001, with no search: being able to skip the search is its selling point.

## 5. Results

![Line chart of "finds the winning column" on the test positions over 50 epochs, one line per recipe. Stage 2's recipe climbs slowly from 16% to 92.9%. Adding cross-entropy starts at 70% and reaches 97.3%. Adding ReLU starts at 88% and reaches 98.2%. Swapping SGD for Adam starts at 63% and reaches 96.0%](images/02-upgrades.png)

| Recipe | After 1 epoch | 5 | 10 | **50** | False alarms | All right |
|---|---|---|---|---|---|---|
| Stage 2's recipe | 15.7% | 68.2% | 76.9% | **92.9%** | 0.9% | 95.0% |
| + cross-entropy | 69.8% | 91.0% | 94.4% | **97.3%** | 0.4% | 97.7% |
| + ReLU | 88.3% | 94.8% | 96.9% | **98.2%** | 0.9% | 97.6% |
| + Adam (instead of SGD) | 62.9% | 86.2% | 91.2% | **96.0%** | 1.4% | 96.1% |

(Test positions; false alarms and all right after 50 epochs.)

- **Cross-entropy is the big one.** After one epoch it finds 70% of wins instead of 16%, and after 5 epochs it has passed what stage 2's recipe managed in 50. The confidently wrong answers that squared error barely moved now get the biggest pushes.
- **ReLU speeds it up again,** and gives the best network of this step: **98.2%**. After a single epoch it's already at 88%.
- **Adam did worse here.** With its usual step size it learned more slowly than SGD with a step size found by searching, and finished at 96.0% against 98.2%. Its practice score (98.3%) is also further above its test score than ReLU's (99.1% against 98.2%), so it memorised a little more. That's an honest result, not a failure of the idea: Adam's advantage is getting a good result with no search at all, and 96% from an untuned setting is good. For a small network whose step size you've already searched for, tuned SGD can win. For a new network where you don't know a good step size, like step 3's, Adam's default is the sensible place to start.

Each line is one training run, from one set of starting dials. A different start would move every number a little, so treat differences of under about a point as uncertain.

## 6. In play

The best network here, `relu` (cross-entropy, ReLU, SGD), playing stage 2's strategy, 1,000 games each:

| Player 1 (X) | Player 2 (O) | Neural wins | Opponent wins | Draws |
|---|---|---|---|---|
| neural | random | **95.8%** | 3.7% | 0.5% |
| random | neural | **94.8%** | 4.7% | 0.5% |
| neural | heuristic | 45.8% | **51.2%** | 3.0% |
| heuristic | neural | 37.1% | **59.6%** | 3.3% |

It handles **99.0%** of "win or block" moments correctly (stage 2's recipe: 94.5%; the Adam network: 96.9%). Section 11 of stage 2, step 4b worked out that a game has about 10 such moments; at 99% each, getting all 10 right happens 0.99¹⁰ ≈ 90% of the time, against about half the time at 94%.

So it's much closer to the heuristic bot, but still loses to it more often than it wins. For comparison, two heuristic bots playing each other give about 51% wins to player 1, 45% to player 2 and 4% draws (see the [docs index](../README.md)). A network that never missed a win or a block would play exactly like the heuristic bot, and would score about those numbers.

## 7. What's left: diagonals

Winning moves spotted, by the direction of the line they complete (test positions):

| | Vertical | Horizontal | Diagonal |
|---|---|---|---|
| Stage 2's recipe | 99.6% | 88.0% | 76.5% |
| Best network of this step | 100.0% | 97.3% | **91.2%** |

(Stage 2's own notes say 76.4% for diagonals. The PyTorch rerun spotted one more of the 2,943, thanks to the tiny rounding differences from step 2a.)

Diagonals are still the weak spot. The network has to learn every diagonal line separately, with its own hidden neurons, at every height and position on the board. Step 3 gives it a way to learn a pattern once and look for it everywhere.

## Summary

- **Cross-entropy** gives confidently wrong guesses the biggest push instead of the smallest. It was the biggest single improvement: 92.9% → 97.3%, and 70% after the first epoch instead of 16%.
- **ReLU** passes blame back to the hidden neurons without shrinking it. Together with cross-entropy: **98.2%**, the best 84 → 128 → 7 network.
- **Adam** adjusts each dial's step size by itself. With its usual setting and no search, it reached 96.0%: good, but behind SGD with a searched step size. Its strength is not needing the search.
- **Choose settings on the practice cards,** never the test cards, or the test stops being a fair exam.
- **In play**, the best network handles 99% of win-or-block moments, and gets close to the heuristic bot without matching it. Diagonals (91%) are still its weak spot.

## New words in this step

| Word | Meaning |
|---|---|
| **Cross-entropy** (here: *binary cross-entropy*) | An error measure for yes/no answers: −log(guess) if the answer is yes, −log(1 − guess) if no. Its slope at the total is simply guess − answer. |
| **Logit** | A total that's about to be squashed into a guess. |
| **ReLU** (rectified linear unit) | A hidden-neuron squash: the total if it's above 0, otherwise 0. |
| **Dead neuron** | A ReLU neuron whose total is below 0 for every card, so it never learns. |
| **Adam** | A rule for turning the dials that gives each dial its own self-adjusting step size, with momentum. Usual step size: 0.001. |
| **Momentum** | Building up speed in a direction the slopes keep pointing, so dials don't jitter. |
| **Validation set** | Cards kept apart for choosing settings, so the test cards stay a fair exam. This step used the practice cards instead. |

## Try it yourself

1. **Give Adam a fairer go.** Search Adam's step size the way SGD's was searched, for example 0.0003 and 0.003, with `replace(tw.RECIPES["adam"], learning_rate=0.003)` (see step 2a's last exercise). Does any setting beat 98.2%?
2. **One upgrade on its own.** Try ReLU *without* cross-entropy: `replace(tw.RECIPES["stage2"], squash="relu")`. You may need to search its step size too. Which upgrade matters more by itself?
3. **Play the best network.** `python -m connect4.torch_nn.winning_moves --recipe relu`, then `connect4 --p2 neural --model models/winning-moves-relu.pt`. Can you still beat it with a diagonal?
