# Learning notes

Step-by-step notes on building and training neural networks, using Connect 4 as the running example. Each page covers the idea in plain language, walks through the code, shows the results, and ends with experiments to try.

## Stage 2: A neural network from scratch (NumPy)

| Step | Page | What you learn |
|---|---|---|
| 1 | [One neuron, by hand](stage-2/01-one-neuron-by-hand.md) | What a neuron is, and exactly how one training step adjusts its weights: loss, slopes (gradients), gradient descent, the chain rule |
| 2 | [Training a neuron on many examples](stage-2/02-training-a-neuron.md) | Averaging over a dataset, watching weights settle, the learning rate, and why one neuron can't learn XOR |
| 3 | Hidden layers and backpropagation (coming) | Wiring neurons into layers, and finding the slope of every weight inside the network |
| 4 | Spotting winning moves in Connect 4 (coming) | Turning a board into numbers, training on generated positions, testing on positions the network has never seen |

To regenerate the charts: `pip install -e ".[plots]"`, then `python scripts/make_stage2_charts.py`.

## Later stages

| Stage | Topic |
|---|---|
| 3 | The same network in PyTorch, then a convolutional network (CNN) |
| 4 | A bot that teaches itself by self-play: reinforcement learning, ε-greedy exploration, discounting |
| 5 | Optional: AlphaZero-style policy + value network with tree search |

## Stage 1 recap: the game engine

Stage 1 built the game (see the main [README](../README.md)). The parts the learning stages rely on:

- Every player, human or bot, uses `choose_move(board) -> column`. The neural-network bots plug in the same way, with no changes to the game.
- `Board.winning_moves()` finds every column that wins immediately. Stage 2, step 4 uses it to label training data. The engine gives the exact answer for any board, so there is unlimited correct training data.
- `play_game()` returns every position of the game in order. Stage 4's self-play training learns from that history.
- `HeuristicPlayer` is the first opponent the trained bot has to beat.

### Baselines

1,000 games each with `--seed 1`, plus 3 × 10,000 games for heuristic vs heuristic:

| Player 1 (X) | Player 2 (O) | Player 1 wins | Player 2 wins | Draws |
|---|---|---|---|---|
| random | random | 55.0% | 44.7% | 0.3% |
| heuristic | random | 96.5% | 3.3% | 0.2% |
| random | heuristic | 4.6% | 95.2% | 0.2% |
| heuristic | heuristic | ~51% | ~45% | ~4% |

Even random play shows the first-move advantage. When both sides block obvious wins, draws rise more than tenfold, but player 1 still leads. A trained bot should beat `heuristic` clearly from either side.
