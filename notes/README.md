# Learning notes

One note per stage. Each covers the idea in plain language, a walkthrough of the code, the results (charts, win rates), and experiments to try.

| Stage | Note | What it teaches |
|---|---|---|
| 1 | (no note, see the main README) | A clean game engine for the bots to plug into |
| 2 | `02-network-from-scratch.md` (coming) | How a network makes a prediction, and how training adjusts its weights: loss, gradients, backpropagation, gradient descent |
| 3 | `03-pytorch-and-cnns.md` | What PyTorch automates, optimisers, overfitting, convolutional networks |
| 4 | `04-self-play-reinforcement-learning.md` | Self-play, exploration vs exploitation (ε-greedy), discounting rewards |
| 5 | Optional | Policy + value networks and tree search (AlphaZero-style) |

## Why stage 1 matters for the later stages

- Every player, human or bot, uses `choose_move(board) -> column`. The neural-network bots will plug in the same way, with no changes to the game.
- `Board.winning_moves()` finds every column that wins immediately. Stage 2 uses it to label training data: "does the player to move have a winning move, and where?" The engine gives the exact answer for any board, so there is unlimited, correct training data.
- `play_game()` returns every position of the game in order. Stage 4's self-play training learns from that history.
- `HeuristicPlayer` is the first opponent the trained bot has to beat.

## Baselines from stage 1

1,000 games each (`--seed 1`):

| Player 1 (X) | Player 2 (O) | Player 1 wins | Player 2 wins | Draws |
|---|---|---|---|---|
| random | random | 55.0% | 44.7% | 0.3% |
| heuristic | random | 96.5% | 3.3% | 0.2% |
| random | heuristic | 4.6% | 95.2% | 0.2% |

Even random play shows the first-move advantage. A trained bot should end up beating `heuristic` clearly from either side.
