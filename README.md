# connect-4-revisited

To revisit my FYP in Uni.

The original project taught two bots to play Connect 4 by storing a score for every board position they saw. That worked, but the table grew to gigabytes. This version teaches the bots with a **neural network** instead, and is built step by step as a way to learn how neural networks work.

## Setup

Needs Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Play

```bash
connect4                                          # two people, taking turns
connect4 --p2 heuristic                           # you (X) against a bot (O)
connect4 --p1 heuristic --p2 human                # the bot moves first
connect4 --p1 random --p2 heuristic --games 1000  # bot vs bot, prints a results tally
```

Type a column number (1-7) to drop a piece, or `q` to quit. `python -m connect4` works too.

Player types:
- `human`: you, at the keyboard.
- `random`: picks any legal column.
- `heuristic`: wins if it can, blocks your win if it must, otherwise random.

Other options: `--seed N` makes bot games repeatable, `--no-color` turns off coloured pieces.

## Tests

```bash
pytest
ruff check .
```

## Code layout

| File | What it does |
|---|---|
| `src/connect4/board.py` | The rules: the board, legal moves, win detection |
| `src/connect4/players.py` | Human and bot players, all with `choose_move(board) -> column` |
| `src/connect4/game.py` | Runs a game and records every position |
| `src/connect4/cli.py` | Terminal display and command-line options |

## Learning path

See [`notes/`](notes/README.md).

| Stage | Build | Status |
|---|---|---|
| 1 | Playable game, human or bot on either side | Done |
| 2 | A neural network written by hand in NumPy | Next |
| 3 | The same network in PyTorch, then a convolutional network | |
| 4 | A bot that teaches itself by self-play | |
| 5 | Optional: AlphaZero-style bot | |
