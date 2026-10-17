# connect-4-revisited

To revisit my FYP in Uni.

The original project taught two bots to play Connect 4 by storing a score for every board position they saw. That worked, but the table grew to gigabytes. This version teaches the bots with a **neural network** instead, and is built step by step as a way to learn how neural networks work.

## Setup

Needs Python 3.11 or newer, and [uv](https://docs.astral.sh/uv/), a fast Python package installer. Install uv once with `pip install uv` (or `brew install uv` on a Mac).

```bash
uv sync --locked --all-extras    # creates .venv with the exact versions in uv.lock
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

Run `source .venv/bin/activate` again in each new terminal before using the commands below.

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

## Dependencies and supply-chain safety

- **What the project uses** is listed in `pyproject.toml`: numpy to run, plus pytest and ruff (the `dev` extra) and matplotlib (the `plots` extra) for development.
- **`uv.lock` pins the exact version of every package**, including the ones those packages pull in, each with a SHA-256 checksum. `uv sync --locked` installs exactly those versions, refuses any download whose checksum doesn't match, and stops if the lock file is out of date with `pyproject.toml`.
- **7-day cooldown:** `exclude-newer = "7 days"` under `[tool.uv]` in `pyproject.toml` makes uv ignore any release published in the last 7 days when it picks versions. Most malicious releases are caught and removed within days.
- **To update on purpose:** run `uv lock --upgrade`, review the version changes in `uv.lock`, run the tests, then commit.
- `pip install -e ".[dev,plots]"` also works, but pip ignores both the lock file and the cooldown.

## Code layout

| File | What it does |
|---|---|
| `src/connect4/board.py` | The rules: the board, legal moves, win detection |
| `src/connect4/players.py` | Human and bot players, all with `choose_move(board) -> column` |
| `src/connect4/game.py` | Runs a game and records every position |
| `src/connect4/cli.py` | Terminal display and command-line options |
| `src/connect4/numpy_nn/` | Stage 2: neural networks written from scratch in NumPy |
| `scripts/make_stage2_charts.py` | Draws the charts in `docs/stage-2/` |

## Learning path

Step-by-step notes live in [`docs/`](docs/README.md). Start with [Start here: the big picture](docs/stage-2/00-start-here.md).

| Stage | Build | Status |
|---|---|---|
| 1 | Playable game, human or bot on either side | Done |
| 2 | A neural network written by hand in NumPy | In progress (steps 1-2 of 4 done) |
| 3 | The same network in PyTorch, then a convolutional network | |
| 4 | A bot that teaches itself by self-play | |
| 5 | Optional: AlphaZero-style bot | |
