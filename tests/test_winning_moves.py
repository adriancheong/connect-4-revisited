import numpy as np
import pytest

from connect4.board import Board
from connect4.cli import main
from connect4.numpy_nn import network as nw
from connect4.numpy_nn import winning_moves as wm

# Red (X, player 1) to move, with three along the bottom: column 1 (index 0) wins.
RED_CAN_WIN = [3, 3, 2, 2, 1, 4]


def test_empty_board_is_all_switches_off():
    x = wm.encode(Board())
    assert x.shape == (wm.INPUTS,)
    assert not x.any()


def test_encoding_is_from_the_point_of_view_of_the_player_to_move():
    one_piece = Board.from_moves([3])  # X in column 4 (cell 3); now O is to move
    x = wm.encode(one_piece)
    assert x[3] == 0  # not O's piece...
    assert x[42 + 3] == 1  # ...it's the opponent's

    two_pieces = one_piece.play(3)  # O on top of it (cell 10); now X is to move
    x = wm.encode(two_pieces)
    assert x[3] == 1  # X's own piece
    assert x[42 + 10] == 1  # O's piece is the opponent's
    assert x.sum() == 2


def test_simple_encoding_uses_plus_and_minus_one():
    x = wm.encode_simple(Board.from_moves([3, 3]))  # X to move
    assert x.shape == (42,)
    assert x[3] == 1
    assert x[10] == -1
    assert np.count_nonzero(x) == 2


def test_swap_sides():
    x = wm.encode(Board.from_moves(RED_CAN_WIN))
    swapped = wm.swap_sides(x)
    assert (swapped[:42] == x[42:]).all()
    assert (wm.swap_sides(swapped) == x).all()


def test_answers_come_from_the_game_engine():
    board = Board.from_moves(RED_CAN_WIN)
    assert wm.answers(board).tolist() == [1, 0, 0, 0, 0, 0, 0]
    assert wm.answers(board, player=2).sum() == 0


def test_positions_are_all_different_and_unfinished():
    positions = wm.make_positions(500, seed=3)
    assert len({(b.grid.tobytes(), b.current_player) for b in positions}) == 500
    assert not any(b.is_terminal() for b in positions)


def test_test_positions_never_appear_in_practice():
    data = wm.make_dataset(practice=1000, test=200, seed=3)
    assert data.practice_inputs.shape == (1000, 84)
    assert data.test_answers.shape == (200, 7)
    practice = {row.tobytes() for row in data.practice_inputs}
    assert not any(row.tobytes() in practice for row in data.test_inputs)


def test_score():
    right = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=float)
    perfect = np.array([[0.9, 0.1, 0, 0, 0, 0, 0], [0.1, 0, 0, 0, 0, 0, 0]])
    s = wm.score(perfect, right)
    assert (s.found, s.false_alarms, s.all_right) == (1.0, 0.0, 1.0)

    wrong = np.array([[0.2, 0.6, 0, 0, 0, 0, 0], [0, 0, 0, 0.7, 0, 0, 0]])
    s = wm.score(wrong, right)
    assert (s.found, s.false_alarms, s.all_right) == (0.0, 1.0, 0.0)

    # Right column on top, but not confident enough to say "win": not counted as found.
    timid = np.array([[0.4, 0.1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
    assert wm.score(timid, right).found == 0.0
    always_no = np.zeros_like(right)
    s = wm.score(always_no, right)
    assert (s.found, s.false_alarms) == (0.0, 0.0)


def test_a_short_training_run_learns_something():
    data = wm.make_dataset(practice=3000, test=500, seed=3)
    untrained = nw.new_network(84, 32, 7, seed=0)
    before = wm.evaluate(untrained, data.test_inputs, data.test_answers)
    _, history = wm.train(data, hidden=32, epochs=5)
    assert len(history) == 5
    assert history[-1].test.error < before.error / 2


class FakeNeuralPlayer(wm.NeuralPlayer):
    """A NeuralPlayer whose guesses we choose, to test its decisions."""

    def __init__(self, mine, theirs):
        super().__init__(net=None, seed=0)
        self._fake = (np.array(mine, dtype=float), np.array(theirs, dtype=float))

    def guesses(self, board):
        return self._fake


LOW = [0.1] * 7


def test_neural_player_takes_a_win_first():
    player = FakeNeuralPlayer(mine=[0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1], theirs=[0.8] + LOW[1:])
    assert player.choose_move(Board()) == 2


def test_neural_player_blocks_when_it_cannot_win():
    player = FakeNeuralPlayer(mine=LOW, theirs=[0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.1])
    assert player.choose_move(Board()) == 5


def test_neural_player_ignores_full_columns():
    full_first_column = Board.from_moves([0] * 6)
    player = FakeNeuralPlayer(mine=[0.99, 0.1, 0.1, 0.6, 0.1, 0.1, 0.1], theirs=LOW)
    assert player.choose_move(full_first_column) == 3


def test_neural_player_plays_randomly_otherwise():
    player = FakeNeuralPlayer(mine=LOW, theirs=LOW)
    board = Board.from_moves([0] * 6)
    assert {player.choose_move(board) for _ in range(50)} <= set(board.legal_moves())


def test_cli_needs_a_trained_network(tmp_path):
    with pytest.raises(SystemExit, match="Train one first"):
        main(["--p1", "neural", "--p2", "random", "--model", str(tmp_path / "missing.npz")])


def test_cli_plays_with_a_saved_network(tmp_path, capsys):
    path = tmp_path / "net.npz"
    nw.save(nw.new_network(84, 8, 7, seed=0), path)
    args = ["--p1", "neural", "--p2", "random", "--games", "3", "--model", str(path)]
    assert main(args) == 0
    assert "Played 3 games" in capsys.readouterr().out
