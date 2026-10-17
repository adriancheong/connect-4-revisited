import pytest

from connect4.board import ROWS, Board
from connect4.players import HeuristicPlayer, HumanPlayer, QuitGame, RandomPlayer


def scripted_human(answers):
    """A HumanPlayer that reads `answers` instead of the keyboard and records what it prints."""
    answers = iter(answers)
    printed = []

    def fake_input(prompt):
        try:
            return next(answers)
        except StopIteration:
            raise EOFError from None

    return HumanPlayer(input_fn=fake_input, output_fn=printed.append), printed


def test_human_column_numbers_start_at_1():
    human, _ = scripted_human(["4"])
    assert human.choose_move(Board()) == 3


def test_human_is_asked_again_after_invalid_input():
    human, printed = scripted_human(["abc", "", "0", "8", "²", "2"])
    assert human.choose_move(Board()) == 1
    assert len(printed) == 5
    assert all("number from 1 to 7" in line for line in printed)


def test_human_cannot_pick_a_full_column():
    human, printed = scripted_human(["1", "2"])
    board = Board.from_moves([0] * ROWS)
    assert human.choose_move(board) == 1
    assert printed == ["Column 1 is full, pick another."]


@pytest.mark.parametrize("answers", [["q"], [" Quit "], []])
def test_human_can_quit(answers):
    human, _ = scripted_human(answers)
    with pytest.raises(QuitGame):
        human.choose_move(Board())


def test_random_player_only_picks_legal_columns():
    board = Board.from_moves([0] * ROWS + [6] * ROWS)
    player = RandomPlayer(seed=0)
    picks = {player.choose_move(board) for _ in range(200)}
    assert picks == {1, 2, 3, 4, 5}


def test_heuristic_takes_a_win():
    board = Board.from_moves([0, 0, 1, 1, 2, 6])  # X to move with three along the bottom
    assert HeuristicPlayer(seed=0).choose_move(board) == 3


def test_heuristic_blocks_a_threat():
    board = Board.from_moves([0, 6, 1, 6, 2])  # O to move, X threatens column 3
    assert HeuristicPlayer(seed=0).choose_move(board) == 3


def test_heuristic_prefers_winning_to_blocking():
    board = Board.from_moves([0, 6, 1, 6, 2, 6, 4])  # O can win in column 6 or block column 3
    assert HeuristicPlayer(seed=0).choose_move(board) == 6
