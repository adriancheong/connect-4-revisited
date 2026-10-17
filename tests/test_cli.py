import io

import pytest

from connect4.board import Board
from connect4.cli import main, render_board


def test_render_board():
    board = Board.from_moves([3, 3, 2])
    assert render_board(board) == "\n".join(
        [
            " 1 2 3 4 5 6 7",
            "|. . . . . . .|",
            "|. . . . . . .|",
            "|. . . . . . .|",
            "|. . . . . . .|",
            "|. . . O . . .|",
            "|. . X X . . .|",
        ]
    )


def test_render_board_with_color_keeps_the_symbols():
    text = render_board(Board.from_moves([3]), color=True)
    assert "\033[" in text
    assert "X" in text


def test_many_bot_games_print_a_tally(capsys):
    assert main(["--p1", "random", "--p2", "heuristic", "--games", "20", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert "Played 20 games" in out
    assert "Draws:" in out


def test_single_bot_game_shows_the_board_and_result(capsys):
    assert main(["--p1", "heuristic", "--p2", "random", "--seed", "1"]) == 0
    out = capsys.readouterr().out
    assert " 1 2 3 4 5 6 7" in out
    assert "wins!" in out or "draw" in out


def test_human_game_can_be_quit(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("q\n"))
    assert main(["--p2", "heuristic"]) == 0
    assert "Game abandoned." in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["--games", "5"],  # humans can't play a batch of games
        ["--p1", "random", "--p2", "random", "--games", "0"],
        ["--p1", "nobody"],
    ],
)
def test_bad_options_are_rejected(argv):
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
