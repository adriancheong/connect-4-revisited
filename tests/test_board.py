import pytest

from connect4.board import COLS, PLAYER_1, PLAYER_2, ROWS, Board

# Fills the whole board in a pattern with no four in a row anywhere.
DRAW_MOVES = [
    0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 4, 2, 2, 2, 2, 2, 2, 3, 3,
    3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 5,
]  # fmt: skip


def mirror(moves):
    """Flip a move sequence left to right."""
    return [COLS - 1 - col for col in moves]


def test_new_board():
    board = Board()
    assert board.current_player == PLAYER_1
    assert board.legal_moves() == list(range(COLS))
    assert board.winner is None
    assert not board.is_terminal()
    assert not board.grid.any()


def test_play_returns_new_board_and_leaves_old_one_unchanged():
    board = Board()
    after = board.play(3)
    assert board.moves == ()
    assert not board.grid.any()
    assert after.moves == (3,)


def test_grid_is_read_only():
    board = Board()
    with pytest.raises(ValueError):
        board.grid[0, 0] = 1


def test_pieces_stack_and_players_alternate():
    board = Board.from_moves([3, 3, 3])
    assert board.grid[0, 3] == PLAYER_1
    assert board.grid[1, 3] == PLAYER_2
    assert board.grid[2, 3] == PLAYER_1
    assert board.current_player == PLAYER_2


@pytest.mark.parametrize(
    "moves",
    [
        pytest.param([0, 0, 1, 1, 2, 2, 3], id="horizontal"),
        pytest.param([3, 3, 4, 4, 5, 5, 6], id="horizontal at right edge"),
        pytest.param([0, 0, 1, 1, 3, 3, 2], id="horizontal, gap filled last"),
        pytest.param([0, 1, 0, 1, 0, 1, 0], id="vertical"),
        pytest.param([0, 6, 1, 6, 6, 0, 6, 1, 6, 0, 6], id="vertical reaching the top row"),
        pytest.param([0, 1, 1, 2, 2, 3, 2, 3, 6, 3, 3], id="diagonal up-right"),
        pytest.param(mirror([0, 1, 1, 2, 2, 3, 2, 3, 6, 3, 3]), id="diagonal up-left"),
    ],
)
def test_player_1_wins(moves):
    before = Board.from_moves(moves[:-1])
    assert before.winner is None
    assert moves[-1] in before.winning_moves()

    board = before.play(moves[-1])
    assert board.winner == PLAYER_1
    assert board.is_terminal()
    assert board.legal_moves() == []


def test_player_2_can_win():
    board = Board.from_moves([0, 1, 0, 1, 0, 2, 6, 1, 6, 1])
    assert board.winner == PLAYER_2


def test_three_in_a_row_is_not_a_win():
    board = Board.from_moves([0, 0, 1, 1, 2])
    assert board.winner is None
    assert not board.is_terminal()


def test_full_column_is_rejected():
    board = Board.from_moves([0] * ROWS)
    assert 0 not in board.legal_moves()
    with pytest.raises(ValueError, match="full"):
        board.play(0)


@pytest.mark.parametrize("col", [-1, COLS])
def test_column_out_of_range_is_rejected(col):
    with pytest.raises(ValueError):
        Board().play(col)


def test_no_moves_after_a_win():
    board = Board.from_moves([0, 1, 0, 1, 0, 1, 0])
    with pytest.raises(ValueError, match="over"):
        board.play(2)


def test_full_board_without_a_line_is_a_draw():
    board = Board.from_moves(DRAW_MOVES)
    assert board.is_full()
    assert board.is_terminal()
    assert board.winner is None
    assert board.legal_moves() == []


def test_winning_moves_for_either_player():
    # X has three along the bottom (columns 0-2). O has three up column 6.
    board = Board.from_moves([0, 6, 1, 6, 2, 6])
    assert board.current_player == PLAYER_1
    assert board.winning_moves() == [3]
    assert board.winning_moves(PLAYER_2) == [6]
    # Checking must not change the board.
    assert board.grid[0, 3] == 0
    assert board.grid[3, 6] == 0
