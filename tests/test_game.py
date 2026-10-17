from connect4.board import PLAYER_1, PLAYER_2
from connect4.game import play_game
from connect4.players import HeuristicPlayer, RandomPlayer


def test_random_games_always_finish_with_a_consistent_result():
    player1, player2 = RandomPlayer(seed=1), RandomPlayer(seed=2)
    for _ in range(200):
        result = play_game(player1, player2)
        final = result.boards[-1]
        assert final.is_terminal()
        assert result.winner == final.winner
        assert result.moves == final.moves
        assert len(result.boards) == len(result.moves) + 1
        assert result.boards[0].moves == ()
        if result.winner is not None:
            # The winner is the player who made the last move.
            assert result.winner == (PLAYER_1 if len(result.moves) % 2 == 1 else PLAYER_2)


def test_on_move_is_called_after_every_move():
    seen = []
    result = play_game(RandomPlayer(seed=3), RandomPlayer(seed=4), on_move=seen.append)
    assert seen == list(result.boards[1:])


def test_same_seeds_give_the_same_game():
    first = play_game(RandomPlayer(seed=5), HeuristicPlayer(seed=6))
    second = play_game(RandomPlayer(seed=5), HeuristicPlayer(seed=6))
    assert first.moves == second.moves


def test_heuristic_beats_random_from_either_side():
    heuristic, rand = HeuristicPlayer(seed=7), RandomPlayer(seed=8)
    games = 200
    as_player_1 = sum(play_game(heuristic, rand).winner == PLAYER_1 for _ in range(games))
    as_player_2 = sum(play_game(rand, heuristic).winner == PLAYER_2 for _ in range(games))
    assert as_player_1 > 0.8 * games
    assert as_player_2 > 0.8 * games
