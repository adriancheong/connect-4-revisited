import numpy as np
import pytest

torch = pytest.importorskip("torch")

from connect4.board import Board  # noqa: E402
from connect4.cli import make_player  # noqa: E402
from connect4.numpy_nn import network as nw  # noqa: E402
from connect4.numpy_nn import winning_moves as wm  # noqa: E402
from connect4.torch_nn import convolution as conv  # noqa: E402
from connect4.torch_nn import winning_moves as tw  # noqa: E402
from connect4.torch_nn.models import ConnectFourCNN, dial_count, make_mlp  # noqa: E402


def test_network_shapes_and_dial_counts():
    x = torch.zeros(5, 84)
    assert make_mlp()(x).shape == (5, 7)
    assert ConnectFourCNN()(x).shape == (5, 7)
    assert dial_count(make_mlp()) == 11_783  # the same as stage 2's network
    assert dial_count(ConnectFourCNN()) == 57_863


def test_planes_put_my_pieces_first_with_row_0_at_the_bottom():
    board = Board.from_moves([0, 6])  # red bottom-left, yellow bottom-right; red to move
    p = conv.planes(board)
    assert p[0, 0, 0] == 1 and p[0].sum() == 1  # mine
    assert p[1, 0, 6] == 1 and p[1].sum() == 1  # opponent's


def test_mlp_recipes_start_from_stage_2s_dials():
    model = tw.build(tw.RECIPES["relu"])
    start = nw.new_network(84, 128, 7, seed=0)
    np.testing.assert_allclose(model[0].weight.detach().numpy(), start.hidden_weights.T, atol=1e-6)
    np.testing.assert_allclose(model[2].bias.detach().numpy(), start.output_biases, atol=1e-6)


def test_error_measures():
    totals = torch.tensor([[2.0, -1.0], [0.0, 3.0]])
    answers = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    p = torch.sigmoid(totals)
    squared = tw.error_measure(tw.Recipe(error="squared"))(totals, answers)
    assert squared.item() == pytest.approx(((p - answers) ** 2).mean().item())
    cross = tw.error_measure(tw.Recipe(error="cross_entropy"))(totals, answers)
    by_hand = -(answers * torch.log(p) + (1 - answers) * torch.log(1 - p)).mean()
    assert cross.item() == pytest.approx(by_hand.item())


@pytest.fixture(scope="module")
def small_data():
    return wm.make_dataset(practice=3_000, test=500)


def test_training_runs_and_learns(small_data):
    recipe = tw.Recipe(squash="relu", error="cross_entropy", optimizer="adam",
                       learning_rate=1e-3, epochs=3)  # fmt: skip
    model, history = tw.train(recipe, small_data, torch.device("cpu"))
    assert len(history) == 3
    assert history[-1].practice.error < history[0].practice.error


def test_save_and_load_give_the_same_guesses(tmp_path, small_data):
    recipe = tw.Recipe(net="cnn", error="cross_entropy", optimizer="adam",
                       learning_rate=1e-3, epochs=1)  # fmt: skip
    model, _ = tw.train(recipe, small_data, torch.device("cpu"), practice=False)
    path = tmp_path / "net.pt"
    tw.save(model, recipe, path)
    loaded, loaded_recipe = tw.load(path, torch.device("cpu"))
    assert loaded_recipe == recipe
    inputs = small_data.test_inputs[:50]
    np.testing.assert_allclose(tw.guesses(loaded, inputs), tw.guesses(model.cpu(), inputs))


def _always_says_column(col):
    """A stand-in network whose guess is near 1 for `col` and near 0 elsewhere."""
    net = torch.nn.Linear(84, 7)
    with torch.no_grad():
        net.weight.zero_()
        net.bias.fill_(-10.0)
        net.bias[col] = 10.0
    return net


def test_player_plays_where_the_network_sees_a_win():
    player = tw.TorchNeuralPlayer(_always_says_column(3), seed=0)
    assert player.choose_move(Board()) == 3


def test_cli_loads_pytorch_networks(tmp_path):
    path = tmp_path / "net.pt"
    recipe = tw.RECIPES["adam"]
    tw.save(tw.build(recipe), recipe, path)
    assert isinstance(make_player("neural", seed=0, model=path), tw.TorchNeuralPlayer)


@pytest.mark.parametrize("shape", [(2, 3, 1), (2, 4, 4), (2, 2, 3)])
def test_sliding_by_hand_matches_pytorch(shape):
    rng = np.random.default_rng(sum(shape))
    board_planes = (rng.random((2, 6, 7)) < 0.3).astype(float)
    weights = rng.normal(size=shape)
    np.testing.assert_allclose(
        conv.slide(board_planes, weights, 0.1), conv.slide_with_torch(board_planes, weights, 0.1)
    )


def test_worked_example_finds_the_one_stack():
    board = Board.from_moves(conv.EXAMPLE_MOVES)
    out = conv.slide(conv.planes(board), conv.STACK_WEIGHTS, conv.STACK_BIAS)
    assert out.shape == (4, 7)
    assert out[0, 2] == 1 and out.sum() == 1  # column 3, rows 1-3, and nowhere else
