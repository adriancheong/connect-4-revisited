import numpy as np
import pytest

torch = pytest.importorskip("torch")

from connect4.numpy_nn import network as np_nn  # noqa: E402
from connect4.torch_nn import xor  # noqa: E402


def test_round_trip_keeps_every_dial():
    net = np_nn.new_network(3, 4, 2, seed=5)
    back = xor.to_numpy(xor.from_numpy(net))
    for a, b in zip(net.dials(), back.dials()):
        np.testing.assert_array_equal(a, b)


def test_same_guesses_as_stage_2():
    net = np_nn.new_network(2, 2, 1, seed=1)
    with torch.no_grad():
        guesses = xor.from_numpy(net)(torch.as_tensor(np_nn.XOR_INPUTS)).numpy()
    np.testing.assert_allclose(guesses, np_nn.predict(net, np_nn.XOR_INPUTS), atol=1e-12)


def test_worked_example_slopes_match_stage_2():
    card, answer = np_nn.EXAMPLE_CARD, np_nn.EXAMPLE_ANSWER
    by_hand = np_nn.named_dials(np_nn.slopes(np_nn.EXAMPLE, card, answer))
    by_torch = np_nn.named_dials(xor.slopes(xor.from_numpy(np_nn.EXAMPLE), card, answer))
    assert by_torch["bO"] == pytest.approx(-0.2134, abs=1e-4)
    for name in by_hand:
        assert by_torch[name] == pytest.approx(by_hand[name], abs=1e-12)


@pytest.mark.parametrize("seed", range(4))
def test_automatic_slopes_match_backpropagation_by_hand(seed):
    rng = np.random.default_rng(seed)
    net = np_nn.new_network(3, 4, 2, seed=seed)
    inputs, answers = rng.random((5, 3)), rng.random((5, 2))
    by_hand = np_nn.slopes(net, inputs, answers)
    by_torch = xor.slopes(xor.from_numpy(net), inputs, answers)
    for a, b in zip(by_hand.dials(), by_torch.dials()):
        np.testing.assert_allclose(b, a, atol=1e-12)


def test_training_matches_stage_2_step_by_step():
    start = np_nn.new_network(2, 2, 1, seed=1)
    by_hand, history_by_hand = np_nn.train(
        np_nn.XOR_INPUTS, np_nn.XOR_ANSWERS, learning_rate=2.0, rounds=500, start=start
    )
    model = xor.from_numpy(start)
    history_by_torch = xor.train(model, np_nn.XOR_INPUTS, np_nn.XOR_ANSWERS, 2.0, 500)
    np.testing.assert_allclose(history_by_torch, history_by_hand, atol=1e-10)
    for a, b in zip(by_hand.dials(), xor.to_numpy(model).dials()):
        np.testing.assert_allclose(b, a, atol=1e-10)


def test_learns_xor():
    model = xor.from_numpy(np_nn.new_network(2, 2, 1, seed=1))
    history = xor.train(model, np_nn.XOR_INPUTS, np_nn.XOR_ANSWERS, 2.0, 10_000)
    assert history[-1] < 0.001
    with torch.no_grad():
        guesses = model(torch.as_tensor(np_nn.XOR_INPUTS)).numpy().ravel()
    assert np.round(guesses).tolist() == np_nn.XOR_ANSWERS.tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no graphics card PyTorch can use")
def test_graphics_card_gives_the_same_slopes():
    net = np_nn.new_network(3, 4, 2, seed=0)
    rng = np.random.default_rng(0)
    inputs, answers = rng.random((5, 3)), rng.random((5, 2))
    cpu = xor.slopes(xor.from_numpy(net), inputs, answers)
    model = xor.from_numpy(net).to("cuda")
    model.zero_grad()
    guesses = model(torch.as_tensor(inputs, device="cuda"))
    torch.nn.functional.mse_loss(guesses, torch.as_tensor(answers, device="cuda")).backward()
    gpu = xor.to_numpy(model, use_slopes=True)
    for a, b in zip(cpu.dials(), gpu.dials()):
        np.testing.assert_allclose(b, a, atol=1e-12)
