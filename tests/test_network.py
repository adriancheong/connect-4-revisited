import numpy as np
import pytest

from connect4.numpy_nn import network as nw


def test_new_network_shapes():
    net = nw.new_network(inputs=2, hidden=2, outputs=1)
    assert net.hidden_weights.shape == (2, 2)
    assert net.hidden_biases.shape == (2,)
    assert net.output_weights.shape == (2, 1)
    assert net.output_biases.shape == (1,)
    assert net.dial_count() == 9
    assert nw.new_network(inputs=84, hidden=32, outputs=7).dial_count() == 2951


def test_worked_example_forward_pass():
    fwd = nw.forward(nw.EXAMPLE, nw.EXAMPLE_CARD)
    assert fwd.hidden[0] == pytest.approx([0.5987, 0.4502], abs=1e-4)
    assert fwd.guesses[0, 0] == pytest.approx(0.5657, abs=1e-4)


def test_worked_example_slopes():
    s = nw.named_dials(nw.slopes(nw.EXAMPLE, nw.EXAMPLE_CARD, nw.EXAMPLE_ANSWER))
    assert s["bO"] == pytest.approx(-0.2134, abs=1e-4)
    assert s["wOA"] == pytest.approx(-0.1278, abs=1e-4)
    assert s["wA1"] == pytest.approx(-0.0256, abs=1e-4)
    assert s["wB1"] == pytest.approx(+0.0158, abs=1e-4)
    # x2 is 0 on this card, so the dials attached to it get no blame.
    assert s["wA2"] == 0
    assert s["wB2"] == 0


@pytest.mark.parametrize("seed", range(4))
def test_backpropagation_matches_nudging(seed):
    rng = np.random.default_rng(seed)
    inputs, hidden, outputs, cards = 3, 4, 2, 5
    net = nw.new_network(inputs, hidden, outputs, seed=seed)
    net = nw.Network(
        net.hidden_weights,
        rng.normal(size=hidden),  # non-zero biases, so they are tested too
        net.output_weights,
        rng.normal(size=outputs),
    )
    x = rng.normal(size=(cards, inputs))
    answers = rng.uniform(size=(cards, outputs))
    exact = nw.slopes(net, x, answers)
    nudged = nw.slopes_by_nudging(net, x, answers)
    for e, n in zip(exact.dials(), nudged.dials()):
        assert e == pytest.approx(n, abs=1e-8)


def test_network_learns_xor():
    net, history = nw.train(nw.XOR_INPUTS, nw.XOR_ANSWERS, hidden=2, seed=1)
    guesses = nw.predict(net, nw.XOR_INPUTS)[:, 0]
    assert np.round(guesses).tolist() == nw.XOR_ANSWERS.tolist()
    assert history[-1] < 0.001 < history[0]


def test_some_random_starts_get_stuck_with_two_hidden_neurons():
    _, history = nw.train(nw.XOR_INPUTS, nw.XOR_ANSWERS, hidden=2, seed=6)
    assert history[-1] == pytest.approx(0.125, abs=0.005)


@pytest.mark.parametrize("seed", [6, 13, 18])
def test_three_hidden_neurons_learn_xor_where_two_got_stuck(seed):
    net, _ = nw.train(nw.XOR_INPUTS, nw.XOR_ANSWERS, hidden=3, seed=seed)
    guesses = nw.predict(net, nw.XOR_INPUTS)[:, 0]
    assert np.round(guesses).tolist() == nw.XOR_ANSWERS.tolist()


def test_identical_hidden_neurons_stay_identical_twins():
    twins = nw.Network(
        hidden_weights=np.full((2, 2), 0.5),
        hidden_biases=np.full(2, 0.5),
        output_weights=np.full((2, 1), 0.5),
        output_biases=np.full(1, 0.5),
    )
    net, history = nw.train(nw.XOR_INPUTS, nw.XOR_ANSWERS, start=twins)
    assert net.hidden_weights[:, 0] == pytest.approx(net.hidden_weights[:, 1])
    assert net.hidden_biases[0] == pytest.approx(net.hidden_biases[1])
    assert history[-1] > 0.1  # acting as one neuron, it can't learn XOR
