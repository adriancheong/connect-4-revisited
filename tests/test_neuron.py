import numpy as np
import pytest

from connect4.numpy_nn import neuron


def test_sigmoid_squashes_into_0_to_1():
    assert neuron.sigmoid(0) == 0.5
    assert neuron.sigmoid(10) > 0.99
    assert neuron.sigmoid(-10) < 0.01


def test_worked_example_forward_pass():
    prediction = neuron.predict(neuron.EXAMPLE_WEIGHTS, neuron.EXAMPLE_BIAS, neuron.EXAMPLE_X)
    assert prediction == pytest.approx(0.5)
    assert neuron.loss(prediction, neuron.EXAMPLE_TARGET) == pytest.approx(0.25)


def test_worked_example_first_step():
    weights, bias = neuron.train_step(
        neuron.EXAMPLE_WEIGHTS, neuron.EXAMPLE_BIAS, neuron.EXAMPLE_X, neuron.EXAMPLE_TARGET, 1.0
    )
    assert weights == pytest.approx([0.45, -0.275])
    assert bias == pytest.approx(0.25)
    assert neuron.predict(weights, bias, neuron.EXAMPLE_X) == pytest.approx(0.637, abs=1e-3)


@pytest.mark.parametrize("seed", range(5))
def test_chain_rule_matches_nudging(seed):
    rng = np.random.default_rng(seed)
    weights, bias = rng.normal(size=3), rng.normal()
    x, target = rng.normal(size=3), rng.uniform()
    nudged_w, nudged_b = neuron.gradients_by_nudging(weights, bias, x, target)
    exact_w, exact_b = neuron.gradients(weights, bias, x, target)
    assert exact_w == pytest.approx(nudged_w, abs=1e-5)
    assert exact_b == pytest.approx(nudged_b, abs=1e-5)


def test_a_neuron_learns_or():
    weights, bias, history = neuron.train(neuron.INPUTS, neuron.OR_TARGETS)
    predictions = [neuron.predict(weights, bias, x) for x in neuron.INPUTS]
    assert np.round(predictions).tolist() == neuron.OR_TARGETS.tolist()
    assert history[-1].loss < 0.01 < history[0].loss


def test_a_single_neuron_cannot_learn_xor():
    weights, bias, history = neuron.train(neuron.INPUTS, neuron.XOR_TARGETS)
    predictions = [neuron.predict(weights, bias, x) for x in neuron.INPUTS]
    assert np.round(predictions).tolist() != neuron.XOR_TARGETS.tolist()
    assert history[-1].loss > 0.2
