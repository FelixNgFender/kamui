import pytest

import kamui.nn as nn
from kamui.core import Value


def test_softmax():
    vals = [Value(1), Value(2), Value(3)]
    probs = nn.softmax(vals)
    total = sum(p.data for p in probs)
    assert pytest.approx(total, 0.0001) == 1.0
    assert all(p.data > 0 for p in probs)


def test_softmax_equal_inputs():
    vals = [Value(0), Value(0), Value(0)]
    probs = nn.softmax(vals)
    assert all(pytest.approx(p.data, 0.0001) == 1 / 3 for p in probs)


def test_cross_entropy_loss_one_hot():
    logits = [Value(1), Value(2), Value(3)]
    label = [Value(0), Value(0), Value(1)]
    loss = nn.cross_entropy_loss(logits, label)
    assert loss.data > 0


def test_cross_entropy_loss_soft_label():
    logits = [Value(1), Value(2), Value(3)]
    label = [Value(0.2), Value(0.3), Value(0.5)]
    loss = nn.cross_entropy_loss(logits, label)
    assert loss.data > 0


def test_cross_entropy_loss_invalid():
    with pytest.raises(ValueError):
        nn.cross_entropy_loss([Value(1), Value(2)], [Value(0), Value(1), Value(0)])


def test_one_hot():
    assert [e.data for e in nn.one_hot(Value(3), num_classes=4)] == [0, 0, 0, 1]


def test_one_hot_invalid():
    with pytest.raises(ValueError):
        nn.one_hot(Value(1.0), num_classes=4)
    with pytest.raises(ValueError):
        nn.one_hot(Value(-1), num_classes=4)
    with pytest.raises(ValueError):
        nn.one_hot(Value(4), num_classes=1)
