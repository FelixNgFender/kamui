import abc
import random
from typing import Iterable, Sequence

import kamui as km


class Module(abc.ABC):
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0

    @abc.abstractmethod
    def parameters(self) -> list[km.Value]:
        pass

    @abc.abstractmethod
    def __call__(self, xs: Iterable[km.Value]) -> Sequence[km.Value]:
        pass


class Neuron(Module):
    def __init__(self, nin: int, nonlin: bool = True) -> None:
        """Creates a new neuron with `nin` inputs."""
        self.weights: list[km.Value] = [
            km.Value(random.uniform(-1.0, 1.0)) for _ in range(nin)
        ]
        self.bias: km.Value = km.Value(0)
        self.nonlin: bool = nonlin

    def __call__(self, xs: Iterable[km.Value]) -> list[km.Value]:
        act = sum((wi * xi for wi, xi in zip(self.weights, xs)), self.bias)
        return [act.relu() if self.nonlin else act]

    def parameters(self) -> list[km.Value]:
        return [*self.weights, self.bias]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.weights)})"


class Layer(Module):
    def __init__(self, nin: int, nout: int, **kwargs) -> None:
        """Creates a new layer with `nout` neurons, each with `nin` inputs."""
        self.neurons: list[Neuron] = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, xs: Iterable[km.Value]) -> list[km.Value]:
        return [n(xs)[0] for n in self.neurons]

    def parameters(self) -> list[km.Value]:
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of {len(self.neurons)} neurons [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, nin: int, nouts: list[int]) -> None:
        """
        Creates a new multi-layer perceptron with `nin` neurons in the input layer
        and `nouts` as the list of number of neurons in each subsequent layer.

        Invariant: the number of neurons in layer n equals to the dimensionality
        of a neuron in layer n + 1
        """
        sz: list[int] = [nin] + nouts
        self.layers: list[Layer] = [
            Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def __call__(self, xs: Iterable[km.Value]) -> list[km.Value]:
        for layer in self.layers:
            xs = layer(xs)
        return xs  # type: ignore

    def parameters(self) -> list[km.Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP of {len(self.layers)} layers [\n{'\n'.join(str(layer) for layer in self.layers)}\n]"


def softmax(logits: Iterable[km.Value]) -> list[km.Value]:
    """
    Converts a list of `n` Value objects into a probability distribution over `n` possible outcomes.
    """
    # log-sum-exp trick for numerical stability
    # https://leimao.github.io/blog/LogSumExp/
    max_x = max(logits, key=lambda x: x.data)
    exps = [(x - max_x).exp() for x in logits]
    sum_exps = sum(exps, km.Value(0))
    return [e / sum_exps for e in exps]


def cross_entropy_loss(pred: Sequence[km.Value], label: Sequence[km.Value]) -> km.Value:
    """
    Converts the input logits `pred` into a probability distribution via softmax and apply the cross
    entropy loss function against the ground truth `label` probability distribution.
    """
    if len(pred) != len(label):
        raise ValueError("pred and label must have the same length")

    pred_softmax = softmax(pred)
    return -sum((yi * pi.log() for yi, pi in zip(label, pred_softmax)), km.Value(0))


def one_hot(input: km.Value, *, num_classes: int) -> list[km.Value]:
    """
    Converts a integer Value object into a one-hot encoded vector.
    """
    if not isinstance(input.data, int) or input.data < 0 or input.data >= num_classes:
        raise ValueError(
            "Input value must be an integer in the range [0, num_classes - 1]"
        )

    return [km.Value(1) if i == input.data else km.Value(0) for i in range(num_classes)]
