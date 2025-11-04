import enum
import dataclasses
from typing import Any, Callable, Self


class Op(enum.StrEnum):
    ADD = "+"
    SUB = "-"
    MUL = "*"
    POW = "**{exponent}"
    RELU = "ReLu"


@dataclasses.dataclass
class AddOperands:
    op = Op.ADD
    operands: tuple["Value", "Value"]


@dataclasses.dataclass
class MulOperands:
    op = Op.MUL
    operands: tuple["Value", "Value"]


@dataclasses.dataclass
class PowOperands:
    op: str
    operands: tuple["Value"]


@dataclasses.dataclass
class ReLuOperands:
    op = Op.RELU
    operands: tuple["Value"]


type Operands = AddOperands | MulOperands | PowOperands | ReLuOperands


class Value:
    def __init__(
        self,
        data: float,
        _operands: Operands | None = None,
    ) -> None:
        self.data: float = data
        self._operands: set[Value] | None = (
            set(_operands.operands) if _operands is not None else None
        )
        self.grad: float = 0.0
        self.grad_fn: Callable[[], None] = lambda: None

    def backward(self) -> None:
        topo: list[Value] = []
        visited: set[Value] = set()

        def _build_topo(v: Value):
            if v in visited:
                return
            visited.add(v)
            if v._operands:
                for child in v._operands:
                    _build_topo(child)
            topo.append(v)

        _build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node.grad_fn()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(data={self.data}, grad={self.grad})"

    def __add__(self, other: Any) -> "Value":
        match other:
            case int() | float():
                return self + Value(other)
            case Value():
                out = Value(self.data + other.data, AddOperands((self, other)))

                def _grad_fn():
                    self.grad += out.grad
                    other.grad += out.grad

                out.grad_fn = _grad_fn
                return out
            case _:
                raise TypeError(f"unsupported operand type {type(other).__name__}")

    def __mul__(self, other: Any) -> "Value":
        match other:
            case int() | float():
                return self * Value(other)
            case Value():
                out = Value(self.data * other.data, MulOperands((self, other)))

                def _grad_fn():
                    self.grad += out.grad * other.data
                    other.grad += out.grad * self.data

                out.grad_fn = _grad_fn
                return out
            case _:
                raise TypeError(f"unsupported operand type {type(other).__name__}")

    def __pow__(self, other: Any) -> "Value":
        match other:
            case int() | float():
                out = Value(
                    self.data**other,
                    PowOperands(Op.POW.format(exponent=other), (self,)),
                )

                def _grad_fn():
                    self.grad += out.grad * other * self.data ** (other - 1)

                out.grad_fn = _grad_fn
                return out
            case _:
                raise TypeError("only supporting int/float powers for now")

    def relu(self) -> "Value":
        out = Value(0 if self.data < 0 else self.data, ReLuOperands((self,)))

        def _grad_fn():
            self.grad += out.grad * (out.data > 0)

        out.grad_fn = _grad_fn
        return out

    def __neg__(self) -> "Value":
        return self * -1

    def __pos__(self) -> "Value":
        return self * 1

    def __radd__(self, other: Any) -> Self:
        return self + other

    def __sub__(self, other: Any) -> Self:
        return self + (-other)

    def __rsub__(self, other: Any) -> Self:
        return -self + other

    def __rmul__(self, other: Any) -> Self:
        return self * other

    def __truediv__(self, other: Any) -> Self:
        return self * other**-1

    def __rtruediv__(self, other: Any) -> Self:
        return self**-1 * other
