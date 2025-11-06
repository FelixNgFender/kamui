import enum
import math
from typing import Any, Callable, Self


class Op(enum.StrEnum):
    NOP = ""
    ADD = "+"
    SUB = "-"
    MUL = "*"
    POW = "**{exponent}"
    RELU = "ReLu"
    EXP = "exp"
    LOG = "log"


class Value:
    def __init__(
        self,
        data: int | float,
        _operands: tuple["Value"] | tuple["Value", "Value"] | None = None,
        _op: Op | str = Op.NOP,
    ) -> None:
        self.data: int | float = data
        self._operands: set[Value] | None = (
            set(_operands) if _operands is not None else None
        )
        self._op: Op | str = _op
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
                out = Value(self.data + other.data, (self, other), Op.ADD)

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
                out = Value(self.data * other.data, (self, other), Op.MUL)

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
                    (self,),
                    Op.POW.format(exponent=other),
                )

                def _grad_fn():
                    self.grad += out.grad * other * self.data ** (other - 1)

                out.grad_fn = _grad_fn
                return out
            case _:
                raise TypeError("only supporting int/float powers for now")

    def relu(self) -> "Value":
        out = Value(0 if self.data < 0 else self.data, (self,), Op.RELU)

        def _grad_fn():
            self.grad += out.grad * (out.data > 0)

        out.grad_fn = _grad_fn
        return out

    def exp(self) -> "Value":
        out = Value(math.exp(self.data), (self,), Op.EXP)

        def _grad_fn():
            self.grad += out.grad * out.data

        out.grad_fn = _grad_fn
        return out

    def log(self) -> "Value":
        out = Value(math.log(self.data), (self,), Op.LOG)

        def _grad_fn():
            self.grad += out.grad * (1 / self.data)

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
