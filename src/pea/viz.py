from typing import Literal

import pea

try:
    import graphviz
except ImportError:
    msg = "graphviz is required to use the visualization features. Please install it via 'uv sync'."
    raise RuntimeError(msg) from None


def trace(
    root: pea.Value,
) -> tuple[set[pea.Value], set[tuple[pea.Value, pea.Value]]]:
    nodes: set[pea.Value] = set()
    edges: set[tuple[pea.Value, pea.Value]] = set()

    def build(v: pea.Value) -> None:
        if v not in nodes:
            nodes.add(v)
            if not v.operands:
                return
            for child in v.operands:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(
    root: pea.Value,
    fmt: Literal["svg", "png"] = "svg",
    rankdir: Literal["LR", "TB"] = "LR",
) -> graphviz.Digraph:
    nodes, edges = trace(root)
    dot = graphviz.Digraph(format=fmt, graph_attr={"rankdir": rankdir})  # , node_attr={'rankdir': 'TB'})

    for n in nodes:
        dot.node(
            name=str(id(n)),
            label=f"{{ data {n.data:.4f} | grad {n.grad:.4f} }}",
            shape="record",
        )
        if n.operands:
            dot.node(name=str(id(n)) + n.op, label=n.op)
            dot.edge(str(id(n)) + n.op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2.op)

    return dot
