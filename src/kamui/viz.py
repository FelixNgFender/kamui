from typing import Literal

import kamui as km

try:
    import graphviz
except ImportError:
    raise RuntimeError(
        "graphviz is required to use the visualization features. Please install it via 'uv sync'."
    )


def trace(root: km.Value) -> tuple[set[km.Value], set[tuple[km.Value, km.Value]]]:
    nodes: set[km.Value] = set()
    edges: set[tuple[km.Value, km.Value]] = set()

    def build(v: km.Value):
        if v not in nodes:
            nodes.add(v)
            if not v._operands:
                return
            for child in v._operands:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(
    root: km.Value,
    format: Literal["svg", "png"] = "svg",
    rankdir: Literal["LR", "TB"] = "LR",
) -> graphviz.Digraph:
    nodes, edges = trace(root)
    dot = graphviz.Digraph(
        format=format, graph_attr={"rankdir": rankdir}
    )  # , node_attr={'rankdir': 'TB'})

    for n in nodes:
        dot.node(
            name=str(id(n)),
            label="{ data %.4f | grad %.4f }" % (n.data, n.grad),
            shape="record",
        )
        if n._operands:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot
