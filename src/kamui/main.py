import kamui as km


def main():
    a = km.Value(2.0)
    b = km.Value(-3.0)
    c = km.Value(10.0)
    d = a * b + c.relu()
    d.backward()
    return d.grad
