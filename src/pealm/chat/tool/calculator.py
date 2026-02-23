import signal
import warnings
from contextlib import contextmanager
from typing import Never


# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(seconds: int, formula):
    def timeout_handler(_, __) -> Never:  # noqa: ANN001
        msg = f"'{formula}': timed out after {seconds} seconds"
        raise Exception(msg)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    yield
    signal.alarm(0)


def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula), warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return eval(formula, {"__builtins__": {}}, {})
    except Exception:
        signal.alarm(0)
        # print(f"Warning: Failed to eval {formula}, exception: {e}") # it's ok ignore wrong calculator usage
        return None


def use_calculator(expr):
    """
    Evaluate a Python expression safely.
    Supports both math expressions and string operations like .count()
    """
    # Remove commas from numbers
    expr = expr.replace(",", "")

    # Check if it's a pure math expression (old behavior)
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # disallow power operator
            return None
        return eval_with_timeout(expr)

    # Check if it's a string operation we support
    # Allow: strings (single/double quotes), .count(), letters, numbers, spaces, parens
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    # Disallow dangerous patterns
    dangerous_patterns = [
        "__",
        "import",
        "exec",
        "eval",
        "compile",
        "open",
        "file",
        "input",
        "raw_input",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
    ]
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # Only allow .count() method for now (can expand later)
    if ".count(" not in expr:
        return None

    # Evaluate with timeout
    return eval_with_timeout(expr)
