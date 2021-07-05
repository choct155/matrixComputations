from time import time
from functools import wraps
from typing import Callable, TypeVar, List, Tuple, Any

A = TypeVar("A")
B = TypeVar("B")


def timed(f: Callable[[A], B]) -> Tuple[B, float]:
    @wraps(f)
    def wrapper(*args, **kwargs) -> Tuple[B, float]:
        start: float = time()
        out: B = f(*args, **kwargs)
        stop: float = time()
        elapsed: float = stop - start
        time_str: str = "{:.4f}".format(elapsed)
        pos_args: Tuple[Any, ...] = args
        kw_args: List[str] = [f"{k}={v}" for k, v in list(**kwargs)]
        msg: str = f"""
        Executed {f.__name__} with the following arguments:
        Positional: {pos_args}
        Keyword: {kw_args}
        Elapsed Time: {time_str} seconds
        """
        print(msg)
        return out, elapsed
    return wrapper
