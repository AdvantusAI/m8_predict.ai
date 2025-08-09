"""
Version : 1.0.0
Maintained by : dpref@o9solutions.com
"""
import logging
import time
from datetime import datetime
from functools import wraps

logger = logging.getLogger("o9_logger")


def timed(func):
    """Decorator for timing function calls

    Args:
        func (function): function as input
    """

    @wraps(func)
    def _wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        logger.info(
            f"Completed {func.__name__} in {time.time() - start:.3f} sec"
        )
        return res

    return _wrapper


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("o9_logger")

    @timed
    def my_function() -> None:
        time.sleep(2)

    # call function
    my_function()
