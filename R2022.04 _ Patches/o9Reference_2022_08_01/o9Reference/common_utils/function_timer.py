"""
Version : 1.0.0
Maintained by : dpref@o9solutions.com
"""
import logging
import time
from functools import wraps

import pandas as pd

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

        # if function is called with keyword arguments, we just need to query for df_keys
        # check if kwargs is empty
        if bool(kwargs):
            # extract slice information, assign empty dict as default value
            slice_info = kwargs.get("df_keys", {})
        else:
            # check if any of the args are of the type dictionary, there could be more than one
            dicts_in_args = [x for x in args if type(x) == dict]
            if dicts_in_args:
                # assuming df_keys will be passed as the last argument to the function, take last element from list
                slice_info = dicts_in_args[-1]
            else:
                slice_info = {}

        logger.info(
            f"Completed {func.__name__} in {time.time() - start:.3f} sec for slice : {slice_info}"
        )
        return res

    return _wrapper


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("o9_logger")

    @timed
    def my_function(df, df_keys) -> None:
        time.sleep(2)

    my_dict = {"Location.[Stat Location]": "1001"}
    df = pd.DataFrame()
    # call function with args
    my_function(df, my_dict)

    # call function with kwargs
    my_function(df=df, df_keys=my_dict)
