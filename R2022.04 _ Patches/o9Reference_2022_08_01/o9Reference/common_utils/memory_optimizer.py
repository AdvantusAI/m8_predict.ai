"""
Version : 1.0.0
Maintained by : dpref@o9solutions.com
"""
import logging

import numpy as np
import pandas as pd
from pandas.api.types import is_categorical_dtype
from pandas.api.types import is_datetime64_any_dtype as is_datetime

logger = logging.getLogger("o9_logger")
logging.basicConfig(level=logging.INFO)


def reduce_memory_usage(X: pd.DataFrame) -> pd.DataFrame:
    """
    Checks min and max values in each column of a dataframe, assigns datatype in a conservative manner.

    Int/Float : Check min and max, convert
    String columns : converts to category if there are less than 20 items
    Datetime columns : left intact
    Categorical columns : left intact

    :param X: pandas dataframe
    :return:
    """
    if X.empty:
        return pd.DataFrame()

    start_mem = X.memory_usage().sum() / 1024**2
    logger.info("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in X.columns:
        if is_datetime(X[col]) or is_categorical_dtype(X[col]):
            continue
        col_type = X[col].dtype

        if col_type != object:
            c_min = X[col].min()
            c_max = X[col].max()
            if (str(col_type)[:3] == "int") or (str(col_type)[:4] == "uint"):
                if (
                    c_min > np.iinfo(np.int8).min
                    and c_max < np.iinfo(np.int8).max
                ):
                    X[col] = X[col].astype(np.int8)
                elif (
                    c_min > np.iinfo(np.int16).min
                    and c_max < np.iinfo(np.int16).max
                ):
                    X[col] = X[col].astype(np.int16)
                elif (
                    c_min > np.iinfo(np.int32).min
                    and c_max < np.iinfo(np.int32).max
                ):
                    X[col] = X[col].astype(np.int32)
                elif (
                    c_min > np.iinfo(np.int64).min
                    and c_max < np.iinfo(np.int64).max
                ):
                    X[col] = X[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    X[col] = X[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    X[col] = X[col].astype(np.float32)
                else:
                    X[col] = X[col].astype(np.float64)
        else:
            logger.info(
                "Number of categories in column {} :".format(X[col].nunique())
            )
            if X[col].nunique() < 20:
                # convert to category if there are less than 20 unique items
                X[col] = X[col].astype("category")
            else:
                # leave datatype as string itself
                pass

    end_mem = X.memory_usage().sum() / 1024**2
    logger.info(
        "Memory usage after optimization is: {:.2f} MB".format(end_mem)
    )
    logger.info(
        "Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem)
    )
    return X


if __name__ == "__main__":
    import pandas as pd

    df = pd.DataFrame(
        np.random.randint(0, 100, size=(1000000, 4)), columns=list("ABCD")
    )
    result = reduce_memory_usage(df)
    print(result)
