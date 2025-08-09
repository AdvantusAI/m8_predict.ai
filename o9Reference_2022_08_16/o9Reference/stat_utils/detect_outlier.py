"""
Version : 1.0.0
Maintained by : dpref@o9solutions.com
"""
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("o9_logger")
import numpy as np
import pandas as pd


def detect_outlier(
    data: np.ndarray, upper_bound: np.ndarray, lower_bound: np.ndarray
):
    """
    Detect outlier in a Time sorted list containing input numeric data
    """

    assert isinstance(data, np.ndarray), "data should be np array type ..."
    assert isinstance(
        upper_bound, np.ndarray
    ), "upper_bound should be np array type ..."
    assert isinstance(
        lower_bound, np.ndarray
    ), "upper_bound should be np array type ..."
    assert len(data) == len(
        upper_bound
    ), "both data and upper_bound should have same size ..."
    assert len(data) == len(
        lower_bound
    ), "both data and lower_bound should have same size ..."

    function_name = "detect_outlier"

    # empty input check
    if data is None or len(data) == 0:
        logger.info("inside function {}: data is empty".format(function_name))
        logger.info(
            "inside function {}: returning empty data".format(function_name)
        )
        return []

    if upper_bound is None or len(upper_bound) == 0:
        logger.info(
            "inside function {}: Upper bound is empty".format(function_name)
        )
        logger.info(
            "inside function {}: returning empty data".format(function_name)
        )
        return []

    if lower_bound is None or len(lower_bound) == 0:
        logger.info(
            "inside function {}: Lower bound is empty".format(function_name)
        )
        logger.info(
            "inside function {}: returning empty data".format(function_name)
        )
        return []
    # column names
    data_col = "data"
    upper_bound_col = "Upper Limit"
    lower_bound_col = "Lower Limit"
    outlier_col = "outlier"

    # creating dataframe
    df = pd.DataFrame(
        np.hstack((data[:, None], upper_bound[:, None], lower_bound[:, None])),
        columns=[data_col, upper_bound_col, lower_bound_col],
    )

    logger.info("detecting outliers...")
    df[outlier_col] = df[data_col].gt(df[upper_bound_col]) | df[data_col].lt(
        df[lower_bound_col]
    )

    return df[outlier_col].to_numpy()


if __name__ == "__main__":
    np.random.seed(0)  # seed for reproducibility
    actual = np.random.randint(100, 400, 12)
    upper_bound = np.random.randint(300, 450, 12)
    lower_bound = np.random.randint(0, 150, 12)

    tuples = list(zip(actual, upper_bound, lower_bound))
    df_tuple = pd.DataFrame(tuples, columns=["actual", "up_bound", "lo_bound"])
    output = detect_outlier(actual, upper_bound, lower_bound)
    df_tuple["output"] = output

    print(df_tuple)
