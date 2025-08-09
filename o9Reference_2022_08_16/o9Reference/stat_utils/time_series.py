"""
Version : 1.0.0
Maintained by : dpref@o9solutions.com
"""
import logging
import statistics
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger("o9_logger")

from sklearn.linear_model import LinearRegression


def get_variability(values: np.ndarray) -> float:
    """
    Get Coefficient of Variation (COV) from np array of values provided.
    Value will be returned as float type.
    """
    assert isinstance(values, np.ndarray), "values should be np array type ..."
    assert not np.isnan(values).any(), "Input cannot contain None ..."

    cov = 0
    try:
        if sum(values) != 0 and len(values) > 1:
            cov = np.std(values, ddof=1) / np.mean(values)
    except Exception as e:
        logger.exception(e)
    return cov


def calculate_trend(values: np.ndarray) -> float:
    """
    Calculates linear trend - slope of linear regression line.
    :param values:
    :return:
    """
    assert isinstance(values, np.ndarray), "values should be np array type ..."
    assert len(values) > 0, "values cannot be empty ..."
    assert not np.isnan(values).any(), "Input cannot contain None ..."

    trend_value = 0
    try:
        x = np.array([x for x in range(1, len(values) + 1)]).reshape(-1, 1)
        linear_model = LinearRegression().fit(x, values)
        trend_value = linear_model.coef_[0]
    except Exception as e:
        logger.exception(e)
    return trend_value


def get_mean_stddev(values: List[float], std_dev_col, mean_col) -> pd.Series:
    # Initialize with zero
    result = {std_dev_col: 0, mean_col: 0}
    if len(values) > 1:
        result = {
            std_dev_col: statistics.stdev(values),
            mean_col: statistics.mean(values),
        }
    return pd.Series(result, index=[std_dev_col, mean_col])


if __name__ == "__main__":
    # result = calculate_trend([])
    # print(result)

    # creating a test dataset
    Data = pd.DataFrame(
        {
            "Week": [
                "W01-21",
                "W02-21",
                "W03-21",
                "W04-21",
                "W05-21",
                "W06-21",
                "W07-21",
                "W08-21",
                "W09-21",
                "W10-21",
                "W11-21",
                "W12-21",
            ],
            "Actual": [
                300,
                100,
                425,
                75,
                100,
                243,
                896,
                172,
                1108,
                673,
                900,
                345,
            ],
        }
    )
    print(calculate_trend(Data["Actual"].to_numpy()))

    print(get_variability(Data["Actual"].to_numpy()))
