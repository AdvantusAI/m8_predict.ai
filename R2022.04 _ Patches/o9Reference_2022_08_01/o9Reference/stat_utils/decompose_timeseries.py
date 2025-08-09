"""
Version : 1.0.0
Maintained by : dpref@o9solutions.com
"""
import logging

import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

from o9Reference.common_utils.function_timer import timed

# import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("o9_logger")


@timed
def decompose_timeseries(data: np.ndarray, frequency: str = "Monthly"):
    """
    Function to decompose a time series data into three components i.e., seasonal, trend and residual
    """
    assert isinstance(data, np.ndarray), "data should be np array type ..."

    empty_array = np.array([])
    function_name = "correct_outlier"

    if data is None or len(data) == 0:
        logger.info("inside function {}: data is empty".format(function_name))
        logger.info(
            "inside function {}: returning empty data".format(function_name)
        )
        return empty_array, empty_array, empty_array

    if frequency == "Monthly":
        if data.size < 24:
            logger.info(
                "inside function {}: Atleast 2 data cycles are required for determining seasonality...".format(
                    function_name
                )
            )
            logger.info(
                "inside function {}: Returning undecomposed data".format(
                    function_name
                )
            )
            return empty_array, empty_array, data
        res = seasonal_decompose(
            data, model="additive", period=12, extrapolate_trend="freq"
        )
    elif frequency == "Weekly":
        if data.size < 104:
            logger.info(
                "inside function {}: Atleast 2 data cycles are required for determining seasonality...".format(
                    function_name
                )
            )
            logger.info(
                "inside function {}: Returning undecomposed data".format(
                    function_name
                )
            )
            return empty_array, empty_array, data
        res = seasonal_decompose(
            data, model="additive", period=52, extrapolate_trend="freq"
        )
    elif frequency == "Quarterly":
        if data.size < 8:
            logger.info(
                "inside function {}: Atleast 2 data cycles are required for determining seasonality...".format(
                    function_name
                )
            )
            logger.info(
                "inside function {}: Returning undecomposed data".format(
                    function_name
                )
            )
            return empty_array, empty_array, data
        res = seasonal_decompose(
            data, model="additive", period=4, extrapolate_trend="freq"
        )
    else:
        logger.info(
            "inside function {}: unknown frequency specified".format(
                function_name
            )
        )
        logger.info(
            "inside function {}: Returning undecomposed data".format(
                function_name
            )
        )
        return empty_array, empty_array, data

    return res.seasonal, res.trend, res.resid


if __name__ == "__main__":
    # generating random 36 data points
    np.random.seed(0)
    actual = np.random.randint(100, 400, 25)

    # decomposing the data
    seasonal, trend, residual = decompose_timeseries(actual)

    print("seasonal : {}".format(seasonal))
    print("trend : {}".format(trend))
    print("residual : {}".format(residual))
