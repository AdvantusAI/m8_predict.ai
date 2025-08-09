"""
Version : 1.0.0
Maintained by : dpref@o9solutions.com
"""
import math

import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


def mean_absolute_percentage_error(y_true, y_pred):
    """function to calculate mean_absolute_percentage_error, excluding zeros in actuals
    Args:
        y_true (np array): array containing true values
        y_pred (np array): array container predictions
    Returns:
        float: mape value in percentage
    """

    # get sum of absolute errors
    sum_of_absolute_errors = np.sum(np.fabs(y_true - y_pred))
    sum_of_actuals = np.sum(y_true)

    # if sum of actuals is zero, return 100% error
    if sum_of_actuals == 0 and sum_of_absolute_errors > 0:
        return 100.00

    # if sum of actuals is zero, and predictions are also zero, return 0 (avoid divide by zero error)
    if sum_of_actuals == 0 and sum_of_absolute_errors == 0:
        return 0.00

    # default case
    mape = round(100 * sum_of_absolute_errors / sum_of_actuals, 2)

    return mape


def calculate_error(
    actuals: np.ndarray, forecasts: np.ndarray, metric: str
) -> float:
    """
    Calculates error based on the metric specified and returns the error value.
    Supported Metrics : RMSE, MAPE, MAE
    :param actuals:
    :param forecasts:
    :param metric:
    :return:
    """

    assert isinstance(
        actuals, np.ndarray
    ), "actuals should be np array type ..."
    assert isinstance(
        forecasts, np.ndarray
    ), "forecasts should be np array type ..."
    assert issubclass(actuals.dtype.type, np.integer) | issubclass(
        actuals.dtype.type, np.float
    ), "actuals should be either int/float ..."
    assert issubclass(forecasts.dtype.type, np.integer) | issubclass(
        forecasts.dtype.type, np.float
    ), "forecasts should be either int/float ..."
    assert (
        actuals.size == forecasts.size
    ), "both actuals and forecasts should have same size ..."

    # fill nas with zero
    actuals[np.isnan(actuals)] = 0
    forecasts[np.isnan(forecasts)] = 0

    # Initialize variable with zero
    error_value = np.nan

    if metric == "RMSE":
        error_value = math.sqrt(mean_squared_error(actuals, forecasts))
    elif metric == "MAPE":
        error_value = mean_absolute_percentage_error(actuals, forecasts)
    elif metric == "MAE":
        error_value = mean_absolute_error(actuals, forecasts)
    else:
        raise ValueError("Unknown metric {} ...".format(metric))

    return error_value


if __name__ == "__main__":
    import numpy as np

    # actuals = np.array([0, 0, 0])
    # forecast = np.array([0.0, 0, 0])

    actuals = np.array([120, np.nan, 120])
    forecast = np.array([120, 130, np.nan])

    # print("MAPE : {}".format(mean_absolute_percentage_error(actuals, forecast)))
    #
    print("RMSE : {}".format(calculate_error(actuals, forecast, "RMSE")))
    print("MAPE : {}".format(calculate_error(actuals, forecast, "MAPE")))
    print("MAE : {}".format(calculate_error(actuals, forecast, "MAE")))
