"""
Version : 1.0.0
Maintained by : dpref@o9solutions.com
"""
import numpy as np


def get_moving_avg_forecast(
    data: np.ndarray, moving_avg_periods: int, forecast_horizon: int
) -> np.array:
    """
    Calculates sliding window forecast with n periods, uses history and forecast available to produce future values iteratively
    """
    assert isinstance(data, np.ndarray), "train should be of type np.array ..."
    assert isinstance(
        forecast_horizon, int
    ), "forecast_horizon should be of type integer ..."
    assert isinstance(
        moving_avg_periods, int
    ), "moving_avg_periods should be of type integer ..."

    assert (
        forecast_horizon > 0
    ), "forecast_horizon should be greater than 0 ..."
    assert (
        moving_avg_periods > 0
    ), "moving_avg_periods should be greater than 0 ..."

    # check for empty array
    if data.size == 0:
        return np.array([])

    # Fill nas with zero
    data[np.isnan(data)] = 0

    # Not enough data, take mean of all available points as forecast
    if data.size == 1 or data.size <= moving_avg_periods:
        return np.repeat(np.mean(data), forecast_horizon)

    # create copy of array to avoid modification to source data
    history_values = np.array(data, copy=True).tolist()

    forecasts = []
    for the_period in range(0, forecast_horizon):
        # take avg of last n periods
        last_n_periods_avg = np.mean(
            history_values[-moving_avg_periods:]
        ).round(decimals=2)

        # append to forecasts
        forecasts.append(last_n_periods_avg)

        # append to source list for next iteration
        history_values.append(last_n_periods_avg)

    assert (
        len(forecasts) == forecast_horizon
    ), "mismatch in output size, check source data and calculation ..."

    return np.array(forecasts)


# if __name__ == "__main__":
#     from o9Reference.stat_utils.get_moving_avg_forecast import get_moving_avg_forecast
#     import numpy as np
#     values = [100, 120, 150, 90, 80, 70, 30, 29, 40, 55, 61, 63, 84]
#     ma_periods = 4
#     fh = 12
#     print(
#         get_moving_avg_forecast(
#             data=np.array(values),
#             moving_avg_periods=ma_periods,
#             forecast_horizon=fh,
#         )
#     )
