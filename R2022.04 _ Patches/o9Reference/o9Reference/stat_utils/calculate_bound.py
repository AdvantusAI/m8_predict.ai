"""
Version : 1.0.0
Maintained by : dpref@o9solutions.com
"""
import logging

logging.basicConfig(level=logging.INFO)
from statsmodels.tsa.seasonal import STL
from tsmoothie.smoother import SpectralSmoother

logger = logging.getLogger("o9_logger")
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def calculate_bound(
    data: np.ndarray,
    UpperThreshold: float,
    LowerThreshold: float,
    time_series_freq: int,
    method: str = "No Cleansing",
    rolling_window_size: int = None,
):
    """
    Detect outlier in a Time sorted list containing input numeric data
    """

    assert isinstance(data, np.ndarray), "data should be np array type ..."
    assert isinstance(method, str), "method should be string type ..."
    assert isinstance(
        UpperThreshold, float
    ), "UpperThreshold should be float type ..."
    assert isinstance(
        LowerThreshold, float
    ), "LowerThreshold should be float type ..."

    default_threshold = 3

    if data is None or len(data) == 0:
        return [], [], []

    if method == "No Cleansing":
        logger.info(
            "Outlier Cleansing method not assigned. Data not cleansed..."
        )
        return data, data, data

    if UpperThreshold is None:
        UpperThreshold = default_threshold
        logger.info(
            "Assigning Upper Threshold a default value : {}".format(
                UpperThreshold
            )
        )

    if LowerThreshold is None:
        LowerThreshold = default_threshold
        logger.info(
            "Assigning Lower Threshold a default value : {}".format(
                LowerThreshold
            )
        )

    if rolling_window_size is None:
        rolling_window_size = time_series_freq // 2

    data_col = "actual"
    upper_bound_col = "Upper Bound"
    lower_bound_col = "Lower Bound"
    moving_avg_col = "moving avg"
    df = pd.DataFrame(data, columns=[data_col], copy=True)
    if method == "Fixed Sigma":
        # calculating mean
        mean = df.mean()

        # calculating standard deviation
        std_dev = df.std() if df.size > 1 else df.iloc[0]

        # calculating thresholds
        UL = mean + UpperThreshold * std_dev.iloc[0]
        LL = mean - LowerThreshold * std_dev.iloc[0]

        return (
            np.full(len(df), UL),
            np.full(len(df), LL),
            np.full(len(df), mean),
        )

    elif method == "Rolling Sigma":

        # calculating mean
        df[moving_avg_col] = (
            df.iloc[:, 0].rolling(window=rolling_window_size).mean().shift(1)
        )

        # back filling the initial nan's
        df[moving_avg_col].fillna(method="backfill", inplace=True)

        if len(df) <= rolling_window_size:
            df[moving_avg_col].fillna(df[data_col].mean(), inplace=True)

        print("df moving avg column: {}".format(df[moving_avg_col]))
        # calculating standard deviation
        std_dev = df.std() if len(df) > 1 else df.iloc[0]

        # calculating thresholds
        upper_margin = UpperThreshold * std_dev.iloc[0]
        lower_margin = LowerThreshold * std_dev.iloc[0]

        df[upper_bound_col] = df[moving_avg_col] + upper_margin
        df[lower_bound_col] = df[moving_avg_col] - lower_margin

        return (
            df[upper_bound_col].to_numpy(),
            df[lower_bound_col].to_numpy(),
            df[moving_avg_col].to_numpy(),
        )

    elif method == "Seasonal IQR":

        n = len(df)
        missng = df[data_col].isna()
        nmiss = missng.sum()
        if nmiss > 0:
            df[data_col] = df[data_col].interpolate(method="linear")

        strength_threshold = 0.6

        # frequency can be monthly(12), weekly(52), quarterly(4)
        if time_series_freq > 1 and n >= 2 * time_series_freq:
            # seasonal parameter requires 13 instead of 12 for monthly, 53 instead of 52 for weekly etc
            decomposed_df = STL(
                df[data_col],
                period=time_series_freq,
                seasonal=time_series_freq + 1,
            ).fit()
            rem = decomposed_df.resid
            detrend = df[data_col] - decomposed_df.trend
            strength = 1 - np.var(rem) / np.var(detrend)
            if strength >= strength_threshold:
                df[data_col] = df[data_col] - decomposed_df.seasonal

        # operate smoothing
        smoother = SpectralSmoother(
            smooth_fraction=0.7, pad_len=rolling_window_size
        )
        smoother.smooth(df[data_col])

        df[data_col] = smoother.smooth_data[0]

        # calculating quartiles
        q1 = np.percentile(df[data_col], 25)
        q3 = np.percentile(df[data_col], 75)

        iqr = q3 - q1

        limitsthresholdIQR = [
            q1 - LowerThreshold * iqr,
            q3 + UpperThreshold * iqr,
        ]

        return (
            np.full(len(df), limitsthresholdIQR[1]),
            np.full(len(df), limitsthresholdIQR[0]),
            df[data_col].to_numpy(),
        )


if __name__ == "__main__":
    import numpy as np

    np.random.seed(0)
    actual = np.random.randint(100, 500, 36)
    Method = "Rolling Sigma"

    ul, ll, mean = calculate_bound(actual, 1.5, 1.5, 12, Method)
    print(ul, ll, mean)
    # plt.plot(actual, color="blue")
    #
    # ul_fs, ll_fs, mean_fs = calculate_bound(
    #     actual, 1.5, 1.5, 12, "Fixed Sigma"
    # )
    # ul_rs, ll_rs, mean_rs = calculate_bound(
    #     actual, 1.5, 1.5, 12, "Rolling Sigma"
    # )
    # ul_iqr, ll_iqr, mean_iqr = calculate_bound(
    #     actual, 1.5, 1.5, 12, "Seasonal IQR"
    # )
    #
    # # Fixed sigma
    # plt.plot(ul_fs, color="yellow")
    # plt.plot(ll_fs, color="yellow")
    # plt.plot(mean_fs, color="yellow")
    #
    # # IQR
    # plt.plot(ul_iqr, color="green")
    # plt.plot(ll_iqr, color="green")
    # plt.plot(mean_iqr, color="green")
    #
    # # Rolling sigma
    # plt.plot(ul_rs, color="red")
    # plt.plot(ll_rs, color="red")
    # plt.plot(mean_rs, color="red")
    #
    # plt.show()
