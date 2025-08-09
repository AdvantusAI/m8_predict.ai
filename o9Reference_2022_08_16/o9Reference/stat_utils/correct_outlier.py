"""
Version : 1.0.0
Maintained by : dpref@o9solutions.com
"""
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("o9_logger")
import numpy as np
import pandas as pd


def correct_outlier(
    data: np.ndarray,
    outlier: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    replaceby: str = "Limit",
):
    """
    function to return a list of outlier corrected values
    """
    assert isinstance(data, np.ndarray), "data should be np array type ..."
    assert isinstance(
        outlier, np.ndarray
    ), "outlier should be np array type ..."
    assert isinstance(
        upper_bound, np.ndarray
    ), "upper_bound should be np array type ..."
    assert isinstance(
        lower_bound, np.ndarray
    ), "lower_bound should be np array type ..."
    assert isinstance(replaceby, str), "replaceby should be string type ..."
    assert len(data) == len(
        outlier
    ), "both data and outlier should have same size ..."
    assert len(data) == len(
        upper_bound
    ), "both data and upper_bound should have same size ..."
    assert len(data) == len(
        lower_bound
    ), "both data and lower_bound should have same size ..."

    function_name = "correct_outlier"
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
            "inside function {}: returning uncleansed data".format(
                function_name
            )
        )
        return data

    if lower_bound is None or len(lower_bound) == 0:
        logger.info(
            "inside function {}: Lower bound is empty".format(function_name)
        )
        logger.info(
            "inside function {}: returning uncleansed data".format(
                function_name
            )
        )
        return data

    if outlier is None or len(outlier) == 0:
        logger.info(
            "inside function {}: outlier is empty".format(function_name)
        )
        logger.info(
            "inside function {}: returning uncleansed data".format(
                function_name
            )
        )
        return data

    # column names
    data_col = "data"
    upper_bound_col = "Upper Limit"
    lower_bound_col = "Lower Limit"
    outlier_col = "outlier"
    cleansed_data_col = "Cleansed Data"

    # creating dataframe
    combined_data = np.hstack(
        (data[:, None], upper_bound[:, None], lower_bound[:, None])
    )
    combined_columns = [data_col, upper_bound_col, lower_bound_col]
    df = pd.DataFrame(combined_data, columns=combined_columns)
    # adding boolean array
    df[outlier_col] = outlier.tolist()

    # replacing outlier by mean or upper/lower bound
    if replaceby == "Interpolation":
        df[cleansed_data_col] = np.where(df[outlier_col], np.nan, df[data_col])
        df[cleansed_data_col] = df[cleansed_data_col].interpolate(
            method="linear"
        )
    elif replaceby == "Limit":
        df[cleansed_data_col] = np.where(
            df[outlier_col],
            np.where(
                df[data_col].gt(df[upper_bound_col]),
                df[upper_bound_col],
                df[lower_bound_col],
            ),
            df[data_col],
        )
    else:
        logger.info(
            "inside function {}: Outlier correction not method not specified ...".format(
                function_name
            )
        )
        logger.info(
            "inside function {}: returning uncleansed data".format(
                function_name
            )
        )
        return data

    return df[cleansed_data_col].to_numpy()


if __name__ == "__main__":
    np.random.seed(0)
    actual = np.random.randint(100, 400, 12)
    upper_bound = np.random.randint(300, 450, 12)
    lower_bound = np.random.randint(0, 150, 12)
    outlier = np.array(
        [
            False,
            False,
            False,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            False,
        ],
        dtype=bool,
    )
    tuples = list(zip(actual, upper_bound, lower_bound, outlier))
    df_tuple = pd.DataFrame(
        tuples, columns=["actual", "up_bound", "lo_bound", "outlier_col"]
    )
    output = correct_outlier(actual, outlier, upper_bound, lower_bound)
    df_tuple["cleansed data"] = output
    print(df_tuple)
