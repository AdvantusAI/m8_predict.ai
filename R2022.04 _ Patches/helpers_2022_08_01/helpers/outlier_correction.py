"""
Version : 2022.08.01
Maintained by : pmm_algocoe@o9solutions.com
"""
import logging

import numpy as np
import pandas as pd
from o9Reference.common_utils.data_utils import (
    check_for_columns_with_same_value,
)
from o9Reference.stat_utils.calculate_bound import calculate_bound
from o9Reference.stat_utils.correct_outlier import correct_outlier
from o9Reference.stat_utils.detect_outlier import detect_outlier

logger = logging.getLogger("o9_logger")


def cleanse_data(
    data,
    upper_threshold,
    lower_threshold,
    outlier_method,
    replace_by,
    time_series_freq,
) -> (pd.DataFrame, np.ndarray, np.ndarray, float):
    function_name = "cleanse_data"
    logger.info("inside function {}...".format(function_name))
    if data is None or len(data) == 0:
        logger.info("inside function {}: data is empty".format(function_name))
        logger.info(
            "inside function {}: returning empty data".format(function_name)
        )
        return np.array([])

    logger.info("...Upper Threshold: {}".format(upper_threshold))
    logger.info("...Lower Threshold: {}".format(lower_threshold))
    logger.info("...Outlier Method: {}".format(outlier_method))
    logger.info("...replace by: {}".format(replace_by))

    upper_bound, lower_bound, mean = calculate_bound(
        data,
        upper_threshold,
        lower_threshold,
        time_series_freq,
        outlier_method,
    )

    # detecting outlier
    outlier_bool = detect_outlier(data, upper_bound, lower_bound)

    # correcting outlier
    cleansed_data = correct_outlier(
        data, outlier_bool, upper_bound, lower_bound, replace_by
    )

    return cleansed_data, upper_bound, lower_bound, mean


def cleanse_data_wrapper(
    df,
    dimensions,
    relevant_time_key,
    history_measure,
    cleansed_data_output,
    upper_threshold_output,
    lower_threshold_output,
    actual_mean_output,
    time_level_col,
    OutlierParameters,
    upper_threshold_col,
    lower_threshold_col,
    outlier_method_col,
    outlier_correction_col,
    time_series_freq,
    interactive_stat=False,
) -> pd.DataFrame():
    result = pd.DataFrame()
    logger.info("Executing for {}".format(df[dimensions].iloc[0].values))
    try:
        if len(df) == 0:
            return result

        # check for similarity of data in each column of dataframe and extract the required flag
        same_value_of_actual_flag = check_for_columns_with_same_value(df)

        # extracting and checking the flag for data column
        if not same_value_of_actual_flag[df.columns.get_loc(history_measure)]:

            # Interactive stat applies the same parameters for all combinations
            if interactive_stat:
                outlier_params = OutlierParameters
            else:
                outlier_params = (
                    df[dimensions]
                    .drop_duplicates()
                    .merge(OutlierParameters, how="inner", on=dimensions)
                )

            if len(outlier_params) == 0:
                logger.warning(
                    "No outlier parameters configured for intersection : {}".format(
                        df[dimensions].iloc[0].values
                    )
                )
                return result

            # Extract data from outlier_params
            upper_threshold = float(
                outlier_params[upper_threshold_col].iloc[0]
            )
            lower_threshold = float(
                outlier_params[lower_threshold_col].iloc[0]
            )
            outlier_method = str(outlier_params[outlier_method_col].iloc[0])
            replace_by = str(outlier_params[outlier_correction_col].iloc[0])

            df.sort_values(relevant_time_key, inplace=True)
            cleansed_data, upper_bound, lower_bound, mean = cleanse_data(
                df[history_measure].copy().to_numpy(),
                upper_threshold,
                lower_threshold,
                outlier_method,
                replace_by,
                time_series_freq,
            )
        else:
            # in case series has same data points, we populate same values in outlier and bounds
            existing_value = df[history_measure].iloc[0]
            cleansed_data = np.full(len(df), existing_value)
            upper_bound = np.full(len(df), existing_value)
            lower_bound = np.full(len(df), existing_value)
            mean = np.full(len(df), existing_value)

        df[cleansed_data_output] = cleansed_data
        df[upper_threshold_output] = upper_bound
        df[lower_threshold_output] = lower_bound
        df[actual_mean_output] = mean

        cols_required_in_res = (
            dimensions
            + [time_level_col]
            + [
                cleansed_data_output,
                upper_threshold_output,
                lower_threshold_output,
                actual_mean_output,
            ]
        )

        result = df[cols_required_in_res]

        if len(result) == 0:
            logger.warning("No data found after cleansing...")
            logger.warning("Returning empty dataframe...")

    except Exception as e:
        logger.error("Exception for {}".format(df[dimensions].iloc[0].values))
        logger.exception(e)
    return result
