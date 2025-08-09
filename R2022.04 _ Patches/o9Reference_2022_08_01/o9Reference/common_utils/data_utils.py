"""
Version : 1.0.0
Maintained by : dpref@o9solutions.com
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("o9_logger")


def check_for_columns_with_same_value(df: pd.DataFrame) -> np.ndarray:
    """
    Checks for columns with same value in a dataframe and returns np array of booleans in specified column order
    """
    a = df.to_numpy()  # df.values (pandas<0.24)
    return (a[0] == a).all(0)


def validate_output(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    forecast_level: list,
    show_warnings: bool = True,
) -> None:
    """
    Given an input dataframe and output dataframe, checks if all combinations in input are present in output.
    Raises a warning message with the missing intersections list.
    :param input_df:
    :param output_df:
    :param forecast_level:
    :return:
    """
    try:
        if show_warnings:
            assert (
                set(forecast_level).issubset(set(input_df.columns)) is True
            ), "input df does not contain the columns specified in forecast_level argument ..."

            if input_df.empty:
                logger.warning(
                    "input_df is empty, returning without checking outputs ..."
                )
                return None

            input_list = list(
                input_df.groupby(forecast_level, observed=True).groups.keys()
            )

            # set warning message
            warning_msg = "Kindly reduce the execution scope to any one of intersections listed above, set multiprocessing_num_cores=1 in plugin code and run the plugin again to see detailed logs ..."

            if output_df.empty:
                logger.warning(
                    "Output not generated for any of the {} intersections, displaying 5 such intersections : {} ...".format(
                        len(input_list), input_list[:5]
                    )
                )
                logger.warning(warning_msg)
                return None

            assert (
                set(forecast_level).issubset(set(output_df.columns)) is True
            ), "output_df does not contain the columns specified in forecast_level argument ..."

            output_list = list(
                output_df.groupby(forecast_level, observed=True).groups.keys()
            )

            # check if all combinations in input has been processed, raise warning otherwise
            if len(input_list) != len(output_list):
                missing_intersections = [
                    x for x in input_list if x not in output_list
                ]
                logger.warning(
                    "Output not generated for {} out of {} intersections, missing_intersections : {}".format(
                        len(missing_intersections),
                        len(input_list),
                        missing_intersections,
                    )
                )
                logger.warning(warning_msg)
    except Exception as e:
        logger.exception(
            "Exception {} from validate_output function ...".format(e)
        )

    return None


if __name__ == "__main__":
    input_df = pd.DataFrame(
        {
            "group": ["A", "B", "C", "D", "E", "F", "G", "H"],
            "value": [30, 40, 44, 46, 47, 56, 7, 8],
        }
    )

    output_df = pd.DataFrame({"group": "A", "value": 50}, index=[0])

    validate_output(input_df, output_df, forecast_level=["group"])

    validate_output(
        input_df,
        pd.DataFrame(columns=["History Measure"]),
        forecast_level=["group"],
    )

    # validate_output(input_df, pd.DataFrame(), forecast_level=["group"])

    # from o9Reference.common_utils.data_utils import check_for_columns_with_same_value
    # import random
    # df = pd.DataFrame({
    #     'Stat Fcst SES': [5.66]*10,
    #     'Stat Fcst DES' : [5.89]*10,
    #     'Stat Fcst TES' : [random.randrange(1, 25, 1) for i in range(10)],
    #     'Stat Fcst TBATS' : [random.randrange(1, 25, 1) for i in range(10)],
    # })
    #
    # cols_with_same_value = check_for_columns_with_same_value(df)
    # print(df.columns[cols_with_same_value])
