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
    input_df: pd.DataFrame, output_df: pd.DataFrame, forecast_level: list
) -> None:
    """
    Given an input dataframe and output dataframe, checks if all combinations in input are present in output.
    Raises a warning message with the missing intersections list.
    :param input_df:
    :param output_df:
    :param forecast_level:
    :return:
    """
    assert (
        set(forecast_level).issubset(set(input_df.columns)) is True
    ), "input df does not contain the columns specified in forecast_level argument ..."
    assert len(input_df) > 0, "input df cannot be empty ..."

    input_list = list(
        input_df.groupby(forecast_level, observed=True).groups.keys()
    )

    if len(output_df) == 0:
        logger.warning(
            "Output not generated for the following intersections : {}".format(
                input_list
            )
        )
        logger.warning(
            "Kindly inspect input data for these intersections/make a dry run ..."
        )
        return None

    assert (
        set(forecast_level).issubset(set(output_df.columns)) is True
    ), "output_df does not contain the columns specified in forecast_level argument ..."

    output_list = list(
        output_df.groupby(forecast_level, observed=True).groups.keys()
    )

    # check if all combinations in input has been processed, raise warning otherwise
    if len(input_list) != len(output_list):
        missing_intersections = [x for x in input_list if x not in output_list]
        logger.warning(
            "Output not generated for the following intersections : {}".format(
                missing_intersections
            )
        )
        logger.warning(
            "Kindly inspect input data for these intersections/make a dry run ..."
        )
    return None


if __name__ == "__main__":
    input_df = pd.DataFrame(
        {"group": ["A", "B", "C", "D"], "value": [30, 40, 44, 46]}
    )

    output_df = pd.DataFrame({"group": "A", "value": 50}, index=[0])

    validate_output(input_df, output_df, forecast_level=["group"])

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
