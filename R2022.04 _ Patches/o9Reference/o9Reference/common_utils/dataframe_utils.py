"""
Version : 1.0.0
Maintained by : dpref@o9solutions.com
"""
import logging

import pandas as pd

logger = logging.getLogger("o9_logger")
from pandas import merge


def concat_to_dataframe(list_of_results: list) -> pd.DataFrame:
    """
    Given a list of dataframes, checks for nulls and returns concatenated dataframe with all rows.

    :param list_of_results: input list containing dataframes
    :return: pd.DataFrame
    """
    result = pd.DataFrame()

    # filter non None values from list
    list_of_results = [x for x in list_of_results if x is not None]

    # Check if list is empty
    if list_of_results:
        result = pd.concat(list_of_results, ignore_index=True)
    return result


def create_cartesian_product(df1, df2):
    """
    Given two dataframes, returns the cartesian product of rows between two dataframes
    :param df1:
    :param df2:
    :return:
    """
    if len(df1) == 0 or len(df2) == 0:
        logger.warning(
            "Empty dataframes supplied, return empty dataframe as cartesian product"
        )
        return pd.DataFrame()

    # create copy so that sources are not affected
    df_1 = df1.copy()
    df_2 = df2.copy()

    # create same column with dummy value in both dataframes
    cross_join_key_col = "key"
    df_1[cross_join_key_col] = "dummy_value"
    df_2[cross_join_key_col] = "dummy_value"

    # join algo params with intersections master
    combined = merge(df_1, df_2, on=cross_join_key_col)
    combined.drop(cross_join_key_col, axis=1, inplace=True)

    # assert length of combined dataframe is cross product
    assert len(combined) == len(df_1.drop_duplicates()) * len(
        df_2.drop_duplicates()
    ), "Join results in invalid result, kindly check both input dataframes ..."

    return combined


if __name__ == "__main__":
    # from o9Reference.common_utils.dataframe_utils import concat_to_dataframe
    # result_list = [
    #     None,
    #     pd.DataFrame({"A": 1}, index=[0]),
    #     pd.DataFrame({"A": 2}, index=[0]),
    #     pd.DataFrame({"A": 110}, index=[0]),
    #     pd.DataFrame({"A": 97}, index=[0]),
    # ]
    # df = concat_to_dataframe(result_list)
    # print(df)

    time_master = pd.DataFrame(
        {"Time.[Month]": ["M01", "M02", "M03", "M04", "M05"]}
    )
    intersections_master = pd.DataFrame(
        {
            "Item": ["I1", "I2", "I1", "I2"],
            "Customer": ["C1", "C1", "C2", "C2"],
        }
    )
    df = create_cartesian_product(df1=time_master, df2=intersections_master)
    print(df)
