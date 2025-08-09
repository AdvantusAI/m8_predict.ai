"""
Version : 1.0.0
Maintained by : dpref@o9solutions.com
"""
import logging

import pandas as pd

logger = logging.getLogger("o9_logger")
from pandas import merge
from o9Reference.common_utils.common_utils import is_dimension
from functools import wraps


def ensure_valid_output_schema(fn):
    """
    Decorator to print inputs and outputs from a function.
    :param fn:
    :return:
    """

    @wraps(fn)
    def func(*args, **kwargs):
        # invoke function, get result
        result = fn(*args, **kwargs)

        logger.info("Validating schema of output dataframes ...")

        # outputs could be multiple (will be a tuple)
        if type(result) == tuple:
            # create empty list to store output
            valid_result = list()

            # check if all items in the tuple have valid schema, fix the invalid ones
            for the_item in result:
                if type(the_item) == pd.DataFrame:
                    valid_result.append(check_schema(the_item))
                else:
                    # no need of any processing
                    valid_result.append(the_item)

            # convert list to tuple
            valid_result = tuple(valid_result)

        # else check for single output dataframe
        else:
            if type(result) == pd.DataFrame:
                valid_result = check_schema(result)
            else:
                valid_result = result

        return valid_result

    return func


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


def convert_category_to_str(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts all the category columns to string
    :param df: input dataframe
    :return: df with columns converted
    """
    for the_col in df.columns:
        try:
            if df[the_col].dtype.name == "category":
                df[the_col] = df[the_col].astype("str")
        except Exception as e:
            logger.error(
                "Exception while converting {} to string ...".format(the_col)
            )
            logger.exception(e)
    return df


def get_o9_empty_df(column_list=None):
    """
    Returns an empty dataframe with o9 valid schema to write back to Live Server.

    :param column_list: list of columns required in dataframe
    :return:
    """

    # if columns list is not supplied, take Version and History Measure as headers
    if column_list is None:
        column_list = ["Version.[Version Name]", "History Measure"]
    return pd.DataFrame(columns=column_list)


def check_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks if the input dataframe follow valid o9 schema, converts to valid schema if not
    :param df:
    :return:
    """
    try:
        # check if type is pandas dataframe
        if type(df) == pd.DataFrame:
            # empty dataframe with no rows/columns, assign valid o9 empty df
            if df.shape == (0, 0):
                return get_o9_empty_df()
            else:
                # dataframe contains only dimension columns, but no measures
                # check if all columns are dimensions
                is_dimension_flags = [is_dimension(x) for x in df.columns]

                # assign valid o9 empty df
                if all(is_dimension_flags):
                    logger.warning(
                        "No measure columns found in df, treating {} columns as dimensions and returning o9 empty df".format(
                            list(df.columns)
                        )
                    )
                    df = get_o9_empty_df()
    except Exception as e:
        logger.exception("Exception {} in check_schema function ".format(e))

    return df


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

    # time_master = pd.DataFrame(
    #     {"Time.[Month]": ["M01", "M02", "M03", "M04", "M05"]}
    # )
    # intersections_master = pd.DataFrame(
    #     {
    #         "Item": ["I1", "I2", "I1", "I2"],
    #         "Customer": ["C1", "C1", "C2", "C2"],
    #     }
    # )
    # df = create_cartesian_product(df1=time_master, df2=intersections_master)
    # print(df)

    # print(get_o9_empty_df())

    # print(get_o9_empty_df(column_list=['Hancel']))

    # my_df = pd.DataFrame({
    #
    # })
    #
    # print(my_df)
    # print("-------------\n")
    # print(check_schema(my_df))
    #
    # print("-------------\n")
    # print("-------------\n")
    # print("-------------\n")
    # my_df = pd.DataFrame({
    #     "Item.[Stat Item]": [], "Sales Domain.[Stat Customer Group]": [], "Location.[Stat Location]": []
    # })
    # print(my_df)
    # print("-------------\n")
    # print(check_schema(my_df))

    @ensure_valid_output_schema
    def my_function():
        my_df1 = pd.DataFrame(
            {
                "Item.[Stat Item]": [],
                "Sales Domain.[Stat Customer Group]": [],
                "Location.[Stat Location]": [],
            }
        )

        my_df3 = pd.DataFrame(
            {
                "Item.[Stat Item]": [],
                "Sales Domain.[Stat Customer Group]": [],
                "Location.[Stat Location]": [],
                "Stat Fcst SES": [],
            }
        )

        my_df2 = pd.DataFrame({})

        test = "my string"
        return my_df1, my_df2, test, my_df3

    print(my_function())
