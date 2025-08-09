"""
Version : 1.0.0
Maintained by : dpref@o9solutions.com
"""
import logging
from typing import List

import numpy as np

from o9Reference.common_utils.function_timer import timed

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("o9_logger")
import pandas as pd


@timed
def split_string(
    values: list, delimiter: str, col_names: list
) -> pd.DataFrame:
    """
    Splits list of string into specified columns and returns pandas dataframe.

    :param values:
    :param delimiter:
    :param col_names:
    :return:
    """
    result = pd.DataFrame()
    logger.info("Length of input list : {}".format(len(values)))
    logger.info("column names : {}".format(col_names))
    logger.info("Splitting list of strings based on {} ...".format(delimiter))
    try:
        assert isinstance(values, list), "Datatype error for values ..."
        assert isinstance(delimiter, str), "Datatype error for delimiter..."
        assert isinstance(col_names, list), "Datatype error for col_names..."
        assert None not in values, "Input cannot contain NaNs ..."
        assert np.nan not in values, "Input cannot contain NaNs ..."

        # check for empty list
        if not values:
            return result

        logger.info("Sample value from input list : {}".format(values[0]))

        # all values in list should contain the delimiter character
        assert any(
            delimiter in the_string for the_string in values
        ), "Delimiter not present in all values, check input data ..."

        # split and form dataframe
        result = pd.Series(values).str.split(delimiter, expand=True)

        # check output
        if len(result.columns) != len(col_names):
            raise ValueError(
                "Number of columns after splitting does not match input column names ..."
            )

        # assign column names
        result.columns = col_names
    except Exception as e:
        logger.exception(e)
    return result


def get_n_time_periods(
    latest_value: str,
    periods: int,
    time_mapping: pd.DataFrame,
    time_attribute: dict,
    include_latest_value: bool = True,
) -> List[str]:
    """
    Get n time periods based on values provided, sample parameters below:
        latest_value : 'M03-20'
        periods : -12
        time_mapping : Contains 'MonthName' and 'MonthKey', and other time columns
        time_attribute : key value pair {'MonthName':'MonthKey'}
        include_latest_value : boolean indicating whether to include current time or not
    """
    assert isinstance(
        latest_value, str
    ), "latest_value should be of type str ..."
    assert isinstance(periods, int), "periods should be of type int ..."
    assert isinstance(
        time_mapping, pd.DataFrame
    ), "time_mapping should be of type pd.DataFrame ..."
    assert isinstance(
        time_attribute, dict
    ), "time_attribute should be of type dictionary ..."

    assert latest_value != "", "latest_value cannot be empty string ..."
    assert bool(time_attribute), "time_attribute cannot be empty ..."
    assert isinstance(
        include_latest_value, bool
    ), "include_current_time should be bool datatype ..."

    time_name = list(time_attribute.keys())[0]
    time_key = time_attribute[time_name]

    assert (
        time_name in time_mapping.columns
    ), "{} not found in time_mapping ...".format(time_name)
    assert (
        time_key in time_mapping.columns
    ), "{} not found in time_mapping ...".format(time_key)

    required_time_cols = [time_name, time_key]
    req_time_mapping = time_mapping[required_time_cols].drop_duplicates()
    req_time_mapping[time_key] = pd.to_datetime(req_time_mapping[time_key])
    req_time_mapping.sort_values(time_key, inplace=True)
    req_time_mapping.reset_index(drop=True, inplace=True)

    assert len(req_time_mapping) > abs(
        periods
    ), "Insufficient time mapping data to map {} periods, time mapping contains only {} {}s ...".format(
        abs(periods), len(req_time_mapping), time_name
    )

    # get index of latest value
    index_of_latest_value = req_time_mapping[
        req_time_mapping[time_name] == latest_value
    ].index[0]

    # get forward looking dates from latest value
    if periods > 0:
        if include_latest_value:
            start_index = index_of_latest_value
        else:
            start_index = index_of_latest_value + 1

        # end is exclusive while filtering, manipulate index (+1) to get last n periods.
        end_index = start_index + periods
        result = req_time_mapping[start_index:end_index]
    else:
        # get last n dates from latest value
        # end is inclusive while filtering, manipulate index (+1) to get last n periods.
        if include_latest_value:
            end_index = index_of_latest_value + 1
        else:
            end_index = index_of_latest_value

        start_index = end_index + periods
        result = req_time_mapping[start_index:end_index]

    assert (
        len(result) > 0
    ), "time mapping contains periods between {} and {}, cannot source {} periods from {}".format(
        req_time_mapping[time_key].iloc[0],
        req_time_mapping[time_key].iloc[-1],
        periods,
        latest_value,
    )
    n_periods = list(result[time_name])
    assert len(n_periods) == abs(
        periods
    ), "Length of output doesn't match specified num of periods ..."
    return n_periods


def filter_relevant_time_mapping(
    frequency: str, time_mapping: pd.DataFrame
) -> pd.DataFrame:
    """
    Given a frequency, this function filters out only the relevant key and name from the master time mapping dataframe.
    Also assigns the key to a date column name specified (for ease of joins later).

    """
    assert isinstance(frequency, str), "frequency should be string type ..."
    assert isinstance(
        time_mapping, pd.DataFrame
    ), "time_mapping should be pandas dataframe type ..."

    if time_mapping.empty:
        logger.warning(
            "Input time_mapping dataframe is empty, returning empty dataframe ..."
        )
        return pd.DataFrame()

    if frequency == "Weekly":
        time_mapping = time_mapping[["WeekName", "WeekKey"]].copy(deep=True)
    elif frequency == "Monthly":
        time_mapping = time_mapping[["MonthName", "MonthKey"]].copy(deep=True)
    elif frequency == "Quarterly":
        time_mapping = time_mapping[["QuarterName", "QuarterKey"]].copy(
            deep=True
        )
    else:
        raise ValueError(
            "Invalid frequency input, supported frequencies are [Weekly, Monthly, Quarterly] ..."
        )
    result = time_mapping.drop_duplicates().reset_index(drop=True)
    return result


def get_relevant_time_name_and_key(frequency: str) -> (str, str):
    """
    Given a valid frequency, returns a tuple of the relevant time name and key.
    Raises ValueError if frequency is invalid
    :param frequency:
    :return:
    """
    if frequency == "Weekly":
        relevant_time_name = "WeekName"
        relevant_time_key = "WeekKey"
    elif frequency == "Monthly":
        relevant_time_name = "MonthName"
        relevant_time_key = "MonthKey"
    elif frequency == "Quarterly":
        relevant_time_name = "QuarterName"
        relevant_time_key = "QuarterKey"
    else:
        raise ValueError(
            "Unknown frequency {}, supported frequencies are Weekly, Monthly and Quarterly"
        )
    return relevant_time_name, relevant_time_key


def get_seasonal_periods(frequency: str) -> int:
    """
    Returns num seasonal periods based on frequency provided
    """
    if frequency == "Weekly":
        return 52
    elif frequency == "Monthly":
        return 12
    elif frequency == "Quarterly":
        return 4
    else:
        raise ValueError(
            "Unknown frequency {}, supported frequencies are Weekly, Monthly and Quarterly".format(
                frequency
            )
        )


def is_dimension(input_string: str) -> bool:
    """
    Checks if the input string contains square bracket open/close, dot character and returns a boolean
    to indicate whether the input string is a dimension on measure

    :param input_string: str
    :return: str
    """
    assert type(input_string) == str, "Input type should be str .."
    assert input_string != "", "Empty string provided ..."

    if "[" in input_string and "]" in input_string and "." in input_string:
        return True
    else:
        return False


if __name__ == "__main__":
    # relevant_time_name, relevant_time_key = get_relevant_time_name_and_key(frequency="Monthly")
    # print(relevant_time_name, relevant_time_key)
    # print(get_seasonal_periods(frequency="Quarterly"))

    # result = split_string(
    #     values=["foo|bar"], delimiter="|", col_names=["foo", "bar"]
    # )
    # logger.info("result : {}".format(result))
    #
    # # import the package
    # import pandas as pd
    #
    # # creating a test dataset
    # TimeDimension = pd.DataFrame(
    #     {
    #         "Week": [
    #             "W01-21",
    #             "W02-21",
    #             "W03-21",
    #             "W04-21",
    #             "W05-21",
    #             "W06-21",
    #             "W07-21",
    #             "W08-21",
    #             "W09-21",
    #             "W10-21",
    #             "W11-21",
    #             "W12-21",
    #             "W13-21",
    #             "W14-21",
    #             "W15-21",
    #             "W16-21",
    #             "W17-21",
    #             "W18-21",
    #             "W19-21",
    #             "W20-21",
    #             "W21-21",
    #             "W22-21",
    #             "W23-21",
    #             "W24-21",
    #             "W25-21",
    #             "W26-21",
    #             "W27-21",
    #             "W28-21",
    #             "W29-21",
    #             "W30-21",
    #             "W31-21",
    #             "W32-21",
    #             "W33-21",
    #             "W34-21",
    #             "W35-21",
    #             "W36-21",
    #             "W37-21",
    #             "W38-21",
    #             "W39-21",
    #             "W40-21",
    #             "W41-21",
    #             "W42-21",
    #             "W43-21",
    #             "W44-21",
    #             "W45-21",
    #             "W46-21",
    #             "W47-21",
    #             "W48-21",
    #             "W49-21",
    #             "W50-21",
    #             "W51-21",
    #             "W52-21",
    #         ],
    #         "Stat R Attribute Week": [
    #             "27-Dec-20|W01-21|M01-21|Q1-21|12/27/2020 12:00:00 AM|12/27/2020 12:00:00 AM|12/27/2020 12:00:00 AM|12/27/2020 12:00:00 AM",
    #             "03-Jan-21|W02-21|M01-21|Q1-21|1/3/2021 12:00:00 AM|1/3/2021 12:00:00 AM|12/27/2020 12:00:00 AM|12/27/2020 12:00:00 AM",
    #             "10-Jan-21|W03-21|M01-21|Q1-21|1/10/2021 12:00:00 AM|1/10/2021 12:00:00 AM|12/27/2020 12:00:00 AM|12/27/2020 12:00:00 AM",
    #             "17-Jan-21|W04-21|M01-21|Q1-21|1/17/2021 12:00:00 AM|1/17/2021 12:00:00 AM|12/27/2020 12:00:00 AM|12/27/2020 12:00:00 AM",
    #             "24-Jan-21|W05-21|M02-21|Q1-21|1/24/2021 12:00:00 AM|1/24/2021 12:00:00 AM|1/24/2021 12:00:00 AM|12/27/2020 12:00:00 AM",
    #             "31-Jan-21|W06-21|M02-21|Q1-21|1/31/2021 12:00:00 AM|1/31/2021 12:00:00 AM|1/24/2021 12:00:00 AM|12/27/2020 12:00:00 AM",
    #             "07-Feb-21|W07-21|M02-21|Q1-21|2/7/2021 12:00:00 AM|2/7/2021 12:00:00 AM|1/24/2021 12:00:00 AM|12/27/2020 12:00:00 AM",
    #             "14-Feb-21|W08-21|M02-21|Q1-21|2/14/2021 12:00:00 AM|2/14/2021 12:00:00 AM|1/24/2021 12:00:00 AM|12/27/2020 12:00:00 AM",
    #             "21-Feb-21|W09-21|M03-21|Q1-21|2/21/2021 12:00:00 AM|2/21/2021 12:00:00 AM|2/21/2021 12:00:00 AM|12/27/2020 12:00:00 AM",
    #             "28-Feb-21|W10-21|M03-21|Q1-21|2/28/2021 12:00:00 AM|2/28/2021 12:00:00 AM|2/21/2021 12:00:00 AM|12/27/2020 12:00:00 AM",
    #             "07-Mar-21|W11-21|M03-21|Q1-21|3/7/2021 12:00:00 AM|3/7/2021 12:00:00 AM|2/21/2021 12:00:00 AM|12/27/2020 12:00:00 AM",
    #             "14-Mar-21|W12-21|M03-21|Q1-21|3/14/2021 12:00:00 AM|3/14/2021 12:00:00 AM|2/21/2021 12:00:00 AM|12/27/2020 12:00:00 AM",
    #             "21-Mar-21|W13-21|M03-21|Q1-21|3/21/2021 12:00:00 AM|3/21/2021 12:00:00 AM|2/21/2021 12:00:00 AM|12/27/2020 12:00:00 AM",
    #             "28-Mar-21|W14-21|M04-21|Q2-21|3/28/2021 12:00:00 AM|3/28/2021 12:00:00 AM|3/28/2021 12:00:00 AM|3/28/2021 12:00:00 AM",
    #             "04-Apr-21|W15-21|M04-21|Q2-21|4/4/2021 12:00:00 AM|4/4/2021 12:00:00 AM|3/28/2021 12:00:00 AM|3/28/2021 12:00:00 AM",
    #             "11-Apr-21|W16-21|M04-21|Q2-21|4/11/2021 12:00:00 AM|4/11/2021 12:00:00 AM|3/28/2021 12:00:00 AM|3/28/2021 12:00:00 AM",
    #             "18-Apr-21|W17-21|M04-21|Q2-21|4/18/2021 12:00:00 AM|4/18/2021 12:00:00 AM|3/28/2021 12:00:00 AM|3/28/2021 12:00:00 AM",
    #             "25-Apr-21|W18-21|M05-21|Q2-21|4/25/2021 12:00:00 AM|4/25/2021 12:00:00 AM|4/25/2021 12:00:00 AM|3/28/2021 12:00:00 AM",
    #             "02-May-21|W19-21|M05-21|Q2-21|5/2/2021 12:00:00 AM|5/2/2021 12:00:00 AM|4/25/2021 12:00:00 AM|3/28/2021 12:00:00 AM",
    #             "09-May-21|W20-21|M05-21|Q2-21|5/9/2021 12:00:00 AM|5/9/2021 12:00:00 AM|4/25/2021 12:00:00 AM|3/28/2021 12:00:00 AM",
    #             "16-May-21|W21-21|M05-21|Q2-21|5/16/2021 12:00:00 AM|5/16/2021 12:00:00 AM|4/25/2021 12:00:00 AM|3/28/2021 12:00:00 AM",
    #             "23-May-21|W22-21|M06-21|Q2-21|5/23/2021 12:00:00 AM|5/23/2021 12:00:00 AM|5/23/2021 12:00:00 AM|3/28/2021 12:00:00 AM",
    #             "30-May-21|W23-21|M06-21|Q2-21|5/30/2021 12:00:00 AM|5/30/2021 12:00:00 AM|5/23/2021 12:00:00 AM|3/28/2021 12:00:00 AM",
    #             "06-Jun-21|W24-21|M06-21|Q2-21|6/6/2021 12:00:00 AM|6/6/2021 12:00:00 AM|5/23/2021 12:00:00 AM|3/28/2021 12:00:00 AM",
    #             "13-Jun-21|W25-21|M06-21|Q2-21|6/13/2021 12:00:00 AM|6/13/2021 12:00:00 AM|5/23/2021 12:00:00 AM|3/28/2021 12:00:00 AM",
    #             "20-Jun-21|W26-21|M06-21|Q2-21|6/20/2021 12:00:00 AM|6/20/2021 12:00:00 AM|5/23/2021 12:00:00 AM|3/28/2021 12:00:00 AM",
    #             "27-Jun-21|W27-21|M07-21|Q3-21|6/27/2021 12:00:00 AM|6/27/2021 12:00:00 AM|6/27/2021 12:00:00 AM|6/27/2021 12:00:00 AM",
    #             "04-Jul-21|W28-21|M07-21|Q3-21|7/4/2021 12:00:00 AM|7/4/2021 12:00:00 AM|6/27/2021 12:00:00 AM|6/27/2021 12:00:00 AM",
    #             "11-Jul-21|W29-21|M07-21|Q3-21|7/11/2021 12:00:00 AM|7/11/2021 12:00:00 AM|6/27/2021 12:00:00 AM|6/27/2021 12:00:00 AM",
    #             "18-Jul-21|W30-21|M07-21|Q3-21|7/18/2021 12:00:00 AM|7/18/2021 12:00:00 AM|6/27/2021 12:00:00 AM|6/27/2021 12:00:00 AM",
    #             "25-Jul-21|W31-21|M08-21|Q3-21|7/25/2021 12:00:00 AM|7/25/2021 12:00:00 AM|7/25/2021 12:00:00 AM|6/27/2021 12:00:00 AM",
    #             "01-Aug-21|W32-21|M08-21|Q3-21|8/1/2021 12:00:00 AM|8/1/2021 12:00:00 AM|7/25/2021 12:00:00 AM|6/27/2021 12:00:00 AM",
    #             "08-Aug-21|W33-21|M08-21|Q3-21|8/8/2021 12:00:00 AM|8/8/2021 12:00:00 AM|7/25/2021 12:00:00 AM|6/27/2021 12:00:00 AM",
    #             "15-Aug-21|W34-21|M08-21|Q3-21|8/15/2021 12:00:00 AM|8/15/2021 12:00:00 AM|7/25/2021 12:00:00 AM|6/27/2021 12:00:00 AM",
    #             "22-Aug-21|W35-21|M09-21|Q3-21|8/22/2021 12:00:00 AM|8/22/2021 12:00:00 AM|8/22/2021 12:00:00 AM|6/27/2021 12:00:00 AM",
    #             "29-Aug-21|W36-21|M09-21|Q3-21|8/29/2021 12:00:00 AM|8/29/2021 12:00:00 AM|8/22/2021 12:00:00 AM|6/27/2021 12:00:00 AM",
    #             "05-Sep-21|W37-21|M09-21|Q3-21|9/5/2021 12:00:00 AM|9/5/2021 12:00:00 AM|8/22/2021 12:00:00 AM|6/27/2021 12:00:00 AM",
    #             "12-Sep-21|W38-21|M09-21|Q3-21|9/12/2021 12:00:00 AM|9/12/2021 12:00:00 AM|8/22/2021 12:00:00 AM|6/27/2021 12:00:00 AM",
    #             "19-Sep-21|W39-21|M09-21|Q3-21|9/19/2021 12:00:00 AM|9/19/2021 12:00:00 AM|8/22/2021 12:00:00 AM|6/27/2021 12:00:00 AM",
    #             "26-Sep-21|W40-21|M10-21|Q4-21|9/26/2021 12:00:00 AM|9/26/2021 12:00:00 AM|9/26/2021 12:00:00 AM|9/26/2021 12:00:00 AM",
    #             "03-Oct-21|W41-21|M10-21|Q4-21|10/3/2021 12:00:00 AM|10/3/2021 12:00:00 AM|9/26/2021 12:00:00 AM|9/26/2021 12:00:00 AM",
    #             "10-Oct-21|W42-21|M10-21|Q4-21|10/10/2021 12:00:00 AM|10/10/2021 12:00:00 AM|9/26/2021 12:00:00 AM|9/26/2021 12:00:00 AM",
    #             "17-Oct-21|W43-21|M10-21|Q4-21|10/17/2021 12:00:00 AM|10/17/2021 12:00:00 AM|9/26/2021 12:00:00 AM|9/26/2021 12:00:00 AM",
    #             "24-Oct-21|W44-21|M11-21|Q4-21|10/24/2021 12:00:00 AM|10/24/2021 12:00:00 AM|10/24/2021 12:00:00 AM|9/26/2021 12:00:00 AM",
    #             "31-Oct-21|W45-21|M11-21|Q4-21|10/31/2021 12:00:00 AM|10/31/2021 12:00:00 AM|10/24/2021 12:00:00 AM|9/26/2021 12:00:00 AM",
    #             "07-Nov-21|W46-21|M11-21|Q4-21|11/7/2021 12:00:00 AM|11/7/2021 12:00:00 AM|10/24/2021 12:00:00 AM|9/26/2021 12:00:00 AM",
    #             "14-Nov-21|W47-21|M11-21|Q4-21|11/14/2021 12:00:00 AM|11/14/2021 12:00:00 AM|10/24/2021 12:00:00 AM|9/26/2021 12:00:00 AM",
    #             "21-Nov-21|W48-21|M12-21|Q4-21|11/21/2021 12:00:00 AM|11/21/2021 12:00:00 AM|11/21/2021 12:00:00 AM|9/26/2021 12:00:00 AM",
    #             "28-Nov-21|W49-21|M12-21|Q4-21|11/28/2021 12:00:00 AM|11/28/2021 12:00:00 AM|11/21/2021 12:00:00 AM|9/26/2021 12:00:00 AM",
    #             "05-Dec-21|W50-21|M12-21|Q4-21|12/5/2021 12:00:00 AM|12/5/2021 12:00:00 AM|11/21/2021 12:00:00 AM|9/26/2021 12:00:00 AM",
    #             "12-Dec-21|W51-21|M12-21|Q4-21|12/12/2021 12:00:00 AM|12/12/2021 12:00:00 AM|11/21/2021 12:00:00 AM|9/26/2021 12:00:00 AM",
    #             "19-Dec-21|W52-21|M12-21|Q4-21|12/19/2021 12:00:00 AM|12/19/2021 12:00:00 AM|11/21/2021 12:00:00 AM|9/26/2021 12:00:00 AM",
    #         ],
    #     }
    # )
    # periods = 7
    # column_to_split = "Stat R Attribute Week"
    # time_delimiter = "|"
    # time_cols = [
    #     "DayName",
    #     "WeekName",
    #     "MonthName",
    #     "QuarterName",
    #     "DayKey",
    #     "WeekKey",
    #     "MonthKey",
    #     "QuarterKey",
    # ]
    # time_mapping = split_string(
    #     values=list(TimeDimension[column_to_split]),
    #     delimiter=time_delimiter,
    #     col_names=time_cols,
    # )
    # print(time_mapping)
    # latest_value = "W05-21"
    # time_attribute = {"WeekName": "WeekKey"}
    # print(
    #     get_n_time_periods(
    #         latest_value=latest_value,
    #         periods=-1 * 25,
    #         time_mapping=time_mapping,
    #         time_attribute=time_attribute,
    #         include_latest_value=True,
    #     )
    # )
    # print(
    #     get_n_time_periods(
    #         latest_value=latest_value,
    #         periods=periods,
    #         time_mapping=time_mapping,
    #         time_attribute=time_attribute,
    #         include_latest_value=True,
    #     )
    # )

    # from o9Reference.common_utils.common_utils import filter_relevant_time_mapping
    #
    # time_mapping = pd.DataFrame({
    #     'WeekName': ['W01-10', 'W02-10', 'W03-10', 'W04-10', 'W05-10'],
    #     'MonthName': ['M01-10', 'M01-10', 'M01-10', 'M01-10', 'M01-10'],
    #     'QuarterName': ['Q1-10', 'Q1-10', 'Q1-10', 'Q1-10', 'Q1-10'],
    #     'WeekKey': ['12/27/2009 12:00:00 AM', '1/3/2010 12:00:00 AM', '1/10/2010 12:00:00 AM',
    #                 '1/17/2010 12:00:00 AM', '1/24/2010 12:00:00 AM'],
    #     'MonthKey': ['12/27/2009 12:00:00 AM', '12/27/2009 12:00:00 AM', '12/27/2009 12:00:00 AM',
    #                  '12/27/2009 12:00:00 AM', '12/27/2009 12:00:00 AM'],
    #     'QuarterKey': ['12/27/2009 12:00:00 AM', '12/27/2009 12:00:00 AM', '12/27/2009 12:00:00 AM',
    #                    '12/27/2009 12:00:00 AM', '12/27/2009 12:00:00 AM', ]
    # })
    #
    # frequency = "Weekly"
    #
    # relevant_time_mapping = filter_relevant_time_mapping(
    #     frequency=frequency,
    #     time_mapping=time_mapping
    # )
    #
    # print(relevant_time_mapping)

    # from o9Reference.common_utils.common_utils import get_relevant_time_name_and_key
    # relevant_time_name, relevant_time_key = get_relevant_time_name_and_key(frequency="Quarterly")
    #
    # print(relevant_time_name)
    # print(relevant_time_key)

    # print(get_seasonal_periods(frequency="Weekly"))
    print(is_dimension("Sales Domain.[Customer Group]"))
    print(is_dimension("Stat Fcst L1"))
