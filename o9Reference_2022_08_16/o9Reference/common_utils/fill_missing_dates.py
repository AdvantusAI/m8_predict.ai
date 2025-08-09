"""
Version : 1.0.0
Maintained by : dpref@o9solutions.com
"""
import logging

import numpy as np
import pandas as pd

from o9Reference.common_utils.dataframe_utils import create_cartesian_product

logger = logging.getLogger("o9_logger")


def fill_missing_dates_pandas(
    df: pd.DataFrame,
    dimensions: list,
    relevant_time_key: str,
    end_date: pd.datetime,
    history_measure: str,
    time_series_freq: str,
):
    """
    Fills the missing dates with zero if provided with a valid pandas date frequency.
    """
    resampled = pd.DataFrame()
    if len(df) == 0:
        return resampled

    assert (
        relevant_time_key in df.columns
    ), "{} should be present in df ...".format(relevant_time_key)
    assert (
        history_measure in df.columns
    ), "{} should be present in df ...".format(history_measure)

    try:
        intersection_min_date = df[relevant_time_key].min()
        intersection_date_range = pd.date_range(
            name=relevant_time_key,
            start=intersection_min_date,
            end=end_date,
            freq=time_series_freq,
        )

        # create resampled dataframe
        resampled = (
            df.set_index(relevant_time_key)
            .reindex(intersection_date_range)
            .reset_index()
        )

        # Fill nulls and cap negatives
        resampled[history_measure].fillna(0, inplace=True)
        resampled[history_measure] = np.where(
            resampled[history_measure] < 0, 0, resampled[history_measure]
        )

        # Fill missing rows in dimension columns
        for the_dim in dimensions:
            resampled[the_dim] = df[the_dim].unique()[0]

        resampled = resampled[
            dimensions + [relevant_time_key, history_measure]
        ]

    except Exception as e:
        logger.error("Exception for {}".format(df[dimensions].values))
        logger.exception(e)

    return resampled


def fill_missing_dates(
    actual: pd.DataFrame,
    forecast_level: list,
    time_level: str,
    history_measure: str,
    relevant_time_name: str,
    relevant_time_key: str,
    relevant_time_periods: list,
    time_mapping: pd.DataFrame,
    fill_nulls_with_zero=True,
    filter_from_start_date=True,
) -> pd.DataFrame:
    assert isinstance(actual, pd.DataFrame), "Datatype error : actual"
    assert isinstance(forecast_level, list), "Datatype error : forecast_level,"
    assert isinstance(time_level, str), "Datatype error : time_level"
    assert isinstance(history_measure, str), "Datatype error : history_measure"
    assert isinstance(
        relevant_time_name, str
    ), "Datatype error : relevant_time_name"
    assert isinstance(
        relevant_time_key, str
    ), "Datatype error : relevant_time_key"
    assert isinstance(
        relevant_time_periods, list
    ), "Datatype error : relevant_time_periods"
    assert isinstance(
        time_mapping, pd.DataFrame
    ), "Datatype error : time_mapping"

    relevant_actuals_nas_filled = pd.DataFrame()
    try:
        for the_col in forecast_level:
            assert (
                the_col in actual.columns
            ), "{} not present in actual".format(the_col)

        assert (
            time_level in actual.columns
        ), "time_level not present in actual ..."

        # join actual with time mapping, store the intersection start dates
        actual_with_time_key = actual.copy().merge(
            time_mapping,
            left_on=time_level,
            right_on=relevant_time_name,
            how="left",
        )
        actual_with_time_key.drop(relevant_time_name, axis=1, inplace=True)
        start_date_col = "start_date"
        intersection_start_dates_df = (
            actual_with_time_key.groupby(forecast_level, observed=True)[
                relevant_time_key
            ]
            .min()
            .reset_index()
        )
        intersection_start_dates_df.rename(
            columns={relevant_time_key: start_date_col}, inplace=True
        )

        # create intersections dataframe
        intersections_master = actual_with_time_key[
            forecast_level
        ].drop_duplicates()

        # convert to df for join
        last_n_period_df = pd.DataFrame({time_level: relevant_time_periods})

        # convert list of dfs to df which has cartesian product of time and forecast level
        time_with_intersections_cartesian = create_cartesian_product(
            intersections_master, last_n_period_df
        )

        # join with time master
        relevant_actuals_nas_filled = actual_with_time_key.merge(
            time_with_intersections_cartesian,
            how="right",
            on=forecast_level + [time_level],
        )

        # populate missing entries in the date key column
        date_name_to_key_mapping = dict(
            zip(
                list(time_mapping[relevant_time_name]),
                list(time_mapping[relevant_time_key]),
            )
        )
        relevant_actuals_nas_filled[
            relevant_time_key
        ] = relevant_actuals_nas_filled[time_level].map(
            date_name_to_key_mapping
        )

        # fill NAs
        if fill_nulls_with_zero:
            relevant_actuals_nas_filled[history_measure].fillna(
                0, inplace=True
            )

        # join with intersection start dates
        relevant_actuals_nas_filled = relevant_actuals_nas_filled.merge(
            intersection_start_dates_df, on=forecast_level
        )

        if filter_from_start_date:
            # filter relevant records where record date is greater than the intersection start date
            filter_clause = (
                relevant_actuals_nas_filled[relevant_time_key]
                >= relevant_actuals_nas_filled[start_date_col]
            )
            relevant_actuals_nas_filled = relevant_actuals_nas_filled[
                filter_clause
            ]

        # drop the start date col, sort by intersections and time key
        relevant_actuals_nas_filled.drop(start_date_col, axis=1, inplace=True)
        relevant_actuals_nas_filled.sort_values(
            forecast_level + [relevant_time_key], inplace=True
        )
        relevant_actuals_nas_filled.reset_index(drop=True, inplace=True)

        # saw cases where version column is present in dataframe and not populated in output
        version_col = "Version.[Version Name]"
        if version_col in relevant_actuals_nas_filled.columns:
            # collect existing version value
            version_fill_value = relevant_actuals_nas_filled[
                version_col
            ].unique()[0]
            # fill in place
            relevant_actuals_nas_filled[version_col].fillna(
                version_fill_value, inplace=True
            )

    except Exception as e:
        logger.exception("Exception {} from fill_missing_dates ...".format(e))

    return relevant_actuals_nas_filled


if __name__ == "__main__":
    pass
    # from o9Reference.common_utils.fill_missing_dates import fill_missing_dates_pandas
    # import random
    # input_df = pd.DataFrame({
    #     'Item':['I1', 'I1', 'I1', 'I1', 'I1', 'I2', 'I2', 'I2', 'I2', 'I2'],
    #     'Customer': ['C1','C1','C1','C1','C1','C1','C1','C1','C1','C1'],
    #     'Date':pd.date_range(start='2020-01-01', periods=10),
    #     'Actuals':[random.randrange(1, 25, 1) for i in range(10)]
    # })
    #
    # # remove rows at random
    # input_df.drop(index=[2, 7, 8], inplace=True)
    #
    # print(input_df)
    #
    # dates_filled = fill_missing_dates_pandas(
    #     df=input_df,
    #     dimensions=['Item', 'Customer'],
    #     relevant_time_key="Date",
    #     end_date=pd.to_datetime('2020-01-10'),
    #     history_measure="Actuals",
    #     time_series_freq="D"
    # )
    #
    # print(dates_filled)

    # from o9Reference.common_utils.fill_missing_dates import fill_missing_dates
    import random

    input_df = pd.DataFrame(
        {
            "Version.[Version Name]": ["CurrentWorkingView"] * 10,
            "Item": [
                "I1",
                "I1",
                "I1",
                "I1",
                "I1",
                "I2",
                "I2",
                "I2",
                "I2",
                "I2",
            ],
            "Customer": [
                "C1",
                "C1",
                "C1",
                "C1",
                "C1",
                "C1",
                "C1",
                "C1",
                "C1",
                "C1",
            ],
            "Date": ["D" + str(x) for x in range(1, 11)],
            "Actuals": [random.randrange(1, 25, 1) for i in range(10)],
        }
    )

    # remove rows at random to mock missing date behavior
    input_df.drop(index=[2, 7, 8], inplace=True)

    print(input_df)

    time_mapping = pd.DataFrame(
        {
            "DayName": ["D" + str(x) for x in range(1, 16)],
            "DayKey": pd.date_range(start="2019-12-25", periods=15),
        }
    )

    last_n_periods = ["D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12"]
    #
    dates_filled = fill_missing_dates(
        actual=input_df,
        forecast_level=["Item", "Customer"],
        time_level="Date",
        history_measure="Actuals",
        relevant_time_name="DayName",
        relevant_time_key="DayKey",
        relevant_time_periods=last_n_periods,
        time_mapping=time_mapping,
        fill_nulls_with_zero=True,
    )

    print(dates_filled)
