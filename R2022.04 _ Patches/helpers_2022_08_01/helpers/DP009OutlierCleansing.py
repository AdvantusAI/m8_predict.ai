"""
Plugin : DP009OutlierCleansing
Version : 2022.08.01
Maintained by : pmm_algocoe@o9solutions.com
"""
import logging

import pandas as pd
from o9Reference.common_utils.function_timer import timed

from helpers.outlier_correction import cleanse_data_wrapper

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

from o9Reference.common_utils.common_utils import (
    split_string,
    get_n_time_periods,
    filter_relevant_time_mapping,
    get_relevant_time_name_and_key,
)

from o9Reference.common_utils.fill_missing_dates import fill_missing_dates

from o9Reference.common_utils.dataframe_utils import (
    concat_to_dataframe,
    get_o9_empty_df,
)
from o9Reference.common_utils.common_utils import get_seasonal_periods

from joblib import delayed
from joblib import Parallel
from o9Reference.common_utils.data_utils import validate_output

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("o9_logger")

from o9Reference.common_utils.o9_memory_utils import _get_memory
import threading

from o9_common_utils.O9DataLake import O9DataLake

from o9Reference.common_utils.function_logger import log_inputs_and_outputs

from o9Reference.common_utils.dataframe_utils import ensure_valid_output_schema


@ensure_valid_output_schema
@log_inputs_and_outputs
@timed
def main(
    VersionLevel,
    TimeLevel,
    Actual,
    Grain2,
    Grain1,
    Grain3,
    history_measure,
    TimeDimension,
    LastTimePeriod,
    HistoryPeriod,
    OutlierParameters,
    ReadFromHive,
    df_keys,
):
    plugin_name = "DP009OutlierCleansing"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))
    try:

        # assert and convert string value to boolean
        assert ReadFromHive in [
            "True",
            "False",
        ], "'{}' is invalid, Allowed values are True/False ...".format(
            ReadFromHive
        )
        ReadFromHive = eval(ReadFromHive)

        # Configurables - define all column names here
        column_to_split = "Stat R Attribute Week"
        time_delimiter = "|"
        time_cols = [
            "DayName",
            "WeekName",
            "MonthName",
            "QuarterName",
            "DayKey",
            "WeekKey",
            "MonthKey",
            "QuarterKey",
        ]
        time_bucket_col = "History Time Buckets"
        history_measure_col = "History Measure"

        # Outlier Parameters
        upper_threshold_col = "Outlier Upper Threshold Limit"
        lower_threshold_col = "Outlier Lower Threshold Limit"
        outlier_correction_col = "Outlier Correction"
        outlier_method_col = "Outlier Method"
        outlier_period_col = "History Period"

        # output measures
        cleansed_data_col = "Actual Cleansed System"
        upper_bound_col = "Outlier Upper Threshold"
        lower_bound_col = "Outlier Lower Threshold"
        actual_mean_col = "Actual Median"

        multiprocessing_num_cores = 4
        logger.info(
            "multiprocessing_num_cores : {}".format(multiprocessing_num_cores)
        )

        logger.info("Extracting dimension cols ...")
        # combine grains to get granular level
        all_grains = [Grain1, Grain2, Grain3]
        dimensions = [str(x) for x in all_grains if x != "NA" and x != ""]
        logger.info("dimensions : {} ...".format(dimensions))

        assert len(dimensions) > 0, "dimensions cannot be empty ..."

        cols_required_in_output = (
            [VersionLevel]
            + dimensions
            + [TimeLevel, upper_bound_col, lower_bound_col, cleansed_data_col]
        )

        # ---------------------------------------------------
        # Actuals might not be present for a particular slice, check and return empty dataframe
        if Actual is None or len(Actual) == 0:
            logger.warning(
                "Actuals is None/Empty for slice : {} ...".format(df_keys)
            )
            logger.warning("Returning empty dataframes as result ...")
            return get_o9_empty_df()

        input_version = Actual[VersionLevel].iloc[0]

        # extract data from HistoryPeriod
        frequency = str(HistoryPeriod[time_bucket_col].iloc[0])
        logger.info("frequency : {}".format(frequency))

        relevant_time_name, relevant_time_key = get_relevant_time_name_and_key(
            frequency=frequency
        )

        if ReadFromHive:
            history_measure = "DP009" + history_measure

        logger.info("history_measure : {}".format(history_measure))

        outlier_period = int(HistoryPeriod[outlier_period_col].iloc[0])
        logger.info("outlier_period : {}".format(outlier_period))

        cols_required_in_Actual = [TimeLevel] + dimensions + [history_measure]

        for the_col in cols_required_in_Actual:
            assert the_col in list(
                Actual.columns
            ), "{} missing in Actual dataframe ...".format(the_col)

        logger.info(
            "filtering {} columns from Actual df ...".format(
                cols_required_in_Actual
            )
        )

        Actual = Actual[cols_required_in_Actual]

        logger.info("Creating time mapping ...")
        # split on string and create dataframe
        time_mapping = split_string(
            values=list(TimeDimension[column_to_split]),
            delimiter=time_delimiter,
            col_names=time_cols,
        )
        logger.info("time mapping head :")
        logger.info(time_mapping.head())

        if len(time_mapping) == 0:
            logger.warning(
                "time mapping cannot be empty for slice : {}...".format(
                    df_keys
                )
            )
            logger.warning("Returning empty dataframes as result ...")
            return pd.DataFrame(columns=cols_required_in_output)

        # Filter only relevant columns based on frequency
        relevant_time_mapping = filter_relevant_time_mapping(
            frequency, time_mapping
        )

        # convert to datetime
        relevant_time_mapping[relevant_time_key] = relevant_time_mapping[
            relevant_time_key
        ].apply(pd.to_datetime, infer_datetime_format=True)

        if len(relevant_time_mapping) == 0:
            logger.warning(
                "time mapping empty after dropping duplicates for slice : {}...".format(
                    df_keys
                )
            )
            logger.warning("Returning empty dataframes as result ...")
            return pd.DataFrame(columns=cols_required_in_output)

        logger.info(
            "relevant_time_mapping shape : {}".format(
                relevant_time_mapping.shape
            )
        )

        time_series_freq = get_seasonal_periods(frequency)

        logger.info("LastTimePeriod head : ")
        logger.info(LastTimePeriod.head())

        latest_time_name = LastTimePeriod[TimeLevel].iloc[0]
        logger.info("latest_time_name : {} ...".format(latest_time_name))

        time_attribute_dict = {relevant_time_name: relevant_time_key}

        logger.info(
            "Getting last {} period dates for history period ...".format(
                outlier_period
            )
        )
        last_n_periods_history = get_n_time_periods(
            latest_time_name,
            -outlier_period,
            relevant_time_mapping,
            time_attribute_dict,
            include_latest_value=True,
        )

        # fill missing dates
        relevant_history_nas_filled = fill_missing_dates(
            actual=Actual,
            forecast_level=dimensions,
            time_mapping=relevant_time_mapping,
            history_measure=history_measure,
            relevant_time_name=relevant_time_name,
            relevant_time_key=relevant_time_key,
            relevant_time_periods=last_n_periods_history,
            time_level=TimeLevel,
            fill_nulls_with_zero=True,
        )

        if len(OutlierParameters) == 0:
            logger.warning(
                "OutlierParameters df is empty for slice : {}".format(df_keys)
            )
            logger.warning("Returning empty dataframe as result ...")
            return pd.DataFrame(columns=cols_required_in_output)

        logger.info("Performing outlier correction for all intersections ...")

        all_results = Parallel(n_jobs=multiprocessing_num_cores, verbose=1)(
            delayed(cleanse_data_wrapper)(
                group,
                dimensions,
                relevant_time_key,
                history_measure,
                cleansed_data_col,
                upper_bound_col,
                lower_bound_col,
                actual_mean_col,
                TimeLevel,
                OutlierParameters,
                upper_threshold_col,
                lower_threshold_col,
                outlier_method_col,
                outlier_correction_col,
                time_series_freq,
            )
            for name, group in relevant_history_nas_filled.groupby(
                dimensions, observed=True
            )
        )

        # Concatenate all results to one dataframe
        output_df = concat_to_dataframe(all_results)

        if len(output_df) == 0:
            logger.warning(
                "No records after processing for slice : {}, returning empty dataframe".format(
                    df_keys
                )
            )
            return pd.DataFrame(columns=cols_required_in_output)

        # validate output, print warning messages if intersections are missing
        validate_output(
            input_df=relevant_history_nas_filled,
            output_df=output_df,
            forecast_level=dimensions,
        )

        # Add input version
        output_df.insert(loc=0, column=VersionLevel, value=input_version)
        logger.info("--------- output_df : head")
        logger.info(output_df.head())

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.error(
            "Exception for slice : {}, returning empty dataframe as output ...".format(
                df_keys
            )
        )
        logger.exception(e)
        output_df = get_o9_empty_df()

    return output_df


