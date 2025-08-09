"""
    Plugin : DP015SystemStat
    Version : 2022.08.01
    Maintained by : pmm_algocoe@o9solutions.com
"""
import logging

import numpy as np
import pandas as pd

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

from joblib import delayed
from joblib import Parallel
from o9Reference.common_utils.common_utils import filter_relevant_time_mapping
from o9Reference.common_utils.common_utils import (
    get_relevant_time_name_and_key,
)
from o9Reference.common_utils.data_utils import validate_output

from helpers.model_params import get_default_params
from helpers.models import train_models_for_one_intersection

from o9Reference.common_utils.function_timer import timed
from o9Reference.common_utils.common_utils import get_n_time_periods
from o9Reference.common_utils.common_utils import split_string
from o9Reference.common_utils.dataframe_utils import (
    concat_to_dataframe,
    get_o9_empty_df,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("o9_logger")

from o9Reference.common_utils.o9_memory_utils import _get_memory
import threading

from o9_common_utils.O9DataLake import O9DataLake
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from o9Reference.common_utils.common_utils import get_seasonal_periods

warnings.simplefilter("ignore", ConvergenceWarning)


# TODO: TBATS : In-sample : bug in sktime 0.7 library, resolved in sktime 0.10.0


def add_columns(df: pd.DataFrame, list_of_cols: list) -> pd.DataFrame:
    """
    Checks for missing columns and adds with a NaN value
    """
    for the_col in list_of_cols:
        if the_col not in list(df.columns):
            df[the_col] = np.nan
    return df


ALL_FORECAST_COLUMNS = [
    "Stat Fcst AR-NNET",
    "Stat Fcst AR-NNET 80% LB",
    "Stat Fcst AR-NNET 80% UB",
    "Stat Fcst Auto ARIMA",
    "Stat Fcst Auto ARIMA 80% LB",
    "Stat Fcst Auto ARIMA 80% UB",
    "Stat Fcst Croston",
    "Stat Fcst Croston 80% LB",
    "Stat Fcst Croston 80% UB",
    "Stat Fcst DES",
    "Stat Fcst DES 80% LB",
    "Stat Fcst DES 80% UB",
    "Stat Fcst ETS",
    "Stat Fcst ETS 80% LB",
    "Stat Fcst ETS 80% UB",
    "Stat Fcst Moving Average",
    "Stat Fcst Moving Average 80% LB",
    "Stat Fcst Moving Average 80% UB",
    "Stat Fcst Naive Random Walk",
    "Stat Fcst Naive Random Walk 80% LB",
    "Stat Fcst Naive Random Walk 80% UB",
    "Stat Fcst Prophet",
    "Stat Fcst Prophet 80% LB",
    "Stat Fcst Prophet 80% UB",
    "Stat Fcst SES",
    "Stat Fcst SES 80% LB",
    "Stat Fcst SES 80% UB",
    "Stat Fcst STLF",
    "Stat Fcst STLF 80% LB",
    "Stat Fcst STLF 80% UB",
    "Stat Fcst Seasonal Naive YoY",
    "Stat Fcst Seasonal Naive YoY 80% LB",
    "Stat Fcst Seasonal Naive YoY 80% UB",
    "Stat Fcst TBATS",
    "Stat Fcst TBATS 80% LB",
    "Stat Fcst TBATS 80% UB",
    "Stat Fcst TES",
    "Stat Fcst TES 80% LB",
    "Stat Fcst TES 80% UB",
    "Stat Fcst Theta",
    "Stat Fcst Theta 80% LB",
    "Stat Fcst Theta 80% UB",
    "Stat Fcst sARIMA",
    "Stat Fcst sARIMA 80% LB",
    "Stat Fcst sARIMA 80% UB",
]

ALL_FORECAST_MODEL_COLUMNS = [
    "AR-NNET Model",
    "Auto ARIMA Model",
    "Croston Model",
    "DES Model",
    "ETS Model",
    "Moving Average Model",
    "Naive Random Walk Model",
    "Prophet Model",
    "SES Model",
    "STLF Model",
    "Seasonal Naive YoY Model",
    "TBATS Model",
    "TES Model",
    "Theta Model",
    "sARIMA Model",
]

from o9Reference.common_utils.function_logger import log_inputs_and_outputs

from o9Reference.common_utils.dataframe_utils import ensure_valid_output_schema


@ensure_valid_output_schema
@log_inputs_and_outputs
@timed
def main(
    TimeLevel,
    Grain1,
    Grain2,
    Grain3,
    history_measure,
    AlgoList,
    Actual,
    TimeDimension,
    ForecastParameters,
    AlgoParameters,
    LastTimePeriod,
    StatSegment,
    ReadFromHive,
    df_keys,
):
    plugin_name = "DP015SystemStat"
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

        # Configurables
        time_col_to_split = "Stat R Attribute Week"
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
        # TODO: use Grain1, Grain2, Grain3 instead of this
        version_col = "Version.[Version Name]"
        stat_cust_group_col = "Sales Domain.[Stat Customer Group]"
        stat_location_col = "Location.[Stat Location]"
        stat_item_col = "Item.[Stat Item]"
        assigned_algo_list_col = "Assigned Algorithm List"
        time_week_col = "Time.[Week]"
        default_stat_param_value_col = "Default Stat Parameter Value"
        history_period_col = "History Period"
        forecast_period_col = "Forecast Period"
        validation_period_col = "Validation Period"
        history_time_buckets_col = "History Time Buckets"
        stat_parameter_col = "Stat Parameter.[Stat Parameter]"
        stat_algo_col = "Stat Algorithm.[Stat Algorithm]"
        system_stat_param_value_col = "System Stat Parameter Value"
        actual_col = "Actual"
        best_fit_method_col = "Bestfit Method"
        class_col = "Class.[Class]"
        prod_cust_segment_col = "Product Customer L1 Segment"
        confidence_interval_alpha = (
            0.20  # corresponds to 80% confidence interval
        )
        multiprocessing_num_cores = 4
        is_day_col = "Is Day"

        logger.info("history_measure : {}".format(history_measure))

        logger.info("Filtering relevant columns from input dataframes ...")
        # Filter only relevant columns from input dataframes
        req_cols = [
            version_col,
            stat_cust_group_col,
            stat_location_col,
            stat_item_col,
            assigned_algo_list_col,
        ]
        AlgoList = AlgoList[req_cols]

        req_cols = [
            version_col,
            stat_cust_group_col,
            stat_location_col,
            stat_item_col,
            TimeLevel,
            history_measure,
        ]
        Actual = Actual[req_cols]

        req_cols = [version_col, time_week_col, time_col_to_split]
        TimeDimension = TimeDimension[req_cols]

        req_cols = [
            version_col,
            history_period_col,
            forecast_period_col,
            validation_period_col,
            history_time_buckets_col,
            best_fit_method_col,
        ]
        ForecastParameters = ForecastParameters[req_cols]

        req_cols = [
            version_col,
            stat_parameter_col,
            stat_cust_group_col,
            stat_location_col,
            stat_item_col,
            stat_algo_col,
            system_stat_param_value_col,
        ]
        AlgoParameters = AlgoParameters[req_cols]

        req_cols = [version_col, TimeLevel, is_day_col]
        LastTimePeriod = LastTimePeriod[req_cols]

        req_cols = [
            version_col,
            class_col,
            stat_location_col,
            stat_cust_group_col,
            stat_item_col,
            prod_cust_segment_col,
        ]
        StatSegment = StatSegment[req_cols]

        AllForecast, ForecastModel = (
            get_o9_empty_df(),
            get_o9_empty_df(),
        )

        if Actual is None or len(Actual) == 0:
            logger.warning(
                "Actuals is None/Empty for slice : {}...".format(df_keys)
            )
            logger.warning("Returning empty dataframes as result ...")
            return AllForecast, ForecastModel

        if Actual[history_measure].sum() == 0:
            logger.warning(
                "Sum of Actuals is zero for slice : {} ...".format(df_keys)
            )
            logger.warning("Returning empty dataframes as result ...")
            return AllForecast, ForecastModel

        history_periods = int(ForecastParameters[history_period_col].iloc[0])
        forecast_periods = int(ForecastParameters[forecast_period_col].iloc[0])
        validation_periods = int(
            ForecastParameters[validation_period_col].iloc[0]
        )
        frequency = ForecastParameters[history_time_buckets_col].iloc[0]
        input_version = Actual[version_col].iloc[0]
        validation_method = str(
            ForecastParameters[best_fit_method_col].iloc[0]
        )

        seasonal_periods = get_seasonal_periods(frequency)

        logger.info(
            "-----------------------------------------------------------"
        )

        logger.info("time_cols : {}".format(time_cols))

        logger.info("Creating time mapping ...")
        logger.info("column_to_split : {}".format(time_col_to_split))

        # split on string and create dataframe
        time_mapping = split_string(
            values=list(TimeDimension[time_col_to_split]),
            delimiter=time_delimiter,
            col_names=time_cols,
        )
        logger.info("time mapping head :")
        logger.info(time_mapping.head())

        # Filter only relevant columns based on frequency
        relevant_time_mapping = filter_relevant_time_mapping(
            frequency, time_mapping
        )

        # Get relevant time name and key based on frequency
        relevant_time_name, relevant_time_key = get_relevant_time_name_and_key(
            frequency
        )

        # convert to datetime
        relevant_time_mapping[relevant_time_key] = relevant_time_mapping[
            relevant_time_key
        ].apply(pd.to_datetime, infer_datetime_format=True)

        logger.info("Extracting forecast level ...")

        # combine grains to get forecast level
        all_grains = [Grain1, Grain2, Grain3]
        forecast_level = [str(x) for x in all_grains if x != "NA" and x != ""]

        logger.info("---------------------------------------")

        logger.info("LastTimePeriod head : ")
        logger.info(LastTimePeriod.head())

        time_attribute_dict = {relevant_time_name: relevant_time_key}

        # note the negative sign to history periods
        latest_time_name = LastTimePeriod[TimeLevel].iloc[0]
        logger.info("Getting last {} period dates ...".format(history_periods))
        last_n_periods = get_n_time_periods(
            latest_time_name,
            -history_periods,
            relevant_time_mapping,
            time_attribute_dict,
        )

        if len(last_n_periods) == 0:
            logger.warning(
                "No dates found after filtering for {} periods for slice {}".format(
                    history_periods, df_keys
                )
            )
            logger.warning("Returning empty dataframe ...")
            return AllForecast, ForecastModel

        # filter relevant history based on dates provided above
        relevant_actuals = Actual[Actual[TimeLevel].isin(last_n_periods)]

        if (
            len(relevant_actuals) == 0
            or relevant_actuals[history_measure].sum() == 0
        ):
            logger.warning(
                "Actuals is Empty/Sum of actuals is zero after filtering {} periods of history for slice : {}...".format(
                    history_periods, df_keys
                )
            )
            logger.warning("Returning empty dataframes as result ...")
            return AllForecast, ForecastModel

        if len(StatSegment) == 0:
            logger.warning(
                "StatSegment input is empty for slice : {}".format(df_keys)
            )
            logger.warning("Returning empty dataframe ...")
            return AllForecast, ForecastModel

        # exclude the discontinued intersections from actuals, AlgoList
        non_disc_intersections = StatSegment[StatSegment[class_col] != "DISC"]
        non_disc_intersections = non_disc_intersections[
            forecast_level
        ].drop_duplicates()

        if len(non_disc_intersections) == 0:
            logger.warning(
                "No data to process after excluding DISC intersections for slice : {}...".format(
                    df_keys
                )
            )
            return AllForecast, ForecastModel

        logger.info("Excluding the discontinued intersections ...")
        relevant_actuals = relevant_actuals.merge(
            non_disc_intersections, how="inner", on=forecast_level
        )

        # TODO : If algo list is not assigned, value will be N/A, need to filter out rows where value is not N/A
        if len(AlgoList) == 0:
            logger.warning(
                "No records found in AlgoList variable for slice : {} ...".format(
                    df_keys
                )
            )
            logger.warning(
                "Kindly run 'Assign Rule and Algo' action button from Forecast Setup screen and trigger this plugin again ..."
            )
            return AllForecast, ForecastModel

        AlgoList = AlgoList.merge(
            non_disc_intersections, how="inner", on=forecast_level
        )

        # create intersections dataframe
        intersections_master = relevant_actuals[
            forecast_level
        ].drop_duplicates()

        if len(AlgoParameters) == 0:
            logger.info(
                "No AlgoParameters supplied, creating master list of algo params for all intersections ..."
            )
            AlgoParameters = get_default_params(
                stat_algo_col,
                stat_parameter_col,
                system_stat_param_value_col,
                frequency,
                intersections_master,
            )

        else:
            logger.info(
                "AlgoParameters supplied, shape : {} ...".format(
                    AlgoParameters.shape
                )
            )
            logger.info(
                "Joining with default params to populate values for all intersections ..."
            )

            DefaultParameters = get_default_params(
                stat_algo_col,
                stat_parameter_col,
                default_stat_param_value_col,
                frequency,
                intersections_master,
            )

            AlgoParameters = AlgoParameters.merge(
                DefaultParameters, how="right"
            )

            AlgoParameters[system_stat_param_value_col] = np.where(
                AlgoParameters[system_stat_param_value_col].isna(),
                AlgoParameters[default_stat_param_value_col],
                AlgoParameters[system_stat_param_value_col],
            )

        # get the forecast dates
        forecast_period_dates = get_n_time_periods(
            latest_time_name,
            forecast_periods,
            relevant_time_mapping,
            time_attribute_dict,
            include_latest_value=False,
        )

        # check confidence interval value
        if confidence_interval_alpha > 1 or confidence_interval_alpha < 0:
            logger.warning(
                "confidence interval parameter should be less than one ..."
            )
            default_conf_interval = 0.20
            logger.warning(
                "overriding with default value of {}".format(
                    default_conf_interval
                )
            )
            confidence_interval_alpha = default_conf_interval

        logger.info(
            "Total Num of intersections : {}".format(len(intersections_master))
        )

        logger.info("Running forecasting for all intersections ...")

        # join on time mapping and sort by key
        relevant_actuals = relevant_actuals.merge(
            relevant_time_mapping,
            left_on=TimeLevel,
            right_on=relevant_time_name,
            how="inner",
        )
        relevant_actuals.sort_values(
            forecast_level + [relevant_time_key], inplace=True
        )
        relevant_actuals.drop(
            [relevant_time_name, relevant_time_key], axis=1, inplace=True
        )

        all_results = Parallel(n_jobs=multiprocessing_num_cores, verbose=1)(
            delayed(train_models_for_one_intersection)(
                df,
                forecast_level,
                TimeLevel,
                history_measure,
                validation_periods,
                validation_method,
                AlgoList,
                seasonal_periods,
                forecast_period_dates,
                confidence_interval_alpha,
                assigned_algo_list_col,
                AlgoParameters,
                stat_algo_col,
                stat_parameter_col,
                system_stat_param_value_col,
            )
            for name, df in relevant_actuals.groupby(forecast_level)
        )

        logger.info("Collected results from parallel processing ...")

        # collect separate lists from the list of tuples returned by multiprocessing function
        all_validation_pred = [x[0] for x in all_results]
        all_forecasts = [x[1] for x in all_results]
        all_model_descriptions = [x[2] for x in all_results]

        # Concat all validation predictions to one dataframe, format output
        all_validation_pred_df = concat_to_dataframe(all_validation_pred)

        if len(all_validation_pred_df) == 0:
            logger.warning(
                "No records in all_validation predictions for slice : {}".format(
                    df_keys
                )
            )
            return AllForecast, ForecastModel

        all_validation_pred_df[version_col] = input_version

        req_col_order = (
            [version_col]
            + forecast_level
            + [TimeLevel]
            + [x for x in all_validation_pred_df.columns if "Stat Fcst" in x]
        )
        all_validation_pred_df = all_validation_pred_df[req_col_order]
        logger.info(" ---------- all_validation_pred_df : head----------")
        logger.info(all_validation_pred_df.head())

        logger.info("Validating output in validation predictions ...")
        validate_output(
            input_df=relevant_actuals,
            output_df=all_validation_pred_df,
            forecast_level=forecast_level,
        )

        # Concat all future predictions to one dataframe, format output
        AllForecast = concat_to_dataframe(all_forecasts)
        AllForecast[version_col] = input_version
        req_col_order = (
            [version_col]
            + forecast_level
            + [TimeLevel]
            + [x for x in AllForecast.columns if "Stat Fcst" in x]
        )
        AllForecast = AllForecast[req_col_order]

        # combine validation and future dates into a single dataframe
        AllForecast = all_validation_pred_df.append(
            AllForecast, ignore_index=True
        )

        if ReadFromHive:
            # Add missing columns
            AllForecast = add_columns(
                df=AllForecast, list_of_cols=ALL_FORECAST_COLUMNS
            )

        logger.info(" ---------- AllForecast : head----------")
        logger.info(AllForecast.head())

        # combine all model descriptions to one dataframe
        ForecastModel = concat_to_dataframe(all_model_descriptions)
        ForecastModel.insert(0, version_col, input_version)

        if ReadFromHive:
            ForecastModel = add_columns(
                df=ForecastModel, list_of_cols=ALL_FORECAST_MODEL_COLUMNS
            )

        logger.info(" ---------- ForecastModel : head----------")
        logger.info(ForecastModel.head())

        logger.info("Successfully executed {} ...".format(plugin_name))
        logger.info("---------------------------------------------")
    except Exception as e:
        logger.error(
            "Exception for slice : {}, returning empty dataframe as output ...".format(
                df_keys
            )
        )
        logger.exception(e)
        AllForecast = get_o9_empty_df()
        ForecastModel = get_o9_empty_df()

    return AllForecast, ForecastModel


