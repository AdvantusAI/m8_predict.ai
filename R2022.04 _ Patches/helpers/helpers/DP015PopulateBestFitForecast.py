"""
    Plugin : DP015PopulateBestFitForecast
    Version : R2022.05
    Maintained by : dpref@o9solutions.com
"""
import logging

import pandas as pd

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("o9_logger")

from o9Reference.common_utils.o9_memory_utils import _get_memory
import threading

# Start a thread to print memory occasionally, change sleep seconds if required,
# Since thread is daemon, it's closed automatically with main script.
back_thread = threading.Thread(
    target=_get_memory,
    kwargs=dict(max_memory=0.0, sleep_seconds=90),
    daemon=True,
)
logger.info("Starting background thread for memory profiling ...")
back_thread.start()

from o9Reference.common_utils.common_utils import (
    split_string,
    get_relevant_time_name_and_key,
)

from o9_common_utils.O9DataLake import O9DataLake

from o9Reference.common_utils.common_utils import (
    get_n_time_periods,
    filter_relevant_time_mapping,
)
from o9Reference.common_utils.function_timer import timed


@timed
def main(
    TimeLevel,
    Grain1,
    Grain2,
    Grain3,
    TimeDimension,
    ForecastParameters,
    LastTimePeriod,
    ForecastData,
    ForecastBounds,
    BestFitAlgo,
    df_keys,
):
    plugin_name = "DP015PopulateBestFitForecast"
    logger.info("Executing {} ...".format(plugin_name))
    dummy_output_cols = ["Version.[Version Name]", "History Measure"]

    try:
        # Configurables
        stat_r_attribute_week_col = "Stat R Attribute Week"
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
        version_col = "Version.[Version Name]"
        forecast_period_col = "Forecast Period"
        validation_period_col = "Validation Period"
        history_time_buckets_col = "History Time Buckets"
        error_metric_col = "Error Metric"
        history_period_col = "History Period"
        best_fit_method_col = "Bestfit Method"
        is_day_col = "Is Day"
        system_best_fit_algo_col = "System Bestfit Algorithm"
        planner_best_fit_algo_col = "Planner Bestfit Algorithm"
        best_fit_col = "System Stat Fcst L1"
        best_fit_ub_algo_col = system_best_fit_algo_col + " UB"
        best_fit_lb_algo_col = system_best_fit_algo_col + " LB"
        best_fit_ub_col = best_fit_col + " 80% UB"
        best_fit_lb_col = best_fit_col + " 80% LB"
        stat_fcst_str = "Stat Fcst "
        multiprocessing_num_cores = 4
        time_week_col = "Time.[Week]"

        stat_fcst_cols = [x for x in ForecastData.columns if "Stat Fcst" in x]

        req_cols = [
            version_col,
            Grain1,
            Grain2,
            Grain3,
            TimeLevel,
        ] + stat_fcst_cols
        ForecastData = ForecastData[req_cols]

        req_cols = [
            version_col,
            time_week_col,
            stat_r_attribute_week_col,
        ]
        TimeDimension = TimeDimension[req_cols]

        req_cols = [
            version_col,
            history_period_col,
            forecast_period_col,
            validation_period_col,
            best_fit_method_col,
            error_metric_col,
            history_time_buckets_col,
        ]
        ForecastParameters = ForecastParameters[req_cols]

        req_cols = [TimeLevel, version_col, is_day_col]
        LastTimePeriod = LastTimePeriod[req_cols]

        req_cols = [
            version_col,
            Grain1,
            Grain2,
            Grain3,
            system_best_fit_algo_col,
            planner_best_fit_algo_col,
        ]
        BestFitAlgo = BestFitAlgo[req_cols]

        if len(BestFitAlgo) == 0:
            logger.warning(
                "BestFitAlgo is empty for slice : {}".format(df_keys)
            )
            logger.warning("Returning empty dataframe ...")
            return pd.DataFrame(columns=dummy_output_cols)

        # Use planner best fit algo wherever available
        BestFitAlgo[planner_best_fit_algo_col].fillna(
            BestFitAlgo[system_best_fit_algo_col], inplace=True
        )
        # Drop system best fit column since not needed further
        BestFitAlgo.drop(system_best_fit_algo_col, axis=1, inplace=True)

        # Drop NAs from BestFitAlgo
        BestFitAlgo.dropna(inplace=True)
        # Drop version col
        BestFitAlgo.drop(version_col, axis=1, inplace=True)

        logger.info(
            "-----------------------------------------------------------"
        )

        logger.info("--------- Grain1 : {}".format(Grain1))
        logger.info("--------- Grain2 : {}".format(Grain2))
        logger.info("--------- Grain3 : {}".format(Grain3))
        logger.info("--------- TimeLevel : {}".format(TimeLevel))

        logger.info("time_cols : {}".format(time_cols))

        logger.info("Creating time mapping ...")
        logger.info("column_to_split : {}".format(stat_r_attribute_week_col))

        # split on string and create dataframe
        time_mapping = split_string(
            values=list(TimeDimension[stat_r_attribute_week_col]),
            delimiter=time_delimiter,
            col_names=time_cols,
        )
        logger.info("time mapping head :")
        logger.info(time_mapping.head())

        frequency = ForecastParameters[history_time_buckets_col].iloc[0]
        logger.info("frequency : {}".format(frequency))

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

        forecast_periods = int(ForecastParameters[forecast_period_col].iloc[0])

        logger.info(
            "multiprocessing_num_cores : {}".format(multiprocessing_num_cores)
        )

        logger.info("Extracting forecast level ...")

        # combine grains to get forecast level
        all_grains = [Grain1, Grain2, Grain3]
        forecast_level = [str(x) for x in all_grains if x != "NA" and x != ""]

        logger.info("LastTimePeriod head : ")
        logger.info(LastTimePeriod.head())

        time_attribute_dict = {relevant_time_name: relevant_time_key}

        # get the validation periods
        # note the negative sign to history periods
        latest_time_name = LastTimePeriod[TimeLevel].iloc[0]

        # Define output columns and dataframes
        BestFitForecast_req_cols = (
            [version_col]
            + forecast_level
            + [TimeLevel, best_fit_col, best_fit_lb_col, best_fit_ub_col]
        )

        BestFitForecast = pd.DataFrame(columns=BestFitForecast_req_cols)

        # get the forecast period dates
        forecast_period_dates = get_n_time_periods(
            latest_time_name,
            forecast_periods,
            relevant_time_mapping,
            time_attribute_dict,
            include_latest_value=False,
        )

        logger.info("forecast_period_dates : {}".format(forecast_period_dates))

        # filter forecast data
        AllForecast = ForecastData[
            ForecastData[TimeLevel].isin(forecast_period_dates)
        ]

        if len(AllForecast) == 0:
            logger.warning(
                "No data found after filtering for forecast_period_dates for slice : {}".format(
                    df_keys
                )
            )
            logger.warning("Returning empty dataframe ...")
            return BestFitForecast

        # merge best fit columns to master df containing all forecast
        AllForecast = AllForecast.merge(
            BestFitAlgo, how="left", on=forecast_level
        )

        # if best fit is not available for any of the combinations, exit out
        if AllForecast[planner_best_fit_algo_col].isnull().all():
            logger.warning(
                "Best fit algorithm not present for any combination for slice : {}".format(
                    df_keys
                )
            )
            logger.warning("Returning empty dataframe")
            return pd.DataFrame(columns=dummy_output_cols)

        # Check best fit algorithms is NA in any of the intersections
        intersections_with_no_bestfit = AllForecast[
            AllForecast[planner_best_fit_algo_col].isna()
        ][forecast_level].drop_duplicates()
        if len(intersections_with_no_bestfit) > 0:
            logger.warning(
                "Best fit algorithm is missing for {} intersections, printing top 5, for slice : {} ...".format(
                    len(intersections_with_no_bestfit), df_keys
                )
            )
            logger.warning(intersections_with_no_bestfit.head())
            logger.warning(
                "intersections_with_no_bestfit, shape : {}".format(
                    intersections_with_no_bestfit.shape
                )
            )

        # Filter rows where best fit columns is not NA
        AllForecast = AllForecast[
            AllForecast[planner_best_fit_algo_col].notna()
        ]

        # Best Fit column contains only the Model name, need to prefix Stat Fcst for lookup to work
        # Example : As of now  planner_best_fit_algo_col contains 'DES', need to make it 'Stat Fcst DES'
        AllForecast[planner_best_fit_algo_col] = stat_fcst_str + AllForecast[
            planner_best_fit_algo_col
        ].astype("str")

        # lookup and populate best fit prediction
        AllForecast[best_fit_col] = AllForecast.lookup(
            AllForecast.index, AllForecast[planner_best_fit_algo_col]
        )

        AllForecast[best_fit_ub_algo_col] = (
            AllForecast[planner_best_fit_algo_col].astype("str") + " 80% UB"
        )
        AllForecast[best_fit_lb_algo_col] = (
            AllForecast[planner_best_fit_algo_col].astype("str") + " 80% LB"
        )

        # Join forecast bounds with forecasts dataframe
        AllForecast = AllForecast.merge(
            ForecastBounds.drop(version_col, axis=1),
            on=forecast_level + [TimeLevel],
        )

        # lookup and populate best fit prediction for upper bound and lower bound
        AllForecast[best_fit_ub_col] = AllForecast.lookup(
            AllForecast.index, AllForecast[best_fit_ub_algo_col]
        )
        AllForecast[best_fit_lb_col] = AllForecast.lookup(
            AllForecast.index, AllForecast[best_fit_lb_algo_col]
        )

        # Collect only the required columns
        BestFitForecast = AllForecast[BestFitForecast_req_cols]

        logger.info("------------ BestFitForecast : head -----------")
        logger.info(BestFitForecast.head())

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.error(
            "Exception for slice : {}, returning empty dataframe as output ...".format(
                df_keys
            )
        )
        logger.exception(e)
        BestFitForecast = pd.DataFrame(columns=dummy_output_cols)

    return BestFitForecast