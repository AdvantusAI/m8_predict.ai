"""
    Plugin : DP015IdentifyBestFitModel
    Version : 2022.08.16
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

from o9Reference.common_utils.data_utils import (
    check_for_columns_with_same_value,
    validate_output,
)
from o9Reference.stat_utils.calculate_error import calculate_error

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("o9_logger")

from o9Reference.common_utils.o9_memory_utils import _get_memory
import threading

from o9Reference.common_utils.common_utils import (
    split_string,
    get_relevant_time_name_and_key,
)

from o9_common_utils.O9DataLake import O9DataLake

from o9Reference.common_utils.dataframe_utils import (
    concat_to_dataframe,
    get_o9_empty_df,
)
from o9Reference.common_utils.common_utils import (
    get_n_time_periods,
    filter_relevant_time_mapping,
)
from joblib import delayed
from joblib import Parallel
from o9Reference.common_utils.function_timer import timed


# calculate errors for each intersection, each pass, each algorithm:
def calc_validation_error(
    df: pd.DataFrame,
    forecast_level: list,
    error_metric: str,
    override_flat_line_forecasts: bool,
    forecast_cols: list,
    history_measure: str,
    round_decimals: int,
):
    the_intersection = df[forecast_level].iloc[0].values
    logger.info("Processing : {}".format(the_intersection))
    error_df = pd.DataFrame()

    try:
        # drop a column if all the values are NA
        forecasts = df[forecast_cols].copy().dropna(axis=1, how="all")
        forecasts = forecasts.round(round_decimals)

        # forecasts might be missing for a particular date across all models
        forecasts.dropna(axis=0, how="all", inplace=True)

        # collect actuals
        actuals = df[history_measure][forecasts.index].to_numpy()

        # need atleast two forecasted points to check if both of them are same
        if override_flat_line_forecasts and len(forecasts) > 1:
            # discard algorithms if all predicted are same
            models_with_all_values_same = list(
                forecasts.columns[check_for_columns_with_same_value(forecasts)]
            )
            logger.info(
                "models_with_all_values_same : {}".format(
                    models_with_all_values_same
                )
            )

            # if all models generate flat line then output first best fit instead of no forecast
            if len(models_with_all_values_same) != len(forecasts.columns):
                forecasts.drop(
                    models_with_all_values_same, axis=1, inplace=True
                )
                # some cases yields non zero dataframe size since we are dropping cols in above step
                # drop the columns again if all rows are null
                forecasts.dropna(axis=0, how="all", inplace=True)

            if len(forecasts) == 0:
                logger.warning(
                    "No rows left in forecast dataframe after dropping models with flat forecast for intersection : {} ...".format(
                        the_intersection
                    )
                )

        # check if any columns/rows exist after dropping forecasts with same value ..
        if len(forecasts) > 0 and len(actuals) > 0:
            # Calculate errors for all models
            errors = forecasts.apply(
                lambda x: calculate_error(
                    actuals=actuals.astype("float64"),
                    forecasts=x.to_numpy().astype("float64"),
                    metric=error_metric,
                )
            )

            # Convert series to dataframe
            error_df = pd.DataFrame(errors).T

            # add dimension columns
            for the_col in forecast_level:
                error_df.insert(0, the_col, df[the_col].unique()[0])

    except Exception as e:
        logger.error("Exception for {} ...".format(the_intersection))
        logger.exception(e)
    return error_df


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
    HistoryMeasure,
    TimeDimension,
    ForecastParameters,
    LastTimePeriod,
    ActualsAndForecastData,
    OverrideFlatLineForecasts,
    ForecastModelDescriptions,
    multiprocessing_num_cores,
    df_keys,
):
    plugin_name = "DP015IdentifyBestFitModel"
    logger.info("Executing {} for slice {} ...".format(plugin_name, df_keys))

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
        best_fit_algo_col = "System Bestfit Algorithm"
        round_decimals = 2
        stat_fcst_str = "Stat Fcst "
        best_fit_algo_col_modified = "Best Fit Modified"
        best_fit_model_desc_col = "Bestfit Model"
        time_week_col = "Time.[Week]"

        stat_fcst_cols = [
            x for x in ActualsAndForecastData.columns if "Stat Fcst" in x
        ]

        req_cols = [
            version_col,
            Grain1,
            Grain2,
            Grain3,
            TimeLevel,
            HistoryMeasure,
        ] + stat_fcst_cols
        ActualsAndForecastData = ActualsAndForecastData[req_cols]

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

        logger.info(
            "-----------------------------------------------------------"
        )

        logger.info("--------- Grain1 : {}".format(Grain1))
        logger.info("--------- Grain2 : {}".format(Grain2))
        logger.info("--------- Grain3 : {}".format(Grain3))
        logger.info("--------- TimeLevel : {}".format(TimeLevel))
        logger.info("--------- HistoryMeasure : {}".format(HistoryMeasure))
        logger.info(
            "--------- OverrideFlatLineForecasts : {}".format(
                OverrideFlatLineForecasts
            )
        )

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
        history_periods = int(ForecastParameters[history_period_col].iloc[0])

        error_metric = ForecastParameters[error_metric_col].iloc[
            0
        ]  # MAPE, RMSE is supported
        override_flat_line_forecasts = eval(OverrideFlatLineForecasts)

        logger.info("error_metric : {}".format(error_metric))
        logger.info(
            "override_straight_line_forecasts : {}".format(
                override_flat_line_forecasts
            )
        )

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

        # we need to pull forecast + actuals data for all available history periods, even if method is out sample
        # there might be cases in which method is OUT Sample, but due to insufficient data, we can only do in sample validation
        logger.info("Getting last {} period dates ...".format(history_periods))
        history_period_dates = get_n_time_periods(
            latest_time_name,
            -history_periods,
            relevant_time_mapping,
            time_attribute_dict,
        )

        logger.info("history_period_dates : {}".format(history_period_dates))

        # history period data - actuals and forecasts- Could be in sample/out sample, that's why pulling all history
        all_validation_pred = ActualsAndForecastData[
            ActualsAndForecastData[TimeLevel].isin(history_period_dates)
        ]

        # collect forecast column names
        forecast_cols = [
            x for x in ActualsAndForecastData.columns if stat_fcst_str in x
        ]

        # Add ' Model' to match the convention on tenant
        forecast_desc_cols = [x[10:] + " Model" for x in forecast_cols]

        # Define output columns and dataframes
        ForecastModel_req_cols = (
            [version_col] + forecast_level + forecast_desc_cols
        )
        BestFitAlgo_req_cols = (
            [version_col] + forecast_level + [best_fit_algo_col]
        )
        ForecastModel, BestFitAlgo = (
            pd.DataFrame(columns=ForecastModel_req_cols),
            pd.DataFrame(columns=BestFitAlgo_req_cols),
        )

        # Check if prediction dataframe is empty, drop a column if all the rows are NAs
        if (
            len(all_validation_pred[forecast_cols].dropna(axis=1, how="all"))
            == 0
        ):
            logger.warning(
                "No forecast data found for validation periods for slice : {}...".format(
                    df_keys
                )
            )
            logger.warning("Returning empty dataframe for this slice ...")
            return ForecastModel, BestFitAlgo

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
        AllForecast = ActualsAndForecastData[
            ActualsAndForecastData[TimeLevel].isin(forecast_period_dates)
        ]

        if len(AllForecast) == 0:
            logger.warning(
                "No data found after filtering for forecast_period_dates for slice : {}".format(
                    df_keys
                )
            )
            logger.warning("Returning empty dataframe ...")
            return ForecastModel, BestFitAlgo

        logger.info("Calculating validation errors ...")

        # for every intersection calculate the validation errors
        all_error_df_list = Parallel(
            n_jobs=multiprocessing_num_cores, verbose=1
        )(
            delayed(calc_validation_error)(
                group,
                forecast_level,
                error_metric,
                override_flat_line_forecasts,
                forecast_cols,
                HistoryMeasure,
                round_decimals,
            )
            for name, group in all_validation_pred.groupby(forecast_level)
        )

        # collect all results to a dataframe
        all_error_df = concat_to_dataframe(all_error_df_list)

        # validate if output dataframe contains result for all groups present in input
        validate_output(
            input_df=all_validation_pred,
            output_df=all_error_df,
            forecast_level=forecast_level,
        )

        if len(all_error_df) == 0:
            logger.warning(
                "Validation Error dataframe is empty, returning empty dataframes for this slice : {}...".format(
                    df_keys
                )
            )
            return ForecastModel, BestFitAlgo

        # set dimension columns as index for idxmin to work properly
        all_error_df.set_index(forecast_level, inplace=True)

        logger.info("Calculating min of all errors and best fit model ...")

        # calculate min of errors and find best fit
        all_error_df[best_fit_algo_col] = all_error_df.idxmin(axis=1)

        # reset index to retain dimension columns
        all_error_df.reset_index(inplace=True)

        # string "Stat Fcst" from column to obtain only the algorithm name
        all_error_df[best_fit_algo_col] = all_error_df[best_fit_algo_col].str[
            10:
        ]

        logger.info("Best fit calculation complete ...")

        logger.info(
            "Adding error values to forecast model description dataframe ..."
        )
        model_desc_with_errors = ForecastModelDescriptions.merge(
            all_error_df, on=forecast_level
        )

        # Filter rows where best fit columns is not NA
        model_desc_with_errors = model_desc_with_errors[
            model_desc_with_errors[best_fit_algo_col].notna()
        ]

        available_model_desc_cols = []
        # Add error values to description for every forecast model
        for the_forecast_col in forecast_cols:
            the_model_desc_col = the_forecast_col[10:] + " Model"
            logger.info("------------- {}".format(the_model_desc_col))

            # check if forecast column is present in dataframe - AR NNet etc might not be there, cannot measure error in this case
            if the_forecast_col in model_desc_with_errors.columns:
                # Filter cases where both columns are not NA
                filter_clause = (
                    model_desc_with_errors[the_model_desc_col].notna()
                ) & (model_desc_with_errors[the_forecast_col].notna())

                # Add MAPE value to the same string for the filtered rows, assign NA to others
                model_desc_with_errors[the_model_desc_col] = np.where(
                    filter_clause,
                    (
                        model_desc_with_errors[the_model_desc_col].astype(
                            "str"
                        )
                        + " | Error : {} ".format(error_metric)
                        + model_desc_with_errors[the_forecast_col]
                        .round(2)
                        .astype("str")
                    ),
                    np.nan,
                )
                available_model_desc_cols.append(the_model_desc_col)

        # append " Model" to the best fit algorithm col to lookup best fit model description
        model_desc_with_errors[best_fit_algo_col_modified] = (
            model_desc_with_errors[best_fit_algo_col].astype("str") + " Model"
        )

        # Lookup values against columns to populate best fit algorithm description
        model_desc_with_errors[
            best_fit_model_desc_col
        ] = model_desc_with_errors.lookup(
            model_desc_with_errors.index,
            model_desc_with_errors[best_fit_algo_col_modified],
        )

        input_version = LastTimePeriod[version_col].unique()[0]
        if len(all_error_df) > 0:
            all_error_df.insert(0, version_col, input_version)
            BestFitAlgo = all_error_df[BestFitAlgo_req_cols]

        ForecastModel_req_cols = (
            [version_col]
            + forecast_level
            + available_model_desc_cols
            + [best_fit_model_desc_col]
        )
        ForecastModel = model_desc_with_errors[ForecastModel_req_cols]

        logger.info("------------ ForecastModel : head -----------")
        logger.info(ForecastModel.head())

        logger.info("------------ BestFitAlgo : head -----------")
        logger.info(BestFitAlgo.head())

        logger.info("Successfully executed {} ...".format(plugin_name))
    except Exception as e:
        logger.error(
            "Exception for slice : {}, returning empty dataframe as output ...".format(
                df_keys
            )
        )
        logger.exception(e)
        ForecastModel = get_o9_empty_df()
        BestFitAlgo = get_o9_empty_df()

    return ForecastModel, BestFitAlgo


