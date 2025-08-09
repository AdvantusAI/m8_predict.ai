"""
Plugin : DP006StatSegmentL1
Version : 2022.08.16
Maintained by : pmm_algocoe@o9solutions.com

"""
import logging
from functools import reduce

import numpy as np
import pandas as pd
from o9Reference.common_utils.fill_missing_dates import fill_missing_dates

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None

from o9Reference.common_utils.common_utils import (
    split_string,
    get_n_time_periods,
    get_relevant_time_name_and_key,
)
from o9Reference.stat_utils.assign_segments import assign_segments
from o9Reference.stat_utils.time_series import (
    calculate_trend,
    get_variability,
    get_mean_stddev,
)
from o9Reference.common_utils.common_utils import filter_relevant_time_mapping
from o9Reference.common_utils.function_timer import timed
from o9Reference.common_utils.dataframe_utils import (
    concat_to_dataframe,
    get_o9_empty_df,
)

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
    COVSegLvl,
    TimeLevel,
    COVThresholdMeasure,
    Actual,
    Grain2,
    Grain1,
    VolumeThresholdMeasure,
    Grain3,
    DimClass,
    SegmentThresholds,
    MiscThresholds,
    TimeDimension,
    VolSegLvl,
    LastTimePeriod,
    ReadFromHive,
    df_keys,
):
    plugin_name = "DP006StatSegmentL1_py"
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
        version_col = "Version.[Version Name]"
        class_delimiter = ","
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
        vol_cov_history_period_col = "Volume-COV History Period"
        history_time_bucket_col = "History Time Buckets"
        history_measure_col = "History Measure"

        # PLC Segmentation
        npi_date_col = "npi_date"
        eol_date_col = "eol_date"

        null_flag_col = "Null Flag"
        zero_sum_col = "ZeroFlagSum"
        zero_ratio_col = "ZeroRatio"

        trend_factor_col = "TrendFactor"
        # seasonal_diff_col = "ndiffs"

        npi_threshold_col = "New Launch Period"
        eol_threshold_col = "Disco Period"
        intermittency_threshold_col = "Intermittency Threshold"
        trend_treshold_col = "Trend Threshold"
        # seasonality_threshold_col = "Seasonality Threshold"
        plc_segmentation_threshold_col = "History Period"
        time_week_col = "Time.[Week]"
        is_day_col = "Is Day"

        # output measures - StatSegmentation
        vol_segment_col = "Volume Segment L1"
        cov_segment_col = "COV Segment L1"
        prod_segment_l1_col = "Product Segment L1"
        los_col = "Length of Series L1"
        num_zeros_col = "Number of Zeros L1"
        intermittency_col = "Intermittent L1"
        plc_col = "PLC Status L1"
        trend_col = "Trend L1"
        # seasonality_col = "Seasonality L1"
        std_dev_col = "Std Dev L1"
        avg_col = "Avg Volume L1"
        cov_col = "COV L1"
        volume_col = "Volume L1"
        vol_share_col = "Volume % L1"
        cumulative_vol_col = "Cumulative Volume % L1"

        # output measures - ProductS
        prod_segment_col = "Product Customer L1 Segment"

        # Filter the required columns from dataframes
        req_cols = [
            version_col,
            intermittency_threshold_col,
            npi_threshold_col,
            eol_threshold_col,
            history_time_bucket_col,
            history_measure_col,
            vol_cov_history_period_col,
            trend_treshold_col,
            # seasonality_threshold_col,
            plc_segmentation_threshold_col,
        ]
        MiscThresholds = MiscThresholds[req_cols]

        history_measure = str(MiscThresholds[history_measure_col].iloc[0])
        if ReadFromHive:
            history_measure = "DP006" + history_measure

        logger.info("history_measure : {}".format(history_measure))

        req_cols = [version_col, VolumeThresholdMeasure, COVThresholdMeasure]
        SegmentThresholds = SegmentThresholds[req_cols]

        req_cols = [version_col, time_week_col, stat_r_attribute_week_col]
        TimeDimension = TimeDimension[req_cols]

        req_cols = [version_col, TimeLevel, is_day_col]
        LastTimePeriod = LastTimePeriod[req_cols]

        logger.info("Extracting segmentation level ...")

        # combine grains to get segmentation level
        all_grains = [Grain1, Grain2, Grain3]
        segmentation_level = [
            str(x) for x in all_grains if x != "NA" and x != ""
        ]

        logger.info("segmentation_level : {}".format(segmentation_level))

        req_cols = (
            [version_col, TimeLevel] + segmentation_level + [history_measure]
        )
        Actual = Actual[req_cols]

        stat_segmentation_cols = (
            [version_col]
            + segmentation_level
            + [
                vol_segment_col,
                cov_segment_col,
                prod_segment_l1_col,
                los_col,
                num_zeros_col,
                intermittency_col,
                plc_col,
                trend_col,
                # seasonality_col,
                std_dev_col,
                avg_col,
                cov_col,
                volume_col,
                vol_share_col,
                cumulative_vol_col,
            ]
        )
        prod_segmentation_cols_output = (
            [version_col] + segmentation_level + [DimClass, prod_segment_col]
        )

        # Actuals might not be present for a particular slice, check and return empty dataframe
        StatSegmentation = get_o9_empty_df()
        ProductSegmentation = get_o9_empty_df()

        if Actual is None or len(Actual) == 0:
            logger.warning(
                "Actuals is None/Empty for slice : {}...".format(df_keys)
            )
            logger.warning("Returning empty dataframes as result ...")
            return StatSegmentation, ProductSegmentation

        input_version = TimeDimension[version_col].iloc[0]

        # split string into lists based on delimiter
        cov_levels = COVSegLvl.split(class_delimiter)
        cov_thresholds = [
            round(float(x), 2)
            for x in list(SegmentThresholds[COVThresholdMeasure])
        ]
        # remove duplicates if any
        cov_thresholds = list(set(cov_thresholds))

        logger.info("cov_levels : {}".format(cov_levels))
        logger.info("cov_thresholds : {}".format(cov_thresholds))

        # split string into lists based on delimiter
        vol_levels = VolSegLvl.split(class_delimiter)
        vol_thresholds = [
            round(float(x), 2)
            for x in list(SegmentThresholds[VolumeThresholdMeasure])
        ]
        # remove duplicates if any
        vol_thresholds = list(set(vol_thresholds))

        logger.info("vol_levels : {}".format(vol_levels))
        logger.info("vol_thresholds : {}".format(vol_thresholds))

        logger.info("Creating time mapping ...")
        logger.info("column_to_split : {}".format(stat_r_attribute_week_col))

        logger.info("time_cols : {}".format(time_cols))

        # split on string and create dataframe
        time_mapping = split_string(
            values=list(TimeDimension[stat_r_attribute_week_col]),
            delimiter=time_delimiter,
            col_names=time_cols,
        )

        logger.info("time mapping head :")
        logger.info(time_mapping.head())

        frequency = str(MiscThresholds[history_time_bucket_col].iloc[0])
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

        # Dictionary for easier lookups
        relevant_time_mapping_dict = dict(
            zip(
                list(relevant_time_mapping[relevant_time_name]),
                list(relevant_time_mapping[relevant_time_key]),
            )
        )

        segmentation_period = int(
            MiscThresholds[vol_cov_history_period_col].iloc[0]
        )
        logger.info("segmentation_period : {}".format(segmentation_period))

        logger.info("---------------------------------------")

        logger.info(
            "filtering rows where {} is not null ...".format(history_measure)
        )
        Actual = Actual[Actual[history_measure].notna()]

        if len(Actual) == 0:
            logger.warning(
                "Actuals df is empty after filtering non null values for slice : {}...".format(
                    df_keys
                )
            )
            logger.warning("Returning empty dataframes as result ...")
            return StatSegmentation, ProductSegmentation

        # check if history measure sum is positive before proceeding further
        if Actual[history_measure].sum() <= 0:
            logger.warning(
                "Sum of actuals is non positive for slice : {}...".format(
                    df_keys
                )
            )
            logger.warning("Returning empty dataframes as result ...")
            return StatSegmentation, ProductSegmentation

        # cap negative values to zero
        Actual[history_measure] = np.where(
            Actual[history_measure] < 0, 0, Actual[history_measure]
        )

        logger.info("LastTimePeriod head : ")
        logger.info(LastTimePeriod.head())

        time_attribute_dict = {relevant_time_name: relevant_time_key}

        # Gather the latest time name
        latest_time_name = LastTimePeriod[TimeLevel].iloc[0]

        # Gather NPI, EOL attributes
        npi_horizon = int(MiscThresholds[npi_threshold_col].iloc[0])
        eol_horizon = int(MiscThresholds[eol_threshold_col].iloc[0])
        intermittency_threshold = float(
            MiscThresholds[intermittency_threshold_col].iloc[0]
        )

        # Join actuals with time mapping
        Actual_with_time_key = Actual.copy().merge(
            relevant_time_mapping,
            left_on=TimeLevel,
            right_on=relevant_time_name,
            how="inner",
        )
        logger.info("Calculating start,end dates for PLC Segmentation ...")
        # aggregation to get start and end date
        plc_df = (
            Actual_with_time_key.assign(
                start_date=Actual_with_time_key[relevant_time_key],
                end_date=Actual_with_time_key[relevant_time_key],
            )
            .groupby(segmentation_level, observed=True)
            .agg(dict(start_date=min, end_date=max))
            .reset_index()
        )
        # get NPI, EOL Dates
        last_n_period_npi = get_n_time_periods(
            latest_time_name,
            -npi_horizon,
            relevant_time_mapping,
            time_attribute_dict,
        )[0]
        npi_cutoff_date = relevant_time_mapping_dict[last_n_period_npi]
        logger.info("npi_cutoff_date : {}".format(npi_cutoff_date))
        plc_df[npi_date_col] = npi_cutoff_date

        last_n_period_eol = get_n_time_periods(
            latest_time_name,
            -eol_horizon,
            relevant_time_mapping,
            time_attribute_dict,
        )[0]
        eol_cutoff_date = relevant_time_mapping_dict[last_n_period_eol]
        logger.info("eol_cutoff_date : {}".format(eol_cutoff_date))
        plc_df[eol_date_col] = eol_cutoff_date

        logger.info("Assigning PLC Segments ...")
        # assign categories NEW LAUNCH, MATURE and DISC
        conditions = [
            plc_df["start_date"] > plc_df[npi_date_col],
            plc_df["end_date"] < plc_df[eol_date_col],
        ]
        choices = ["NEW LAUNCH", "DISC"]

        plc_df[plc_col] = np.select(conditions, choices, default=None)
        plc_df[plc_col].fillna("MATURE", inplace=True)

        # get last n periods based on vol-cov segmentation period
        logger.info(
            "Getting last {} period dates for vol-cov segmentation ...".format(
                segmentation_period
            )
        )
        # note the negative sign to segmentation period
        last_n_periods_vol_cov = get_n_time_periods(
            latest_time_name,
            -segmentation_period,
            relevant_time_mapping,
            time_attribute_dict,
        )

        # convert to df for join
        last_n_period_vol_cov_df = pd.DataFrame(
            {relevant_time_name: last_n_periods_vol_cov}
        )

        if len(last_n_period_vol_cov_df) == 0:
            logger.warning(
                "No dates found after filtering last {} periods from time mapping for slice {}...".format(
                    segmentation_period, df_keys
                )
            )
            logger.warning("Returning empty dataframes for this slice ...")
            return StatSegmentation, ProductSegmentation

        logger.info(
            "Joining actuals with time mapping with last n period dates ... "
        )
        # filter relevant history based on dates provided above
        vol_cov_segmentation_input = Actual.merge(
            last_n_period_vol_cov_df,
            left_on=TimeLevel,
            right_on=relevant_time_name,
            how="right",
        )

        # create a copy and use for subsequent joins with other dataframes
        result = plc_df.copy()
        result[prod_segment_l1_col] = result[plc_col]

        if len(vol_cov_segmentation_input) > 0:

            logger.info("Calculating volume segments ...")
            # groupby and take aggregate volume
            volume_df = (
                vol_cov_segmentation_input.groupby(
                    segmentation_level, observed=True
                )
                .sum()[[history_measure]]
                .rename(columns={history_measure: volume_col})
                .reset_index()
            )
            volume_df.sort_values(volume_col, ascending=False, inplace=True)
            volume_df.reset_index(drop=True, inplace=True)

            # calculate volume share percentages
            total_volume = volume_df[volume_col].sum()

            # check denominator is not zero, to avoid division by zero error in next step
            if total_volume <= 0:
                logger.warning(
                    "Total Volume should be positive for volume share calculation, slice {} ...".format(
                        df_keys
                    )
                )
                logger.warning("Returning empty df for this slice ...")
                return StatSegmentation, ProductSegmentation

            volume_df[vol_share_col] = volume_df[volume_col] / total_volume

            # Find cumulative volume of items
            volume_df[cumulative_vol_col] = volume_df[vol_share_col].cumsum()

            # assign volume segment
            volume_df[vol_segment_col] = assign_segments(
                volume_df[cumulative_vol_col].to_numpy(),
                vol_thresholds,
                vol_levels,
            )

            logger.info("Calculating variability segments ...")
            # groupby and calculate variability
            variability_df = (
                vol_cov_segmentation_input.groupby(
                    segmentation_level, observed=True
                ).apply(
                    lambda x: get_variability(x[history_measure].to_numpy())
                )
            ).reset_index(name=cov_col)

            # calculate mean and std deviation separately
            mean_std_df = (
                vol_cov_segmentation_input.groupby(
                    segmentation_level, observed=True
                ).apply(
                    lambda x: get_mean_stddev(
                        list(x[history_measure]), std_dev_col, avg_col
                    )
                )
            ).reset_index()

            variability_df = variability_df.merge(
                mean_std_df, on=segmentation_level, how="inner"
            )
            # assign variability segments
            variability_df[cov_segment_col] = assign_segments(
                variability_df[cov_col].to_numpy(), cov_thresholds, cov_levels
            )

            logger.info("Merging volume, variability, plc dataframes ...")

            result = reduce(
                lambda x, y: pd.merge(
                    x, y, on=segmentation_level, how="outer"
                ),
                [volume_df, variability_df, result],
            )

            logger.info("Merge complete, shape  : {}".format(result.shape))

            # Fill NAs with defaults - low volume and high variability segments
            result[vol_segment_col].fillna(vol_levels[-1], inplace=True)
            result[cov_segment_col].fillna(cov_levels[-1], inplace=True)

            logger.info("Assigning final PLC Segment ...")
            result[prod_segment_l1_col] = np.where(
                result[prod_segment_l1_col] == "MATURE",
                result[vol_segment_col] + result[cov_segment_col],
                result[prod_segment_l1_col],
            )

        trend_treshold = float(MiscThresholds[trend_treshold_col].iloc[0])
        # seasonality_treshold = float(
        #     MiscThresholds[seasonality_threshold_col].iloc[0]
        # )
        plc_segmentation_periods = int(
            MiscThresholds[plc_segmentation_threshold_col].iloc[0]
        )

        logger.info(
            "Getting last {} period dates for PLC segmentation ...".format(
                plc_segmentation_periods
            )
        )

        # note the negative sign to segmentation period
        last_n_periods_plc = get_n_time_periods(
            latest_time_name,
            -plc_segmentation_periods,
            relevant_time_mapping,
            time_attribute_dict,
        )

        last_n_periods_plc_df = pd.DataFrame(
            {relevant_time_name: last_n_periods_plc}
        )

        if len(last_n_periods_plc_df) == 0:
            logger.warning(
                "No dates found after filtering last {} periods from time mapping for PLC segmentation, slice : {} ...".format(
                    plc_segmentation_periods, df_keys
                )
            )
            logger.warning("Returning empty df for this slice ...")
            return StatSegmentation, ProductSegmentation

        # join to get relevant data
        plc_segmentation_input = Actual.merge(
            last_n_periods_plc_df,
            left_on=TimeLevel,
            right_on=relevant_time_name,
            how="inner",
        )

        list_of_dfs = list()
        if len(plc_segmentation_input) > 0:

            plc_segmentation_input = fill_missing_dates(
                actual=Actual,
                forecast_level=segmentation_level,
                history_measure=history_measure,
                time_level=TimeLevel,
                relevant_time_key=relevant_time_key,
                relevant_time_name=relevant_time_name,
                relevant_time_periods=last_n_periods_plc,
                time_mapping=relevant_time_mapping,
                fill_nulls_with_zero=False,
            )

            # flag nulls, fill with zeros
            plc_segmentation_input[null_flag_col] = np.where(
                plc_segmentation_input[history_measure].isna(), 1, 0
            )
            plc_segmentation_input[history_measure].fillna(0, inplace=True)

            # Create df with nulls/missing dates replaced with zero
            logger.info("Looping through all intersections ...")
            logger.info("-------------------------------------------")
            for (
                the_intersection,
                the_intersection_df,
            ) in plc_segmentation_input.groupby(
                segmentation_level, observed=True
            ):
                logger.info(
                    "Processing intersection : {}".format(the_intersection)
                )
                try:
                    # create attributes for the intersection
                    req_attributes_df = pd.DataFrame(index=[0])

                    # assign segmentation level columns to new df
                    for the_col in segmentation_level:
                        req_attributes_df[the_col] = the_intersection_df[
                            the_col
                        ].iloc[0]

                    # calculate attributes
                    req_attributes_df[zero_sum_col] = the_intersection_df[
                        null_flag_col
                    ].sum()
                    req_attributes_df[zero_ratio_col] = req_attributes_df[
                        zero_sum_col
                    ] / len(the_intersection_df)

                    # calculate trend
                    req_attributes_df[trend_factor_col] = calculate_trend(
                        the_intersection_df[history_measure].to_numpy()
                    )

                    # calculate seasonality
                    # req_attributes_df[seasonal_diff_col] = ndiffs(
                    #     the_intersection_df[history_measure],
                    #     alpha=0.05,
                    #     test="kpss",
                    # )

                    # calculate LOS
                    req_attributes_df[los_col] = len(the_intersection_df)

                    # append to master dataframe
                    list_of_dfs.append(req_attributes_df)
                except Exception as e:
                    logger.error(
                        "Exception for intersection : {}".format(
                            the_intersection
                        )
                    )
                    logger.exception(e)

            logger.info("Processed all intersections ...")
            logger.info("------------------------------------")

            logger.info("Combining all intersection attributes to one df ...")
            ts_attributes_df = concat_to_dataframe(list_of_dfs)

            result = reduce(
                lambda x, y: pd.merge(
                    x, y, on=segmentation_level, how="outer"
                ),
                [ts_attributes_df, result],
            )

            logger.info(
                "df shape after combining plc with ts attributes : {}".format(
                    result.shape
                )
            )

            # Assign trend category
            conditions = [
                result[trend_factor_col] > trend_treshold,
                (result[trend_factor_col] <= trend_treshold)
                & (result[trend_factor_col] >= -trend_treshold),
                result[trend_factor_col] < trend_treshold,
            ]
            choices = ["UPWARD", "NO TREND", "DOWNWARD"]

            logger.info("Assigning trend categories ...")
            result[trend_col] = np.select(conditions, choices, default=None)

            logger.info("Assigning seasonality categories ...")
            # Assign seasonality category
            # result[seasonality_col] = np.where(
            #     result[seasonal_diff_col] < seasonality_treshold,
            #     "Does Not Exist",
            #     "Exists",
            # )

            logger.info("Assigning intermittency categories ...")
            # Assign intermittency category
            result[intermittency_col] = np.where(
                result[zero_ratio_col] >= intermittency_threshold, "YES", "NO"
            )

            # calculate number of zeros
            result[num_zeros_col] = round(
                result[los_col] * result[zero_ratio_col], 0
            )

        # collect version from input data
        result[version_col] = input_version

        # Assign 1.0 value to product segment column
        result[prod_segment_col] = 1.0

        logger.info("Filtering relevant columns to output ...")

        if (
            len(vol_cov_segmentation_input) > 0
            and len(plc_segmentation_input) > 0
        ):
            # Filter relevant columns
            StatSegmentation = result[stat_segmentation_cols]

        # Filter relevant columns
        prod_segmentation_cols = (
            [version_col]
            + segmentation_level
            + [prod_segment_l1_col, prod_segment_col]
        )
        ProductSegmentation = result[prod_segmentation_cols]
        ProductSegmentation.rename(
            columns={prod_segment_l1_col: DimClass}, inplace=True
        )

        logger.info("Successfully executed {} ...".format(plugin_name))
        logger.info("---------------------------------------------")
    except Exception as e:
        logger.error(
            "Exception for slice : {}, returning empty dataframe as output ...".format(
                df_keys
            )
        )
        logger.exception(e)
        StatSegmentation = get_o9_empty_df()
        ProductSegmentation = get_o9_empty_df()

    return StatSegmentation, ProductSegmentation


