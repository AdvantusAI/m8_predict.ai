"""
Plugin : DP037SeasonalityDetection
Version : 2022.08.16
Maintained by : pmm_algocoe@o9solutions.com

"""
import logging
import threading

from joblib import delayed
from joblib import Parallel
from o9Reference.common_utils.data_utils import validate_output
from o9Reference.common_utils.function_logger import log_inputs_and_outputs
from o9Reference.common_utils.function_timer import timed
from o9Reference.common_utils.o9_memory_utils import _get_memory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("o9_logger")
import math

import numpy as np
import pandas as pd

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

pd.options.mode.chained_assignment = None
from scipy.fftpack import fft, fftfreq
from tsfeatures.tsfeatures import stl_features, entropy
from o9Reference.common_utils.common_utils import (
    split_string,
    filter_relevant_time_mapping,
    get_relevant_time_name_and_key,
    get_n_time_periods,
    get_seasonal_periods,
)
from o9Reference.common_utils.dataframe_utils import (
    concat_to_dataframe,
    get_o9_empty_df,
)
from o9Reference.common_utils.fill_missing_dates import fill_missing_dates


class FFTDetector:
    """Fast Fourier Transform Seasoanlity detector

    Use Fast Fourier Transform to detect seasonality, and find out the
    potential cycle's length.

    Attributes:
        data: The input time series data from TimeSeriesData.
    """

    def __init__(self, data):
        self.data = data

    def detector(
        self, sample_spacing: float = 1.0, mad_threshold: float = 6.0
    ):
        """Detect seasonality with FFT

        Args:
            sample_spacing: Optional; float; scaling FFT for a different time unit.
                I.e. for hourly time series, sample_spacing=24.0, FFT x axis will be
                1/day.
            mad_threshold: Optional; float; constant for the outlier algorithm for peak
                detector. The larger the value the less sensitive the outlier algorithm
                is.

        Returns:
           presence of seasonality with periods
        """

        # get top 3 seasons
        no_of_seasons = 3
        series = np.asarray(self.data)
        # Compute FFT
        series_fft = fft(series)

        # Compute the power
        power = np.abs(series_fft)

        # Get the corresponding frequencies
        sample_freq = fftfreq(series_fft.size)

        # Find the peak frequency: we only need the positive frequencies
        pos_mask = np.where(sample_freq > 0)
        freqs = sample_freq[pos_mask]
        powers = power[pos_mask]

        # find top frequencies and corresponding time periods for seasonal pattern
        top_powers = np.argpartition(powers, -no_of_seasons)[-no_of_seasons:]

        time_periods_from_fft = 1 / freqs[top_powers]
        time_periods = time_periods_from_fft.astype(int)
        seasonality_presence = len(time_periods) > 0
        selected_seasonalities = []
        if seasonality_presence:
            selected_seasonalities = list(time_periods)

        return {
            "seasonality_presence": seasonality_presence,
            "seasonalities": selected_seasonalities,
        }


class SeasonalDetection:
    """
    This class provides utilities to decompose the time series  data
    Pass specific arguments corresponding to input data parameters and frequency of output data
    Attributes:
        TSData : the input time series data

    """

    def __init__(self, ActualsDataframe, TimeColumn, TargetCol, Frequency):

        self.Actuals = ActualsDataframe
        self.TimeColumn = TimeColumn
        self.TargetColumn = TargetCol
        self.Frequency = Frequency

    def TimeSeriesDetect(self) -> (pd.DataFrame, str, list):

        # Final Periods can have a number which is more than max_p, but for certain cases the time period explores and returns an index which is not present in the input dataframe.
        # Hence max_p has been set to the seasonal periods value (4, 12, and 52 for quarterly, monthly and weekly respectively)
        if self.Frequency == "Quarterly":
            self.freq = "QS"
            self.max_p = 4
        elif self.Frequency == "Monthly":
            self.freq = "MS"
            self.max_p = 12
        elif self.Frequency == "Weekly":
            self.freq = "W"
            self.max_p = 52
        else:
            raise ValueError(
                "Unknown frequency {}, allowed values are Weekly/Monthly/Quarterly ..."
            )

        # filter relevant columns
        ts_data = self.Actuals[[self.TimeColumn, self.TargetColumn]]
        try:
            # create regularly spaced time series indexes with W/M/Q pandas frequencies
            all_dates = pd.date_range(
                start=min(ts_data[self.TimeColumn]),
                freq=self.freq,
                periods=len(ts_data),
            )

            # drop existing time key column since this is of planning frequency and cannot be fed to algorithms
            ts_data.drop(self.TimeColumn, axis=1, inplace=True)

            # assign new regularly spaced time stamps
            ts_data[self.TimeColumn] = all_dates.values

            # drop and reset index
            ts_data.reset_index(drop=True, inplace=True)

            # set index from 1  - weeks/months/quarters run from 1 to n, not zero
            ts_data.index += 1

            logger.info("Seasonality Detection has started!")

            # detect seasonalities
            detect_model = FFTDetector(ts_data[self.TargetColumn])
            detect_output = detect_model.detector()

            logger.info("seasonality presence calculation...")
            seasonality_presence = detect_output["seasonality_presence"]

            logger.info(
                "seasonality_presence : {}".format(seasonality_presence)
            )

            # assign string based on boolean value true/false
            seasonality_presence = "Yes" if seasonality_presence else "No"

            logger.info("seasonalities period calculation...")
            seasonal_periods = detect_output["seasonalities"]

            logger.info("seasonal_periods : {}".format(seasonal_periods))

            final_period = []
            for i in seasonal_periods:
                period1 = [math.floor(i) if i <= self.max_p else None]
                period2 = [math.ceil(i) if i <= self.max_p else None]
                period = period1 + period2
                final_period = period + final_period
                final_period = list(set(final_period))
                final_period = list(filter(None, final_period))
                if self.Frequency == "Weekly":
                    final_period = list(
                        ts_data[self.TimeColumn][final_period].dt.week
                    )
                    final_period = list(set(final_period))
                elif self.Frequency == "Monthly":
                    final_period = list(
                        ts_data[self.TimeColumn][final_period].dt.month
                    )
                    final_period = list(set(final_period))
                else:
                    final_period = list(
                        ts_data[self.TimeColumn][final_period].dt.quarter
                    )
                    final_period = list(set(final_period))

            logger.info("final_period : {}".format(final_period))

        except Exception as e:
            logger.exception(e)
            seasonality_presence = "No"
            final_period = []

        return ts_data, seasonality_presence, final_period


class Seasonality_Features:
    """
    This class provides utilities to produce features of the data
    Pass specific arguments corresponding to input data parameters and frequency of output data
    Attributes:
        TSData : the input time series data

    """

    def __init__(self, ActualsDataframe, TimeColumn, TargetCol, Frequency):

        self.Actuals = ActualsDataframe
        self.TimeColumn = TimeColumn
        self.TargetColumn = TargetCol
        if Frequency == "Quarterly":
            self.freq = 4
        elif Frequency == "Monthly":
            self.freq = 12
        elif Frequency == "Weekly":
            self.freq = 52
        else:
            self.freq = 1

    def TimeSeriesFeatures(self):
        logger.info("Feature Production has started!")
        tsdata = self.Actuals[self.TargetColumn]
        try:
            features = stl_features(x=tsdata.values, freq=self.freq)
            spec_entropy = entropy(x=tsdata.values, freq=self.freq)["entropy"]
            trend_strength = features["trend"]
            spikiness = features["spike"]
            seasonal_strength = features["seasonal_strength"]
        except Exception as e:
            logger.error("Error during feature production ...")
            logger.exception(e)
            trend_strength = 0
            spikiness = 0
            seasonal_strength = 0
            spec_entropy = 0
        return trend_strength, spikiness, seasonal_strength, spec_entropy


def calc_seasonality(
    ActualData: pd.DataFrame,
    relevant_time_key: str,
    history_measure: str,
    frequency: str,
    segmentation_level: list,
    seasonality_l1_col: str,
    seasonal_periods_l1_col: str,
    trend_strength_l1_col: str,
    seasonal_strength_l1_col: str,
    entropy_l1_col: str,
) -> pd.DataFrame:
    the_intersection = ActualData[segmentation_level].iloc[0].values
    logger.info("the_intersection : {}".format(the_intersection))
    try:
        # check if data has atleast one seasonal cycle, or sum of actuals is zero
        if (
            len(ActualData) < get_seasonal_periods(frequency)
            or ActualData[history_measure].sum() <= 0
        ):
            # create fallback dataframe for results
            result = ActualData[segmentation_level].drop_duplicates()
            result[seasonality_l1_col] = "No"
            result[seasonal_periods_l1_col] = "[]"
            result[trend_strength_l1_col] = 0
            result[seasonal_strength_l1_col] = 0
            result[entropy_l1_col] = 0
        else:
            SeasonalDetectionModel = SeasonalDetection(
                ActualData, relevant_time_key, history_measure, frequency
            )
            (
                NewTSData,
                SeasonalityPresence,
                SeasonalPeriods,
            ) = SeasonalDetectionModel.TimeSeriesDetect()
            FeatruesModel = Seasonality_Features(
                NewTSData, relevant_time_key, history_measure, frequency
            )
            (
                TrendStrength,
                Spikiness,
                SeasonalStrength,
                SpecEntropy,
            ) = FeatruesModel.TimeSeriesFeatures()

            # create dataframe for results
            result = ActualData[segmentation_level].drop_duplicates()
            result[seasonality_l1_col] = SeasonalityPresence

            # if list is not empty, form comma separated string, strip brackets
            if SeasonalPeriods:
                result[seasonal_periods_l1_col] = str(SeasonalPeriods).strip(
                    "[\]"
                )
            else:
                result[seasonal_periods_l1_col] = "[]"

            result[trend_strength_l1_col] = TrendStrength
            result[seasonal_strength_l1_col] = SeasonalStrength
            result[entropy_l1_col] = SpecEntropy

            # Assign categories, fallback values, fill NAs
            result[seasonality_l1_col] = result[seasonality_l1_col].fillna(
                "No"
            )
            result[seasonal_periods_l1_col] = result[
                seasonal_periods_l1_col
            ].fillna("[]")
            result.fillna(0, inplace=True)

    except Exception as e:
        logger.exception(
            "intersection : {}, exception : {}".format(the_intersection, e)
        )
        result = pd.DataFrame()

    return result


from o9Reference.common_utils.dataframe_utils import ensure_valid_output_schema


@ensure_valid_output_schema
@log_inputs_and_outputs
@timed
def main(
    TimeLevel,
    Grain1,
    Grain2,
    Grain3,
    VersionColumn,
    Data,
    TimeDimension,
    LastTimePeriod,
    Parameters,
    ReadFromHive,
    multiprocessing_num_cores,
    df_keys,
):
    plugin_name = "DP037SeasonalityDetection"
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
        seasonality_l1_col = "Seasonality L1"
        seasonal_periods_l1_col = "Seasonal Periods L1"
        trend_strength_l1_col = "Trend Strength L1"
        seasonal_strength_l1_col = "Seasonal Strength L1"
        entropy_l1_col = "Entropy L1"
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
        history_time_buckets_col = "History Time Buckets"
        history_measure_col = "History Measure"
        history_period_col = "History Period"

        history_measure = Parameters[history_measure_col].iloc[0]
        if ReadFromHive:
            history_measure = "DP006" + history_measure

        frequency = Parameters[history_time_buckets_col].iloc[0]
        history_periods = int(Parameters[history_period_col].iloc[0])

        logger.info("history_measure : {}".format(history_measure))
        logger.info("frequency : {}".format(frequency))
        logger.info("history_periods : {}".format(history_periods))

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

        # combine grains to get forecast level
        all_grains = [Grain1, Grain2, Grain3]
        segmentation_level = [
            str(x) for x in all_grains if x != "NA" and x != ""
        ]

        logger.info("segmentation_level : {}".format(segmentation_level))

        logger.info("LastTimePeriod head : ")
        logger.info(LastTimePeriod.head())

        time_attribute_dict = {relevant_time_name: relevant_time_key}
        input_version = LastTimePeriod[VersionColumn].unique()[0]

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
            return get_o9_empty_df()

        # fill missing dates
        relevant_history_nas_filled = fill_missing_dates(
            actual=Data,
            forecast_level=segmentation_level,
            time_mapping=relevant_time_mapping,
            history_measure=history_measure,
            relevant_time_name=relevant_time_name,
            relevant_time_key=relevant_time_key,
            relevant_time_periods=last_n_periods,
            time_level=TimeLevel,
            fill_nulls_with_zero=True,
        )

        MonthlyFinalDataFull_list = Parallel(
            n_jobs=multiprocessing_num_cores, verbose=1
        )(
            delayed(calc_seasonality)(
                ActualData=group,
                relevant_time_key=relevant_time_key,
                history_measure=history_measure,
                frequency=frequency,
                segmentation_level=segmentation_level,
                seasonality_l1_col=seasonality_l1_col,
                seasonal_periods_l1_col=seasonal_periods_l1_col,
                trend_strength_l1_col=trend_strength_l1_col,
                seasonal_strength_l1_col=seasonal_strength_l1_col,
                entropy_l1_col=entropy_l1_col,
            )
            for name, group in relevant_history_nas_filled.groupby(
                segmentation_level, observed=True
            )
        )

        logger.info("Collected results from parallel processing ...")

        # concat list of results to df
        MonthlyFinalDataFull = concat_to_dataframe(
            list_of_results=MonthlyFinalDataFull_list
        )
        logger.info("Validating output for all intersections ...")

        # validate if output dataframe contains result for all groups present in input
        validate_output(
            input_df=relevant_history_nas_filled,
            output_df=MonthlyFinalDataFull,
            forecast_level=segmentation_level,
        )

        logger.info("Assigning values to {} ...".format(seasonality_l1_col))

        # assign values based on condition
        MonthlyFinalDataFull[seasonality_l1_col] = np.where(
            MonthlyFinalDataFull[seasonality_l1_col] == "Yes",
            "Exists",
            "Does Not Exist",
        )

        logger.info(
            "Assigning values to {} ...".format(seasonal_periods_l1_col)
        )

        MonthlyFinalDataFull[seasonal_periods_l1_col] = np.where(
            MonthlyFinalDataFull[seasonal_periods_l1_col] == "[]",
            "Does Not Exist",
            MonthlyFinalDataFull[seasonal_periods_l1_col],
        )

        # insert version column
        MonthlyFinalDataFull.insert(0, VersionColumn, input_version)

        req_cols_in_output = [
            VersionColumn,
            Grain1,
            Grain2,
            Grain3,
            seasonality_l1_col,
            seasonal_periods_l1_col,
            trend_strength_l1_col,
            seasonal_strength_l1_col,
            entropy_l1_col,
        ]
        logger.info(
            "Filtering {} from output dataframe ...".format(req_cols_in_output)
        )

        MonthlyFinalDataFull = MonthlyFinalDataFull[req_cols_in_output]
        logger.info("Successfully executed {} ...".format(plugin_name))

    except Exception as e:
        logger.exception(e)
        MonthlyFinalDataFull = get_o9_empty_df()

    return MonthlyFinalDataFull


