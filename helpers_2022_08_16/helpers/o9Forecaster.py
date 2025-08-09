import logging

import numpy as np
import pandas as pd
from o9Reference.stat_utils.get_moving_avg_forecast import (
    get_moving_avg_forecast,
)
from sktime.forecasting.base import ForecastingHorizon
from statsmodels.tsa.arima.model import ARIMA as statsmodelsArima
from statsmodels.tsa.forecasting.stl import STLForecast

from helpers.utils import get_ts_freq_prophet

logger = logging.getLogger("o9_logger")
from sklearn.linear_model import LinearRegression
from sktime.forecasting.compose import make_reduction


class o9Forecaster:
    def __init__(
        self,
        train,
        seasonal_periods,
        in_sample_flag,
        forecast_horizon,
        confidence_interval_alpha,
    ):
        self.train = train
        self.seasonal_periods = seasonal_periods
        self.in_sample_flag = in_sample_flag
        self.forecast_horizon = forecast_horizon
        self.fcst_horizon = np.arange(1, self.forecast_horizon + 1)
        self.confidence_interval_alpha = confidence_interval_alpha

        # random start date to assign for prophet/stlf
        self.random_start_date = "2016-01-01"

    def __initialize_results(self):
        # Initialize with Nan to avoid reference before assignment error
        the_forecast = pd.Series([np.nan] * self.forecast_horizon)
        the_forecast_intervals = pd.DataFrame()
        return the_forecast, the_forecast_intervals

    def get_snaive_forecast(self):
        the_forecast, the_forecast_intervals = self.__initialize_results()
        try:
            if len(self.train) < self.seasonal_periods:
                # if there are not enough seasonal points present, pad with zeros in points prior to start date
                num_points_to_pad = self.seasonal_periods - len(self.train)
                snaive_train = pd.Series(
                    np.pad(
                        self.train.values,
                        (num_points_to_pad, 0),
                        "constant",
                    )
                )
            else:
                snaive_train = self.train

            if self.in_sample_flag:
                if len(snaive_train) == self.seasonal_periods:
                    # cannot generate in sample predictions if only one cycle data is present
                    pass
                else:
                    snaive_train_df = pd.DataFrame({"actual": snaive_train})

                    # shift actuals by seasonal periods to form lagged series
                    snaive_train_df["pred"] = snaive_train_df["actual"].shift(
                        self.seasonal_periods
                    )

                    # remove rows where actuals are NA (to filter only in sample rows)
                    snaive_train_df = snaive_train_df[
                        snaive_train_df["actual"].notna()
                    ]

                    # filter the series
                    the_forecast = snaive_train_df["pred"]
            else:
                # get last n values based on seasonality
                last_n_values = snaive_train.tail(self.seasonal_periods).values

                # get number of times to repeat this series
                num_repeats = int(
                    np.ceil(self.forecast_horizon / len(last_n_values))
                )

                # repeat the snaive array
                forecast_array = np.tile(last_n_values, num_repeats)

                # collect required points as specified in forecast horizon
                the_forecast = pd.Series(forecast_array).head(
                    self.forecast_horizon
                )
        except Exception as e:
            logger.exception(e)
        return the_forecast, the_forecast_intervals

    def get_arnnet_forecast(self):
        the_forecast, the_forecast_intervals = self.__initialize_results()
        try:
            # AR - NNET is not available in Python, implementing a Linear Regression model which uses past 6 periods
            if self.in_sample_flag:
                # in sample predictions are not yet implemented for sktime make_reduction
                pass
            else:
                # create linear regression model
                regressor = LinearRegression()

                # reduce the data to form
                the_estimator = make_reduction(
                    regressor,
                    window_length=self.seasonal_periods // 2,
                    strategy="recursive",
                )

                # fit model
                the_estimator.fit(self.train)

                # try to get predictions with intervals
                (
                    the_forecast,
                    the_forecast_intervals,
                ) = the_estimator.predict(
                    fh=self.fcst_horizon,
                    return_pred_int=True,
                    alpha=self.confidence_interval_alpha,
                )
        except Exception as e:
            logger.exception(e)
        return the_forecast, the_forecast_intervals

    def get_moving_avg_forecast(self, ma_periods):
        the_forecast, the_forecast_intervals = self.__initialize_results()
        try:
            if self.in_sample_flag:
                # take in sample values
                the_forecast = pd.Series(
                    self.train.rolling(window=ma_periods).mean()
                ).shift(1)
                # fill NAs with series mean
                the_forecast.fillna(np.mean(self.train), inplace=True)
            else:
                the_forecast = pd.Series(
                    get_moving_avg_forecast(
                        data=self.train.values,
                        moving_avg_periods=ma_periods,
                        forecast_horizon=self.forecast_horizon,
                    )
                )
        except Exception as e:
            logger.exception(e)
        return the_forecast, the_forecast_intervals

    def get_naive_random_walk_forecast(self):
        the_forecast, the_forecast_intervals = self.__initialize_results()
        try:
            # collect latest history datapoint
            latest_history_value = self.train.values[-1]

            # populate above value as forecast into future n points
            the_forecast = pd.Series(
                [latest_history_value] * self.forecast_horizon
            )
        except Exception as e:
            logger.exception(e)
        return the_forecast, the_forecast_intervals

    def get_stlf_forecast(self):
        the_forecast, the_forecast_intervals = self.__initialize_results()
        try:
            # requires statsmodels library, hence the custom logic
            stlf_train_index = pd.date_range(
                start=self.random_start_date,
                periods=len(self.train),
                freq=get_ts_freq_prophet(self.seasonal_periods),
            )
            stlf_train_test_index = pd.date_range(
                start=self.random_start_date,
                periods=len(self.train) + self.forecast_horizon,
                freq=get_ts_freq_prophet(self.seasonal_periods),
            )
            stlf_train = pd.Series(self.train.values, index=stlf_train_index)
            the_stlf_model = STLForecast(stlf_train, statsmodelsArima)
            the_stlf_fit = the_stlf_model.fit()

            if self.in_sample_flag:
                the_stlf_forecast_df = the_stlf_fit.get_prediction(
                    start=stlf_train_index[0], end=stlf_train_index[-1]
                ).summary_frame()

            else:

                the_stlf_forecast_df = the_stlf_fit.get_prediction(
                    start=stlf_train_test_index[len(stlf_train)],
                    end=stlf_train_test_index[-1],
                ).summary_frame()

            the_forecast = pd.Series(the_stlf_forecast_df["mean"].values)
            the_forecast_intervals["lower"] = the_stlf_forecast_df[
                "mean_ci_lower"
            ].values
            the_forecast_intervals["upper"] = the_stlf_forecast_df[
                "mean_ci_upper"
            ].values
        except Exception as e:
            logger.exception(e)
        return the_forecast, the_forecast_intervals

    def get_arima_sarima_forecast(self, the_estimator):
        the_forecast, the_forecast_intervals = self.__initialize_results()
        try:
            # fit model
            the_estimator.fit(self.train)
            if self.in_sample_flag:
                (
                    the_forecast,
                    the_forecast_interval_array,
                ) = the_estimator.predict_in_sample(
                    return_conf_int=True,
                    alpha=self.confidence_interval_alpha,
                )

            else:
                the_forecast, the_forecast_interval_array = pd.Series(
                    the_estimator.predict(
                        self.forecast_horizon,
                        return_conf_int=True,
                        alpha=self.confidence_interval_alpha,
                    )
                )

            # convert array to pandas series
            the_forecast = pd.Series(the_forecast)
            the_forecast_intervals = pd.DataFrame(
                the_forecast_interval_array, columns=["lower", "upper"]
            )
        except Exception as e:
            logger.exception(e)
        return the_forecast, the_forecast_intervals

    def get_prophet_forecast(self, the_estimator):
        the_forecast, the_forecast_intervals = self.__initialize_results()
        try:
            # generate a time series index based on frequency, assign any random start date and generate equally distanced time series
            # prophet requires equally spaced time series, planning calendar will not work
            prophet_train_index = pd.date_range(
                start=self.random_start_date,
                periods=len(self.train),
                freq=get_ts_freq_prophet(self.seasonal_periods),
            )
            prophet_train = pd.Series(
                self.train.values, index=prophet_train_index
            )
            the_estimator.fit(prophet_train)

            if self.in_sample_flag:
                fcst_horizon = ForecastingHorizon(
                    values=prophet_train_index, is_relative=False
                )

            (the_forecast, the_forecast_intervals,) = the_estimator.predict(
                fh=fcst_horizon,
                return_pred_int=True,
                alpha=self.confidence_interval_alpha,
            )
        except Exception as e:
            logger.exception(e)
        return the_forecast, the_forecast_intervals

    def get_ses_des_tes_forecast(self, the_estimator):
        the_forecast, the_forecast_intervals = self.__initialize_results()
        try:
            the_estimator = the_estimator.fit(maxiter=1000)
            start_index, end_index = (
                self.train.index.min(),
                self.train.index.max(),
            )
            if self.in_sample_flag:
                the_pred_with_bounds = the_estimator.get_prediction(
                    start=start_index, end=end_index
                ).summary_frame(alpha=self.confidence_interval_alpha)
            else:
                the_pred_with_bounds = the_estimator.get_prediction(
                    start=end_index + 1,
                    end=end_index + self.forecast_horizon,
                ).summary_frame(alpha=self.confidence_interval_alpha)

            the_forecast = pd.Series(the_pred_with_bounds["mean"].values)
            the_forecast_intervals["lower"] = the_pred_with_bounds[
                "pi_lower"
            ].values
            the_forecast_intervals["upper"] = the_pred_with_bounds[
                "pi_upper"
            ].values
        except Exception as e:
            logger.exception(e)
        return the_forecast, the_forecast_intervals

    def get_sktime_forecast(self, the_estimator, the_model_name):
        the_forecast, the_forecast_intervals = self.__initialize_results()
        try:
            # TBATS, Theta, ETS, Croston Estimators
            # fit model
            the_estimator.fit(self.train)

            if self.in_sample_flag:
                fcst_horizon = ForecastingHorizon(
                    self.train.index, is_relative=False
                )
            else:
                fcst_horizon = self.fcst_horizon

            # bug present for TBATS when trying to get prediction intervals
            # TODO: TBATS : In-sample, cannot produce prediction intervals: bug in sktime 0.7 library, resolved in sktime 0.10.0
            if the_model_name == "TBATS" and self.in_sample_flag:
                the_forecast = the_estimator.predict(fh=fcst_horizon)
            # TODO: TBATS : Fails to generate when number of training records are less than forecast horizon, resolution version not known, need to check
            elif the_model_name == "TBATS" and (
                len(self.train) < self.forecast_horizon
            ):
                # cannot produce predictions, results in error, hence continue to next model
                pass
            else:
                # try to get predictions with intervals
                (
                    the_forecast,
                    the_forecast_intervals,
                ) = the_estimator.predict(
                    fh=fcst_horizon,
                    return_pred_int=True,
                    alpha=self.confidence_interval_alpha,
                )
        except NotImplementedError:
            # capture the not implemented error for prediction interval
            # logger.error(NotImplementedError.__name__)
            # re attempt predictions
            try:
                the_forecast = the_estimator.predict(fh=fcst_horizon)
            except Exception as e:
                logger.exception(e)
        except Exception as e:
            logger.exception(e)
        return the_forecast, the_forecast_intervals
