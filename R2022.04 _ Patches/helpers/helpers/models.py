"""
Version : R2022.05
Maintained by : dpref@o9solutions.com
"""
import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.forecasting.stl import STLForecast

from helpers.algo_param_extractor import AlgoParamExtractor
from helpers.model_params import get_fitted_params
from helpers.utils import get_bound_col_name
from helpers.utils import get_measure_name
from helpers.utils import get_model_desc_name
from helpers.utils import get_ts_freq_prophet

logger = logging.getLogger("o9_logger")
from pmdarima.arima import AutoARIMA, ARIMA
from sktime.forecasting.croston import Croston
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.tbats import TBATS
from o9Reference.stat_utils.get_moving_avg_forecast import (
    get_moving_avg_forecast,
)
from sktime.forecasting.compose import make_reduction
from sklearn.neural_network import MLPRegressor
from sktime.forecasting.fbprophet import Prophet
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.base import ForecastingHorizon
import copy
from statsmodels.tsa.arima.model import ARIMA as statsmodelsArima


def fit_models(
    all_estimator_list: list,
    algo_list: list,
    train: pd.Series,
    forecast_horizon: int,
    confidence_interval_alpha: float,
    seasonal_periods: int,
    validation_method: str,
    is_first_pass: bool,
) -> (pd.DataFrame, pd.DataFrame):
    # Dataframe to store predictions with intervals
    all_model_pred = pd.DataFrame()

    # Dataframe to store model descriptions
    all_model_descriptions = pd.DataFrame(index=[0])

    # random start date to assign for prophet/stlf
    random_start_date = "2016-01-01"

    fcst_horizon = np.arange(1, forecast_horizon + 1)

    in_sample_flag = False
    if is_first_pass and validation_method == "In Sample":
        in_sample_flag = True

    try:
        # filter estimators to be evaluated from the algo list
        # x here will be a tuple of ('name', 'estimator'), we are checking if name exists in algo list
        estimators_to_be_run = [
            x for x in all_estimator_list if x[0] in algo_list
        ]

        # Iterate through all models
        for (the_model_name, the_estimator) in estimators_to_be_run:

            logger.info("---- Fitting {} ...".format(the_model_name))

            # Initialize with Nan to avoid reference before assignment error
            the_forecast = pd.Series([np.nan] * forecast_horizon)
            the_forecast_intervals = pd.DataFrame()

            # get measure name, define forecast horizon
            the_measure_name = get_measure_name(the_model_name)

            try:
                if the_model_name == "Seasonal Naive YoY":
                    if in_sample_flag:
                        # populate same values in train set as forecast
                        the_forecast = pd.Series(train.copy())
                    else:
                        # get last n values based on seasonality
                        last_n_values = train.tail(seasonal_periods).values

                        # get number of times to repeat this series
                        num_repeats = int(
                            np.ceil(forecast_horizon / len(last_n_values))
                        )

                        # repeat the snaive array
                        forecast_array = np.tile(last_n_values, num_repeats)

                        # collect required points as specified in forecast horizon
                        the_forecast = pd.Series(forecast_array).head(
                            forecast_horizon
                        )

                elif the_model_name == "AR-NNET":
                    if in_sample_flag:
                        # in sample predictions are not yet implemented for sktime make_reduction
                        pass
                    else:
                        # create perceptron model
                        regressor = MLPRegressor(
                            hidden_layer_sizes=(6, 3),
                            learning_rate="adaptive",
                            random_state=1,
                            max_iter=250,
                        )

                        # reduce the data to form
                        the_estimator = make_reduction(
                            regressor,
                            window_length=seasonal_periods // 2,
                            strategy="recursive",
                        )

                        # fit model
                        the_estimator.fit(train.astype("float64"))

                        # try to get predictions with intervals
                        (
                            the_forecast,
                            the_forecast_intervals,
                        ) = the_estimator.predict(
                            fh=fcst_horizon,
                            return_pred_int=True,
                            alpha=confidence_interval_alpha,
                        )

                elif the_model_name == "Moving Average":
                    # Moving average model - implemented in o9Reference - passing MA periods in estimator
                    ma_periods = int(the_estimator)
                    if in_sample_flag:
                        # take in sample values
                        the_forecast = pd.Series(
                            train.rolling(window=ma_periods).mean()
                        ).shift(1)
                        # fill NAs with series mean
                        the_forecast.fillna(np.mean(train), inplace=True)
                    else:
                        the_forecast = pd.Series(
                            get_moving_avg_forecast(
                                data=train.values,
                                moving_avg_periods=ma_periods,
                                forecast_horizon=forecast_horizon,
                            )
                        )
                elif the_model_name == "Naive Random Walk":
                    # collect latest history datapoint
                    latest_history_value = train.values[-1]

                    # populate above value as forecast into future n points
                    the_forecast = pd.Series(
                        [latest_history_value] * forecast_horizon
                    )

                elif the_model_name == "STLF":
                    # requires statsmodels library, hence the custom logic
                    stlf_train_index = pd.date_range(
                        start=random_start_date,
                        periods=len(train),
                        freq=get_ts_freq_prophet(seasonal_periods),
                    )
                    stlf_train = pd.Series(
                        train.values, index=stlf_train_index
                    )
                    stlf = STLForecast(stlf_train, statsmodelsArima)
                    res = stlf.fit()
                    if in_sample_flag:
                        the_forecast = res.get_prediction(
                            start=stlf_train_index[0], end=stlf_train_index[-1]
                        ).predicted_mean
                    else:
                        the_forecast = res.forecast(forecast_horizon)

                elif the_model_name in ["Auto ARIMA", "sARIMA"]:
                    # fit model
                    the_estimator.fit(train.astype("float64"))
                    if in_sample_flag:
                        the_forecast = pd.Series(
                            the_estimator.predict_in_sample()
                        )
                    else:
                        the_forecast = pd.Series(
                            the_estimator.predict(forecast_horizon)
                        )

                elif the_model_name == "Prophet":
                    # generate a time series index based on frequency, assign any random start date and generate equally distanced time series
                    # prophet requires equally spaced time series, planning calendar will not work
                    prophet_train_index = pd.date_range(
                        start=random_start_date,
                        periods=len(train),
                        freq=get_ts_freq_prophet(seasonal_periods),
                    )
                    prophet_train = pd.Series(
                        train.values, index=prophet_train_index
                    )
                    the_estimator.fit(prophet_train)

                    if in_sample_flag:
                        fcst_horizon = ForecastingHorizon(
                            values=prophet_train_index, is_relative=False
                        )
                        (
                            the_forecast,
                            the_forecast_intervals,
                        ) = the_estimator.predict(
                            fh=fcst_horizon,
                            return_pred_int=True,
                            alpha=confidence_interval_alpha,
                        )
                    else:
                        (
                            the_forecast,
                            the_forecast_intervals,
                        ) = the_estimator.predict(
                            fh=fcst_horizon,
                            return_pred_int=True,
                            alpha=confidence_interval_alpha,
                        )

                else:
                    # All sktime estimators
                    # fit model
                    the_estimator.fit(train.astype("float64"))

                    if in_sample_flag:
                        fcst_horizon = ForecastingHorizon(
                            train.index, is_relative=False
                        )

                    # bug present for TBATS when trying to get prediction intervals
                    if the_model_name != "TBATS":
                        # try to get predictions with intervals
                        (
                            the_forecast,
                            the_forecast_intervals,
                        ) = the_estimator.predict(
                            fh=fcst_horizon,
                            return_pred_int=True,
                            alpha=confidence_interval_alpha,
                        )
                    else:
                        the_forecast = the_estimator.predict(fh=fcst_horizon)
                        the_forecast_intervals = pd.DataFrame(
                            {"lower": the_forecast, "upper": the_forecast}
                        )

                    # fill nulls if any with average value
                    the_forecast.fillna(np.mean(train), inplace=True)
                    the_forecast_intervals.fillna(np.mean(train), inplace=True)

            except NotImplementedError:
                # capture the not implemented error for prediction interval
                # logger.error(NotImplementedError.__name__)
                # re attempt predictions
                the_forecast = the_estimator.predict(fh=fcst_horizon)

                # fill nulls with average value for seasonal naive models etc
                the_forecast.fillna(np.mean(train), inplace=True)
            except Exception as e:
                # capture any other exception
                logger.error("Error for model : {}".format(the_model_name))
                logger.exception(e)
            finally:
                assert (
                    type(the_forecast) == pd.Series
                ), "Datatype of the_forecast should be pandas series ..."

                # clip lower fails if there are NAs in the array, convert to array to avoid index related issues while writing to all model_pred
                if not np.isnan(the_forecast).any():
                    the_forecast = the_forecast.clip(lower=0).values
                else:
                    the_forecast = the_forecast.values

                all_model_pred[the_measure_name] = the_forecast

                # check and populate upper and lower bounds
                all_model_pred = populate_bounds(
                    all_model_pred,
                    the_measure_name,
                    confidence_interval_alpha,
                    the_forecast_intervals,
                    the_forecast,
                )

                # get fitted params
                fitted_params = get_fitted_params(
                    the_model_name, the_estimator
                )

                # create model description string
                the_model_desc_string = "Algo = {} | Parameters = {} | Validation Method = {} ({})".format(
                    the_model_name,
                    fitted_params,
                    validation_method,
                    forecast_horizon,
                )
                all_model_descriptions[
                    get_model_desc_name(the_model_name)
                ] = the_model_desc_string

    except Exception as ex:
        logger.exception(ex)

    return all_model_pred, all_model_descriptions


def populate_bounds(
    all_model_pred,
    the_measure_name,
    confidence_interval_alpha,
    the_forecast_intervals,
    the_forecast,
):
    lower_bound_col = get_bound_col_name(
        the_measure_name, confidence_interval_alpha, "LB"
    )
    upper_bound_col = get_bound_col_name(
        the_measure_name, confidence_interval_alpha, "UB"
    )

    # if prediction intervals are not available, use values from forecast itself to populate
    if the_forecast_intervals.empty:
        all_model_pred[lower_bound_col] = the_forecast
        all_model_pred[upper_bound_col] = the_forecast
    else:
        # extract values from dataframe, clip lower to zero
        all_model_pred[lower_bound_col] = np.clip(
            the_forecast_intervals["lower"].values,
            a_min=0,
            a_max=None,
        )
        all_model_pred[upper_bound_col] = np.clip(
            the_forecast_intervals["upper"].values,
            a_min=0,
            a_max=None,
        )
    return all_model_pred


def get_algo_list(
    AlgoList, forecast_level, the_intersection, assigned_algo_list_col
):
    logger.info("Extracting algorithm list ...")
    algo_list = []

    # create dummy filter clause with all True
    filter_clause = pd.Series([True] * len(AlgoList))

    AlgoList[forecast_level] = AlgoList[forecast_level].astype(str)

    # Combine elements in tuple into the filter clause to filter for the right intersection
    for the_index, the_level in enumerate(forecast_level):
        filter_clause = filter_clause & (
            AlgoList[the_level] == the_intersection[the_index]
        )

    if len(AlgoList[filter_clause]) > 0:
        algo_list = (
            AlgoList[filter_clause][assigned_algo_list_col].iloc[0].split(",")
        )
        logger.info("------ algo_list : {}".format(algo_list))

    return algo_list


def train_models_for_one_intersection(
    df,
    forecast_level,
    TimeLevel,
    history_measure,
    validation_periods,
    validation_method,
    algo_list_master,
    seasonal_periods,
    forecast_period_dates,
    confidence_interval_alpha,
    assigned_algo_list_col,
    AlgoParameters,
    stat_algo_col,
    stat_parameter_col,
    system_stat_param_value_col,
):
    valid_pred_df, forecast_df, valid_model_desc_df = (
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )
    the_intersection = tuple(df[forecast_level].iloc[0])
    try:
        logger.info("the_intersection  : {}".format(the_intersection))

        # split data into train, validation
        time_series_df = df[[TimeLevel, history_measure]].reset_index(
            drop=True
        )

        # filter non null rows
        time_series_df = time_series_df[
            time_series_df[history_measure].notna()
        ]
        time_series_df.reset_index(drop=True, inplace=True)
        time_series = time_series_df[history_measure]

        # get forecast horizon
        forecast_horizon = len(forecast_period_dates)

        # get list of algos to be run
        algo_list = get_algo_list(
            algo_list_master,
            forecast_level,
            the_intersection,
            assigned_algo_list_col,
        )

        if len(algo_list) == 0:
            logger.warning(
                "No algorithms configured for the intersection : {}".format(
                    the_intersection
                )
            )
            return valid_pred_df, forecast_df, valid_model_desc_df

        logger.debug("Extracting all model parameter values ...")

        # Initialize param extractor class
        param_extractor = AlgoParamExtractor(
            forecast_level=forecast_level,
            intersection=the_intersection,
            AlgoParams=AlgoParameters,
            stat_algo_col=stat_algo_col,
            stat_parameter_col=stat_parameter_col,
            system_stat_param_value_col=system_stat_param_value_col,
        )

        if validation_method == "In Sample":
            logger.warning("Using in sample validation ...")
            # use in sample validation
            train, valid = time_series, time_series
        elif validation_method == "Out Sample":
            # Check if data conditions satisfy for out sample validation
            if (
                len(time_series) <= validation_periods
                or len(time_series) <= 2 * seasonal_periods
            ):
                logger.warning(
                    "Found only {} datapoints for intersection : {}".format(
                        len(time_series), the_intersection
                    )
                )
                logger.warning("Using in sample validation ...")
                validation_method = "In Sample"
                # use in sample validation
                train, valid = time_series, time_series
            else:
                train, valid = temporal_train_test_split(
                    time_series, test_size=validation_periods
                )
        else:
            raise ValueError(
                "Unknown validation method {}, In Sample and Out Sample are supported".format(
                    validation_method
                )
            )

        logger.info("---- PASS 1 : Fitting models ...")

        pass_1_estimators = get_valid_estimators(
            train=train,
            seasonal_periods=seasonal_periods,
            param_extractor=param_extractor,
        )

        valid_pred_df, valid_model_desc_df = fit_models(
            all_estimator_list=copy.deepcopy(pass_1_estimators),
            algo_list=algo_list,
            train=train,
            forecast_horizon=len(valid),
            confidence_interval_alpha=confidence_interval_alpha,
            seasonal_periods=seasonal_periods,
            validation_method=validation_method,
            is_first_pass=True,
        )
        valid_pred_df.insert(0, history_measure, valid.values)
        valid_pred_df.insert(
            0, TimeLevel, time_series_df.loc[valid.index][TimeLevel].values
        )
        # Add dimension columns
        for the_index, the_level in enumerate(forecast_level):
            valid_pred_df.insert(0, the_level, the_intersection[the_index])

        # filter required points from valid predictions
        valid_pred_df = valid_pred_df.tail(validation_periods)

        # Add dimension columns
        for the_index, the_level in enumerate(forecast_level):
            valid_model_desc_df.insert(
                0, the_level, the_intersection[the_index]
            )

        logger.info("---- PASS 2 : Fitting models ...")

        # we need a copy of estimators because once these model classes are fit, they have to be re initialized (call by reference))
        pass_2_estimators = get_valid_estimators(
            train=time_series,
            seasonal_periods=seasonal_periods,
            param_extractor=param_extractor,
        )

        forecast_df, model_desc_df = fit_models(
            all_estimator_list=copy.deepcopy(pass_2_estimators),
            algo_list=algo_list,
            train=time_series,
            forecast_horizon=forecast_horizon,
            confidence_interval_alpha=confidence_interval_alpha,
            seasonal_periods=seasonal_periods,
            validation_method="None",
            is_first_pass=False,
        )
        forecast_df.insert(0, TimeLevel, forecast_period_dates)
        # Add dimension columns
        for the_index, the_level in enumerate(forecast_level):
            forecast_df.insert(0, the_level, the_intersection[the_index])
    except Exception as e:
        logger.exception(e)

    return valid_pred_df, forecast_df, valid_model_desc_df


def get_valid_estimators(
    train: pd.Series,
    seasonal_periods: int,
    param_extractor: AlgoParamExtractor,
) -> list:
    # Models and Parameters
    # ETS - No parameters
    # SES
    ses_alpha_lower = param_extractor.extract_param_value(
        algorithm="SES",
        parameter="Alpha Lower",
    )

    ses_alpha_upper = param_extractor.extract_param_value(
        algorithm="SES",
        parameter="Alpha Upper",
    )

    # DES
    des_alpha_lower = param_extractor.extract_param_value(
        algorithm="DES",
        parameter="Alpha Lower",
    )

    des_alpha_upper = param_extractor.extract_param_value(
        algorithm="DES",
        parameter="Alpha Upper",
    )

    des_beta_lower = param_extractor.extract_param_value(
        algorithm="DES",
        parameter="Beta Lower",
    )

    des_beta_upper = param_extractor.extract_param_value(
        algorithm="DES",
        parameter="Beta Upper",
    )

    des_phi_lower = param_extractor.extract_param_value(
        algorithm="DES",
        parameter="Phi Lower",
    )

    des_phi_upper = param_extractor.extract_param_value(
        algorithm="DES",
        parameter="Phi Upper",
    )

    # TES
    tes_alpha_lower = param_extractor.extract_param_value(
        algorithm="TES",
        parameter="Alpha Lower",
    )
    tes_alpha_upper = param_extractor.extract_param_value(
        algorithm="TES",
        parameter="Alpha Upper",
    )
    tes_beta_lower = param_extractor.extract_param_value(
        algorithm="TES",
        parameter="Beta Lower",
    )
    tes_beta_upper = param_extractor.extract_param_value(
        algorithm="TES",
        parameter="Beta Upper",
    )
    tes_gamma_lower = param_extractor.extract_param_value(
        algorithm="TES",
        parameter="Gamma Lower",
    )
    tes_gamma_upper = param_extractor.extract_param_value(
        algorithm="TES",
        parameter="Gamma Upper",
    )
    tes_phi_lower = param_extractor.extract_param_value(
        algorithm="TES",
        parameter="Phi Lower",
    )
    tes_phi_upper = param_extractor.extract_param_value(
        algorithm="TES",
        parameter="Phi Upper",
    )

    # Moving Average
    moving_avg_periods = param_extractor.extract_param_value(
        algorithm="Moving Average",
        parameter="Period",
    )

    # SARIMA
    sarima_ar_order = param_extractor.extract_param_value(
        algorithm="sARIMA",
        parameter="AR Order",
    )
    sarima_diff_order = param_extractor.extract_param_value(
        algorithm="sARIMA",
        parameter="Differencing",
    )
    sarima_ma_order = param_extractor.extract_param_value(
        algorithm="sARIMA",
        parameter="MA Order",
    )
    sarima_seasonal_ar_order = param_extractor.extract_param_value(
        algorithm="sARIMA",
        parameter="Seasonal AR Order",
    )
    sarima_seasonal_diff_order = param_extractor.extract_param_value(
        algorithm="sARIMA",
        parameter="Seasonal Differencing",
    )
    sarima_seasonal_ma_order = param_extractor.extract_param_value(
        algorithm="sARIMA",
        parameter="Seasonal MA Order",
    )

    multiprocessing_num_cores = 4
    train_size = len(train)
    params = {"initialization_method": "estimated"}
    if train_size < 10 or train_size < 2 * seasonal_periods:
        logger.warning(
            "Cannot do parameter search with less than 10 datapoints ..."
        )
        logger.warning("Assigning default parameters")
        params = {
            "initialization_method": "known",
            "initial_level": 0.0,
            "initial_trend": 0.0,
            "initial_seasonal": 0.0,
        }

    non_seasonal_estimators = [
        (
            "SES",
            AutoETS(
                auto=False,
                additive_only=True,
                **params,
                bounds={
                    "smoothing_level": (
                        ses_alpha_lower,
                        ses_alpha_upper,
                    )
                },
                n_jobs=multiprocessing_num_cores,
            ),
        ),
        (
            "DES",
            AutoETS(
                auto=False,
                additive_only=True,
                trend="add",
                damped_trend=True,
                **params,
                bounds={
                    "smoothing_level": (
                        des_alpha_lower,
                        des_alpha_upper,
                    ),
                    "smoothing_trend": (
                        des_beta_lower,
                        des_beta_upper,
                    ),
                    "damping_trend": (
                        des_phi_lower,
                        des_phi_upper,
                    ),
                },
                n_jobs=multiprocessing_num_cores,
            ),
        ),
        (
            "Auto ARIMA",
            AutoARIMA(
                seasonal=False,
                suppress_warnings=True,
                n_jobs=multiprocessing_num_cores,
            ),
        ),
        ("Croston", Croston()),
        ("Moving Average", moving_avg_periods),
        ("Naive Random Walk", "Dummy Placeholder"),
        ("Seasonal Naive YoY", "Dummy Placeholder"),
    ]

    seasonal_estimators = [
        (
            "ETS",
            AutoETS(
                auto=True,
                additive_only=True,
                sp=seasonal_periods,
                n_jobs=multiprocessing_num_cores,
                suppress_warnings=True,
                **params,
            ),
        ),
        (
            "TES",
            AutoETS(
                auto=False,
                additive_only=True,
                trend="add",
                seasonal="add",
                damped_trend=True,
                sp=seasonal_periods,
                bounds={
                    "smoothing_level": (
                        tes_alpha_lower,
                        tes_alpha_upper,
                    ),
                    "smoothing_trend": (
                        tes_beta_lower,
                        tes_beta_upper,
                    ),
                    "smoothing_seasonal": (
                        tes_gamma_lower,
                        tes_gamma_upper,
                    ),
                    "damping_trend": (
                        tes_phi_lower,
                        tes_phi_upper,
                    ),
                },
                n_jobs=multiprocessing_num_cores,
                **params,
            ),
        ),
        (
            "sARIMA",
            ARIMA(
                order=(
                    sarima_ar_order,
                    sarima_diff_order,
                    sarima_ma_order,
                ),
                seasonal_order=(
                    sarima_seasonal_ar_order,
                    sarima_seasonal_diff_order,
                    sarima_seasonal_ma_order,
                    seasonal_periods,
                ),
            ),
        ),
        ("STLF", "Dummy PlaceHolder"),
        (
            "Prophet",
            Prophet(
                yearly_seasonality="auto",
                weekly_seasonality="auto",
                daily_seasonality="auto",
                growth="linear",
                uncertainty_samples=100,
                mcmc_samples=0,
            ),
        ),
        ("AR-NNET", "Dummy Placeholder"),
    ]

    tbats_estimator = [
        (
            "TBATS",
            TBATS(
                use_box_cox=False,
                use_trend=True,
                use_damped_trend=False,
                sp=seasonal_periods,
                use_arma_errors=False,
            ),
        )
    ]

    theta_estimator = [("Theta", ThetaForecaster(sp=seasonal_periods))]

    if train_size > seasonal_periods:
        estimators = (
            non_seasonal_estimators
            + seasonal_estimators
            + tbats_estimator
            + theta_estimator
        )
        # if train_size > 2 * seasonal_periods:
        #     estimators = estimators + tbats_estimator
        #     # Multiplicative Seasonality models require series to be strictly positive
        #     if train.min() > 0:
        #         estimators = estimators + theta_estimator
    else:
        estimators = non_seasonal_estimators
    return estimators
