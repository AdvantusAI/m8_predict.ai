"""
Version : R2022.05
Maintained by : dpref@o9solutions.com
"""
import logging

import pandas as pd
from o9Reference.common_utils.dataframe_utils import create_cartesian_product

logger = logging.getLogger("o9_logger")
import re


def get_default_params(
    stat_algo_col,
    stat_parameter_col,
    stat_param_value_col,
    frequency,
    intersections_master,
):
    master_algo_list = [
        "Auto ARIMA",
        "Croston",
        "DES",
        "DES",
        "DES",
        "DES",
        "DES",
        "DES",
        "Moving Average",
        "SES",
        "SES",
        "TES",
        "TES",
        "TES",
        "TES",
        "TES",
        "TES",
        "TES",
        "TES",
        "Theta",
        "AR-NNET",
        "ETS",
        "Naive Random Walk",
        "sARIMA",
        "sARIMA",
        "sARIMA",
        "sARIMA",
        "sARIMA",
        "sARIMA",
        "Seasonal Naive YoY",
        "STLF",
        "TBATS",
    ]
    param_list = [
        "No Parameters",
        "No Parameters",
        "Alpha Lower",
        "Alpha Upper",
        "Beta Lower",
        "Beta Upper",
        "Phi Lower",
        "Phi Upper",
        "Period",
        "Alpha Lower",
        "Alpha Upper",
        "Alpha Lower",
        "Alpha Upper",
        "Beta Lower",
        "Beta Upper",
        "Gamma Lower",
        "Gamma Upper",
        "Phi Lower",
        "Phi Upper",
        "No Parameters",
        "No Parameters",
        "No Parameters",
        "No Parameters",
        "AR Order",
        "Differencing",
        "MA Order",
        "Seasonal AR Order",
        "Seasonal Differencing",
        "Seasonal MA Order",
        "No Parameters",
        "No Parameters",
        "No Parameters",
    ]
    param_value_list = [
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        get_default_value_for_ma_periods(frequency),
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    # create default params and dummy value column on intersections master df for cross join in pandas
    AlgoParameters_df = pd.DataFrame(
        {
            stat_algo_col: master_algo_list,
            stat_parameter_col: param_list,
            stat_param_value_col: param_value_list,
        }
    )

    # join algo params with intersections master
    AlgoParameters = create_cartesian_product(
        df1=AlgoParameters_df, df2=intersections_master
    )

    return AlgoParameters


def get_default_value_for_ma_periods(frequency):
    """
    Returns default parameter value for moving average periods based on frequency
    """
    if frequency == "Weekly":
        return 13
    elif frequency == "Monthly":
        return 3
    elif frequency == "Quarterly":
        return 1
    else:
        raise ValueError("Invalid frequency {}".format(frequency))


def get_fitted_params(the_model_name, the_estimator):
    # initialize fitted params
    fitted_params = "No Parameters"

    models_with_parameters = [
        "SES",
        "DES",
        "TES",
        "ETS",
        "Auto ARIMA",
        "sARIMA",
    ]

    try:
        if the_model_name == "Moving Average":
            fitted_params = "Period : {}".format(int(the_estimator))

        if the_model_name in models_with_parameters:
            if the_model_name in ["SES", "DES", "TES", "ETS"]:
                param_to_value_mapping = get_param_to_value_mapping(
                    the_estimator
                )
                # exponential smoothing family
                # get param, assign default value of 0, strip leading and trailing spaces from string
                alpha = param_to_value_mapping.get(
                    "smoothing_level", "NA"
                ).strip()
                beta = param_to_value_mapping.get(
                    "smoothing_trend", "NA"
                ).strip()
                gamma = param_to_value_mapping.get(
                    "smoothing_seasonal", "NA"
                ).strip()
                phi = param_to_value_mapping.get("damping_trend", "NA").strip()

                fitted_params = (
                    "Alpha = {}, Beta = {}, Gamma = {}, Phi = {}".format(
                        alpha, beta, gamma, phi
                    )
                )
            elif the_model_name in ["Auto ARIMA", "sARIMA"]:
                p, d, q, P, D, Q = 0, 0, 0, 0, 0, 0
                summary_string = str(the_estimator.summary())
                if the_model_name == "Auto ARIMA":
                    param = re.findall(
                        "SARIMAX\(([0-9]+), ([0-9]+), ([0-9]+)",
                        summary_string,
                    )
                else:
                    # TODO : Find pattern in case of seasonal ARIMA
                    param = re.findall(
                        "SARIMAX\(([0-9]+), ([0-9]+), ([0-9]+)\)x\(([0-9]+), ([0-9]+), ([0-9]+), ([0-9]+)\)",
                        summary_string,
                    )

                    # there might be cases where the seasonal order does not exist
                    if len(param) == 0:
                        param = re.findall(
                            "SARIMAX\(([0-9]+), ([0-9]+), ([0-9]+)",
                            summary_string,
                        )

                if len(param) > 0:
                    p, d, q = (
                        int(param[0][0]),
                        int(param[0][1]),
                        int(param[0][2]),
                    )
                    if len(param) > 1:
                        P, D, Q = (
                            int(param[1][0]),
                            int(param[1][1]),
                            int(param[1][2]),
                        )
                fitted_params = (
                    "p = {}, d = {}, q = {}, P = {}, D = {}, Q = {}".format(
                        p, d, q, P, D, Q
                    )
                )

    except Exception as e:
        logger.error(
            "Error while trying to fetch fitted model params for {}".format(
                the_model_name
            )
        )
        logger.exception(e)

    return fitted_params


def get_param_to_value_mapping(the_estimator):
    # re.findall('smoothing_level     \d+.\d+', summary_string)
    coefficients_df = pd.DataFrame(the_estimator.summary().tables[1])
    coefficients_df = coefficients_df[coefficients_df.columns[:2]]
    coefficients_df.columns = ["param", "value"]
    coefficients_df["param"] = coefficients_df["param"].astype("str")
    coefficients_df["value"] = coefficients_df["value"].astype("str")
    # create dictionary for easier lookup and default value assignment
    param_to_value_mapping = dict(
        zip(
            list(coefficients_df["param"]),
            list(coefficients_df["value"]),
        )
    )
    return param_to_value_mapping
