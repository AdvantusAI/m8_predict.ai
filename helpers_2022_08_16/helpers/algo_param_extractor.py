"""
Version : 2022.08.16
Maintained by : pmm_algocoe@o9solutions.com
"""
import logging

import pandas as pd

logger = logging.getLogger("o9_logger")


class AlgoParamExtractor:
    def __init__(
        self,
        forecast_level,
        intersection,
        AlgoParams,
        stat_algo_col,
        stat_parameter_col,
        system_stat_param_value_col,
    ):
        self.the_forecast_level = forecast_level
        self.the_intersection = intersection
        self.AlgoParams = AlgoParams
        self.stat_algo_col = stat_algo_col
        self.stat_parameter_col = stat_parameter_col
        self.system_stat_param_value_col = system_stat_param_value_col

        self.AlgoParams[self.the_forecast_level] = self.AlgoParams[
            self.the_forecast_level
        ].astype(str)

        # create dummy filter clause with all True
        self.filter_clause = pd.Series([True] * len(self.AlgoParams))

        # Combine elements in tuple into the filter clause to filter for the right intersection
        for the_index, the_level in enumerate(forecast_level):
            self.filter_clause = self.filter_clause & (
                self.AlgoParams[the_level] == self.the_intersection[the_index]
            )

        # filter relevant data for the intersection from AlgoParams
        self.the_intersection_df = self.AlgoParams[self.filter_clause]

    def extract_param_value(
        self,
        algorithm: str,
        parameter: str,
    ) -> object:
        # Add filter clause with algorithm and parameter filter
        filter_clause = (
            self.the_intersection_df[self.stat_algo_col] == algorithm
        ) & (self.the_intersection_df[self.stat_parameter_col] == parameter)

        param_value = 0
        if len(self.the_intersection_df[filter_clause]) > 0:
            param_value = float(
                self.the_intersection_df[filter_clause][
                    self.system_stat_param_value_col
                ].iloc[0]
            )
            logger.debug(
                "the_intersection : {}, algorithm : {}, parameter : {}, param_value : {}".format(
                    self.the_intersection,
                    algorithm,
                    parameter,
                    param_value,
                )
            )
        else:
            logger.error(
                "Parameter Value not found in AlgoParameters for {}, {}".format(
                    algorithm, parameter
                )
            )
        return param_value
