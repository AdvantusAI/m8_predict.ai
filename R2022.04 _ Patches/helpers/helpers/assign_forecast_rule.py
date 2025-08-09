"""
Version : dev
Maintained by : dpref@o9solutions.com
"""
import numpy as np
import pandas as pd


def assign_forecast_rule(
    data: pd.DataFrame,
    vol_segment_col: str,
    variability_segment_col: str,
    intermittency_col: str,
    plc_status_col: str,
    trend_col: str,
    seasonality_col: str,
) -> pd.DataFrame:
    """
    Assigns forecasting rule based on conditions and returns rule assigned dataframe
    """
    rule_col = "Rule"
    rule_desc_col = "Desc"
    rule_algo_col = "Algo"

    rule_mapping = pd.DataFrame(
        {
            rule_col: [1, 2, 3, 4, 5, 6, 7, 8],
            rule_desc_col: [
                "Intermittent",
                "Discontinued",
                "New Launch",
                "Low variability",
                "Low volume, High Variability",
                "High Volume, High Variability with trend",
                "High Volume, High Variability with seasonality, no trend",
                "No Match",
            ],
            rule_algo_col: [
                "Croston, Seasonal Naive YoY",
                "DES, TES",
                "DES, Moving Average, TES",
                "AR-NNET,Auto ARIMA,sARIMA,STLF,TBATS,TES,Theta,Prophet",
                "DES,Moving Average,Naive Random Walk,SES",
                "Auto ARIMA,sARIMA,TES,Theta",
                "sARIMA,STLF,TBATS,TES",
                "ETS",
            ],
        }
    )

    conditions = [
        data[intermittency_col] == "YES",
        (data[intermittency_col] == "NO") & (data[plc_status_col] == "DISC"),
        (data[intermittency_col] == "NO")
        & (data[plc_status_col] == "NEW LAUNCH"),
        (data[intermittency_col] == "NO")
        & (data[plc_status_col].isin(["MATURE"]))
        & (data[variability_segment_col].isin(["X"])),
        (data[intermittency_col] == "NO")
        & (data[plc_status_col].isin(["MATURE"]))
        & (data[variability_segment_col].isin(["Y"]))
        & ((data[vol_segment_col].isin(["B"]))),
        (data[intermittency_col] == "NO")
        & (data[plc_status_col].isin(["MATURE"]))
        & (data[variability_segment_col].isin(["Y"]))
        & ((data[vol_segment_col].isin(["A"])))
        & (data[trend_col].isin(["UPWARD", "DOWNWARD"])),
        (data[intermittency_col] == "NO")
        & (data[plc_status_col].isin(["MATURE"]))
        & (data[variability_segment_col].isin(["Y"]))
        & ((data[vol_segment_col].isin(["A"])))
        & (data[trend_col].isin(["NO TREND"]))
        & (data[seasonality_col].isin(["Exists"])),
    ]

    choices = [1, 2, 3, 4, 5, 6, 7]

    # assign rule to relevant conditions
    data[rule_col] = np.select(conditions, choices, default=8)

    # join with master data to get algorithm and description
    data = data.merge(rule_mapping, on=rule_col)

    return data
