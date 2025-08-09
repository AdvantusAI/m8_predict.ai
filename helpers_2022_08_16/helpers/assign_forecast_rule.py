"""
Version : 2022.08.16
Maintained by : pmm_algocoe@o9solutions.com
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("o9_logger")


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


def assign_rules(
    segmentation_output: pd.DataFrame,
    rule_df: pd.DataFrame,
    column_mapping: dict,
    rule_col: str,
    algo_col: str,
) -> pd.DataFrame:
    logger.info("Inside assign_rules function ...")
    try:
        # assign None value
        segmentation_output[rule_col] = None

        # iterate through every row in segmentation output
        for seg_idx, the_seg_row in segmentation_output.iterrows():
            # logger.info("--- seg_idx : {}".format(seg_idx))

            # iterate through every rule in rule df and check if matches
            for rule_idx, the_rule_row in rule_df.iterrows():

                # flag to denote whether any of the rule conditions are not met, initialize to False
                proceed_to_next_rule = False

                # logger.info("------ rule_idx : {}".format(rule_idx))
                # logger.info("------ the_rule_row : \n{}".format(the_rule_row))

                # create and condition with all the clauses
                for the_seg_col, the_rule_col in column_mapping.items():

                    # logger.info("------ the_seg_col : {}".format(the_seg_col))
                    # logger.info("------ the_rule_col : {}".format(the_rule_col))

                    # collect actual value and the value in rule
                    actual_value = the_seg_row[the_seg_col].strip()
                    rule_value = the_rule_row[the_rule_col].strip()

                    # If rule value is N/A, ignore this condition and proceed to check next condition in the rule
                    if rule_value == "N/A":
                        # logger.info("rule_value is N/A, proceeding to next condition".format(rule_value))
                        this_condition_satisfied = True
                    else:
                        # logger.info("------ actual_value : {}".format(actual_value))
                        # logger.info("------ rule_value : {}".format(rule_value))

                        # check for in condition
                        if "," in rule_value:

                            # volume segment rule could be 'A,B', split on comma and check if LHS is present in the ['A', 'B']
                            if actual_value in rule_value.split(","):
                                this_condition_satisfied = True
                            else:
                                this_condition_satisfied = False
                        else:
                            # check for equality
                            if actual_value == rule_value:
                                this_condition_satisfied = True
                            else:
                                this_condition_satisfied = False

                    # if any one of the rule conditions is not satisfied, break and proceed to next rule
                    if not this_condition_satisfied:
                        proceed_to_next_rule = True
                        break

                if proceed_to_next_rule:
                    # logger.info("One of the conditions not met, proceeding to next rule ...")
                    continue

                # logger.info("rule_matched for rule : {}".format(the_rule_row))
                # update the rule value with the appropriate value
                segmentation_output.loc[seg_idx, rule_col] = the_rule_row[
                    rule_col
                ]
                # logger.info("segmentation_output : \n{}".format(segmentation_output))

                # break out to next row in segmentation dataframe
                break

        # identify the rule for which all clauses are NO MATCH
        filter_clause = pd.Series([True] * len(rule_df))
        for the_rule_col in column_mapping.values():
            filter_clause = filter_clause & (
                rule_df[the_rule_col].str.strip() == "NO MATCH"
            )

        # fill no match rule as fallback value
        if len(rule_df[filter_clause]) > 0:
            no_match_rule = rule_df[filter_clause][rule_col].iloc[0]
            logger.info("no_match_rule : {}".format(no_match_rule))
            segmentation_output[rule_col].fillna(no_match_rule, inplace=True)
        else:
            logger.warning(
                "No fallback rule found in Rules, kindly configure one fallback rule with NO MATCH in all 6 attributes ..."
            )

        logger.info("Merging on rule_df to obtain algo list ...")

        # logger.info("segmentation_output : \n{}".format(segmentation_output))
        # logger.info("rule_df[[rule_col, algo_col]] : \n{}".format(rule_df[[rule_col, algo_col]]))

        # join on rule df to get algo list
        segmentation_output = segmentation_output.merge(
            rule_df[[rule_col, algo_col]], on=rule_col, how="left"
        )
    except Exception as e:
        logger.exception(
            "Exception : {} while assigning forecast rules ...".format(e)
        )

    return segmentation_output
