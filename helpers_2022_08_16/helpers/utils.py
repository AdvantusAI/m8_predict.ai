"""
Version : 2022.08.16
Maintained by : pmm_algocoe@o9solutions.com
"""
import logging

logger = logging.getLogger("o9_logger")


def get_measure_name(key: str):
    return "Stat Fcst " + key


def get_model_desc_name(key: str):
    return key + " Model"


def get_bound_col_name(measure_name, alpha, direction):
    # assume alpha = 0.05, we want to publish 95 into interval
    interval = int(100 - (alpha * 100))
    return measure_name + " {}% {}".format(interval, direction)


def get_ts_freq_prophet(seasonal_periods):
    # Assign default value of 12
    ts_freq = "M"  # Month Start
    if seasonal_periods == 52:
        ts_freq = "W"  # Weekly
    elif seasonal_periods == 4:
        ts_freq = "Q"  # Quarter Start
    return ts_freq


def get_seasonal_periods(frequency: str) -> int:
    """
    Returns num seasonal periods based on string
    """
    if frequency == "Weekly":
        return 52
    elif frequency == "Monthly":
        return 12
    elif frequency == "Quarterly":
        return 4
    else:
        raise ValueError("Unknown frequency {}".format(frequency))
