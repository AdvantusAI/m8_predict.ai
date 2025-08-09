"""
Version : 1.0.0
Maintained by : dpref@o9solutions.com
"""
import logging

from o9Reference.common_utils.function_timer import timed

logging.basicConfig(level=logging.INFO)
from typing import List

logger = logging.getLogger("o9_logger")
import numpy as np
import pandas as pd


@timed
def assign_segments(
    data: np.ndarray, thresholds: list, segments: list
) -> List[str]:
    """
    Assign segments based on thresholds and data provided.
    """
    assert isinstance(data, np.ndarray), "data should be np array type ..."
    assert isinstance(thresholds, list), "thresholds should be list type ..."
    assert isinstance(segments, list), "segments should be list type ..."

    assert len(thresholds) > 0, "thresholds cannot be empty ..."
    assert len(segments) > 0, "segments cannot be empty ..."
    assert None not in data, "Input cannot contain None ..."
    assert not np.isnan(data).any(), "Input cannot contain NaNs ..."
    assert (
        len(segments) == len(thresholds) + 1
    ), "Need n thresholds for n+1 segments ..."
    assert thresholds == sorted(
        thresholds
    ), "thresholds should be passed in the ascending order ..."

    logger.info("Assigning segments ...")

    # Return empty list if input is empty
    if data.size == 0:
        return []

    # if duplicates are present in tresholds, remove them and remove the segment
    de_duplicated_thresholds = []
    for idx, the_threshold in enumerate(thresholds):
        if the_threshold not in de_duplicated_thresholds:
            de_duplicated_thresholds.append(the_threshold)
        else:
            logger.warning(
                "Deleting duplicate segment {} from segments ...".format(
                    segments[idx]
                )
            )
            del segments[idx]

    # insert very large min and max bound for thresholds - for cut function to work
    de_duplicated_thresholds.insert(0, np.finfo(np.float32).min)
    de_duplicated_thresholds.append(np.finfo(np.float32).max)

    col_name = "value"
    df = pd.DataFrame(data, columns=[col_name])

    segment_col = "segment"
    df[segment_col] = pd.cut(
        df[col_name].to_numpy(),
        bins=de_duplicated_thresholds,
        labels=segments,
        include_lowest=True,
    )

    assert len(list(df[segment_col])) == len(
        data
    ), "Length of result list different from input ..."
    return list(df[segment_col])


if __name__ == "__main__":

    # result = assign_segments(
    #     np.array([0.5, 0.6, 0.7, 0.1, 0.7, 0.9]),
    #     thresholds=[0.5, 0.5, 0.8],
    #     segments=["a", "b", "c", "d"],
    # )
    # print(result)

    # creating a test dataset
    Data = pd.DataFrame(
        {
            "Item": [
                "Item1",
                "Item2",
                "Item3",
                "Item4",
                "Item5",
                "Item6",
                "Item7",
                "Item8",
                "Item9",
                "Item10",
                "Item11",
                "Item12",
            ],
            "Actual": [
                300,
                100,
                425,
                75,
                777,
                243,
                896,
                172,
                1108,
                673,
                99,
                101,
            ],
        }
    )
    Thresholds = [0.8, 0.95]
    SegmentNames = ["A", "B", "C"]
    # Determine the segments
    Data["Actual %"] = Data["Actual"] / Data["Actual"].sum()
    Data.sort_values("Actual %", ascending=False, inplace=True)
    Data["Actual % Cumulative"] = Data["Actual %"].cumsum()
    Data["Segment"] = assign_segments(
        Data["Actual % Cumulative"].to_numpy(),
        Thresholds,
        SegmentNames,
    )
    print(Data)
