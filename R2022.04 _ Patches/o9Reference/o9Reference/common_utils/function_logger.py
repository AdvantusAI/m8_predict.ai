import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("o9_logger")
import pandas as pd

from functools import wraps


def log_inputs_and_outputs(fn):
    """
    Decorator to print inputs and outputs from a function.
    :param fn:
    :return:
    """

    @wraps(fn)
    def func(*args, **kwargs):
        function_name = fn.__name__

        # print args if not empty
        if args:
            logger.info("With args : {}".format(args))

        logger.info("------ Arguments -------")
        # for every kwarg, print key and value, print head if it's a dataframe
        for key, value in kwargs.items():
            if type(value) == pd.DataFrame:
                logger.info("------ {} ------".format(key))
                __log_df_summary(value)
            else:
                logger.info("--- {} : {}".format(key, value))

        result = fn(*args, **kwargs)

        logger.info(
            "------ Outputs from {} function ----".format(function_name)
        )
        if type(result) == tuple:
            for the_item in result:
                if type(the_item) == pd.DataFrame:
                    __log_df_summary(df=the_item)
        elif type(result) == pd.DataFrame:
            __log_df_summary(df=result)

        return result

    return func


def __log_df_summary(df: pd.DataFrame) -> None:
    """
    Prints the dataframe summary
    :param df:
    :return:
    """
    logger.info("-------df head -------")
    logger.info(df.head())
    logger.info("--- shape : {}".format(df.shape))
    logger.info("--- null count ----")
    logger.info(df.isnull().sum())


@log_inputs_and_outputs
def my_test_function(param1, param2, df):
    test1 = pd.DataFrame()
    test2 = pd.DataFrame()
    return test1, test2


if __name__ == "__main__":
    # my_test_function('hancel', 'rahul', pd.DataFrame())
    my_test_function(param1="hancel", param2="rahul", df=pd.DataFrame())
