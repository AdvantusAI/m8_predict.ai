import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("o9_logger")
import pandas as pd

pd.options.display.max_rows = 25
pd.options.display.max_columns = 50
pd.options.display.max_colwidth = 100
pd.set_option("display.width", 1000)
pd.options.display.precision = 3

from functools import wraps


def log_inputs_and_outputs(fn):
    """
    Decorator to print inputs and outputs from a function.
    :param fn:
    :return:
    """

    @wraps(fn)
    def func(*args, **kwargs):
        # get function name
        function_name = fn.__name__

        logger.info(
            "--- Invoking {} with below arguments (excluding dataframes) ---".format(
                function_name
            )
        )

        # print args if not empty
        if args:
            for the_arg in args:
                # dataframe stats (shape, dtypes) are already available in PythonServer logs
                if type(the_arg) != pd.DataFrame:
                    logger.info("------ {} ------".format(the_arg))

        # for every kwarg, print key and value
        for key, value in kwargs.items():
            if type(value) != pd.DataFrame:
                logger.info("------ {} : {}".format(key, value))

        logger.info("--------------------------")

        # invoke function, get result
        result = fn(*args, **kwargs)

        logger.info(
            "------ Outputs from {} function ----".format(function_name)
        )

        # outputs could be multiple (will be a tuple)
        if type(result) == tuple:
            output_count = len(result)
            # specify start index 1 with enumerate
            for idx, the_item in enumerate(result, 1):
                logger.info("------ Output {} of {}".format(idx, output_count))
                __log_result(the_item)
        # else log the single output dataframe
        else:
            __log_result(result)

        logger.info("------ End of logs ------")

        return result

    return func


def __log_result(the_result):
    logger.info("-------- type : {} ------".format(type(the_result)))
    if type(the_result) == pd.DataFrame:
        __log_df_summary(df=the_result)
    else:
        logger.info("-------- {} ------".format(the_result))


def __log_df_summary(df: pd.DataFrame) -> None:
    """
    Prints the dataframe summary
    :param df:
    :return:
    """
    if len(df) == 0 or df.empty:
        logger.warning("-------- Empty Dataframe ------")
    else:
        logger.info("-------- dataframe head \n{}".format(df.head()))
        logger.info("-------- dtypes \n{}".format(df.dtypes))
        logger.info("-------- shape : {}".format(df.shape))
        logger.info("-------- null count \n{}".format(df.isnull().sum()))


@log_inputs_and_outputs
def main(param1, param2, df):
    test1 = pd.DataFrame(
        np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD")
    )
    # test2 = pd.DataFrame()
    x = "100"
    return test1, x


if __name__ == "__main__":
    # my_test_function('hancel', 'rahul', pd.DataFrame())
    import numpy as np

    df1 = pd.DataFrame(
        np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD")
    )

    main("hancel", "rahul", df1)
    # my_test_function(param1="hancel", param2="rahul", df=pd.DataFrame())
