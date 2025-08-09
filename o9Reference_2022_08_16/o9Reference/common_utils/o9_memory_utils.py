"""
Version : 1.0.0
Maintained by : dpref@o9solutions.com
"""
"""
Background thread to measure memory usage :

from o9Reference.common_utils.o9_memory_utils import _get_memory
import logging
logger = logging.getLogger("o9_logger")

logger.info("Starting thread to measure memory usage ...")
# Start a thread to print memory occasionally
back_thread = threading.Thread(
    target=_get_memory, kwargs=dict(max_memory=0.0), daemon=True
)

"""
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("o9_logger")
import time
import psutil
import os


# Memory profiling code begins
def memory_instance():
    process = psutil.Process(os.getpid())
    info = process.memory_info()
    return info.rss / 1000000


def get_max_memory(curr_mem, max_memory_used):
    max_memory_used = (
        max_memory_used if max_memory_used > curr_mem else curr_mem
    )
    return max_memory_used


# Prints the memory profile every n seconds
def _get_memory(max_memory, sleep_seconds=60, df_keys={}):
    try:
        while True:
            mem_used = memory_instance()
            max_memory = get_max_memory(mem_used, max_memory)

            if bool(df_keys):
                logger.info(
                    "------------- Used memory : {} M for slice {}".format(
                        mem_used, df_keys
                    )
                )
                logger.info(
                    "------------- Max memory usage : {} M for slice {}".format(
                        max_memory, df_keys
                    )
                )
            else:
                logger.info(
                    "------------- Used memory : {} M".format(mem_used)
                )
                logger.info(
                    "------------- Max memory usage : {} M".format(max_memory)
                )

            time.sleep(sleep_seconds)
    except Exception as e:
        logger.error("failed run get memory daemon ...")
        logger.exception(e)


if __name__ == "__main__":

    # df_keys = {}
    df_keys = {"Item": "50"}
    import threading

    # Start a thread to print memory occasionally, change sleep seconds if required,
    # Since thread is daemon, it's closed automatically with main script.
    back_thread = threading.Thread(
        target=_get_memory,
        kwargs=dict(max_memory=0.0, sleep_seconds=90, df_keys=df_keys),
        daemon=True,
    )
    logger.info("Starting background thread for memory profiling ...")
    back_thread.start()
    logger.info("sleeping...")
    time.sleep(30)
    logger.info("woke up...")
