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
def _get_memory(max_memory, sleep_seconds=60):
    try:
        while True:
            mem_used = memory_instance()
            max_memory = get_max_memory(mem_used, max_memory)
            logger.info("------------- Used memory : {} M".format(mem_used))
            logger.info(
                "------------- Max memory usage : {} M".format(max_memory)
            )
            time.sleep(sleep_seconds)
    except Exception as e:
        logger.error("failed run get memory daemon ...")
        logger.exception(e)
