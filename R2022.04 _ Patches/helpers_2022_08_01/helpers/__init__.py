__version__ = "2022.08.01"

import logging

logger = logging.getLogger("o9_logger")

logger.info(
    "Importing module : {}, version : {}".format(__name__, __version__)
)
