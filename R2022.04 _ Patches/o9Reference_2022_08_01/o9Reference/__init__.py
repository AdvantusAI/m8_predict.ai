__version__ = "2022.08.01"
# Change version from 1.0.0 to release version
import logging

logger = logging.getLogger("o9_logger")

logger.info(
    "Importing module : {}, version : {}".format(__name__, __version__)
)
