import sys, os
import pip
import config
import platform
from datetime import datetime

#additional functions
# Error log
error_log = ""


def error(message):
    """
    Print error message and save it in error log.
    Error log will be sent at the end of execution.
    :param message: error message
    """
    log = config.log
    log.error("ERROR: " + message)

    global error_log
    error_log += "[{0}] ERROR OCCURRED:\n{1}\n\n\n".format(datetime.now().strftime("%H:%M:%S"), message)
