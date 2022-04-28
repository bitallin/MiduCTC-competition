import logging
import os
import re
from logging.handlers import TimedRotatingFileHandler


def setup_log(log_name):
    logger = logging.getLogger(log_name)
    log_path = os.path.join("logs", log_name)
    logger.setLevel(logging.DEBUG)
    file_handler = TimedRotatingFileHandler(
        filename=log_path, when="MIDNIGHT", interval=1, backupCount=30
    )
    file_handler.suffix = "%Y-%m-%d.log"
    file_handler.extMatch = re.compile(r"^\d{4}-\d{2}-\d{2}.log$")
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(process)d] [%(levelname)s] - %(module)s.%(funcName)s (%(filename)s:%(lineno)d) - %(message)s"
    )

    stream_handler.setFormatter(file_handler)
    file_handler.setFormatter(
        formatter
    )
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


logger = setup_log("ctc.log")
