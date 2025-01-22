"""
Модуль логирования.
"""
import logging
from datetime import datetime
from pytz import timezone
#
DEFAULT_LOGGER_NAME = ''
#
DEFAULT_LOGGER_FORMATTER = '%(asctime)s %(name)s %(levelname)s: %(message)s'
DEFAULT_DATE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
#
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


# Создаем кастомный форматтер
class CustomFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)
        dt = dt.astimezone(timezone('Europe/Moscow'))
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            s = dt.isoformat(timespec='milliseconds')
        return s


def get_logger(name=DEFAULT_LOGGER_NAME,
               level=INFO,
               formatter=DEFAULT_LOGGER_FORMATTER,
               dt_format=DEFAULT_DATE_TIME_FORMAT):
    """
    Функция инициализации логгера

    :param name: имя логгера
    :param level: уровень логирования
    :param formatter: формат сообщения логгера
    :param dt_format: формат времени логгера
    :return: logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    #
    ch = logging.StreamHandler()

    # ch.setFormatter(logging.Formatter(fmt=formatter, datefmt=dt_format))
    ch.setFormatter(CustomFormatter(fmt=formatter, datefmt=dt_format))

    logger.addHandler(ch)

    return logger
