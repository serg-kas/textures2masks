"""
Модуль логирования.
"""
import logging
from datetime import datetime
from pytz import timezone
#
DEFAULT_LOGGER_NAME = 'ballot_ocr'
#
DEFAULT_LOGGER_FORMATTER = '%(asctime)s %(name)s %(levelname)s: %(message)s'
DEFAULT_DATE_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
#
DEFAULT_TIMEZONE = 'Europe/Moscow'
#
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL
#
ERROR_LEVELS = {'DEBUG':DEBUG,
                'INFO':INFO,
                'WARNING':WARNING,
                'ERROR':ERROR,
                'CRITICAL':CRITICAL}


# Создаем кастомный форматтер
class CustomFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created)
        dt = dt.astimezone(timezone(DEFAULT_TIMEZONE))
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            s = dt.isoformat(timespec='milliseconds')
        return s


def get_logger(name=DEFAULT_LOGGER_NAME,
               level=INFO,
               formatter=DEFAULT_LOGGER_FORMATTER,
               dt_format=DEFAULT_DATE_TIME_FORMAT,
               prevent_propagate=True):
    """
    Функция инициализации логгера

    :param name: имя логгера
    :param level: уровень логирования
    :param formatter: формат сообщения логгера
    :param dt_format: формат времени логгера
    :param prevent_propagate: отключить возможное наследование обработчика
    :return: logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    #
    ch = logging.StreamHandler()

    # ch.setFormatter(logging.Formatter(fmt=formatter, datefmt=dt_format))
    ch.setFormatter(CustomFormatter(fmt=formatter, datefmt=dt_format))

    logger.addHandler(ch)
    #
    if prevent_propagate:
        logger.propagate = False

    return logger


def check_if_logger_exists(logger_name):
    """
    Проверяет существование логгера
    """
    loggers = logging.getLogger().manager.loggerDict
    return logger_name in loggers


def set_logger_level(logger_name, level_name):
    """
    Устанавливает уровень сообщений логгера

    :param logger_name: имя логгера
    :param level_name: уровень логирования
    :return: True если операция успешна, False если такого логгера нет
    """
    assert level_name in ERROR_LEVELS.keys(), "Некорректное название уровня логгирования"

    if check_if_logger_exists(logger_name):
        logger = logging.getLogger(logger_name)
        logger.setLevel(ERROR_LEVELS[level_name])
        return True
    else:
        return False


def get_logger_level(logger_name):
    """
    Возвращает уровень логгирования по имени логгера

    :param logger_name: имя логгера
    :return: уровень логгирования, если логгер не найден возвращает 'WRONG LOGGER NAME'
    """
    if check_if_logger_exists(logger_name):
        logger = logging.getLogger(logger_name)
        level = logger.level
        if level == CRITICAL:
            return "CRITICAL"
        elif level == ERROR:
            return "ERROR"
        elif level == WARNING:
            return "WARNING"
        elif level == INFO:
            return "INFO"
        elif level == DEBUG:
            return "DEBUG"
        else:
            return "NOTSET"
    else:
        return None


def get_logger_func(logger=None):
    """
    Возвращает универсальную функцию логирования в зависимости от типа logger.
    TODO: работает только с методом 'info'

    :param logger:
        - None: возвращает "пустую" функцию (ничего не делает)
        - print: возвращает функцию print
        - объект логгера: возвращает функцию, вызывающую logger.info()
    :param logger:
        = level: уровень логгирования
    :return: callable функция логирования
    """
    # Пустая функция заменяет отсутствующий логгер
    if logger is None:
        def log_func(*args, **kwargs):
            pass
        #
        return log_func

    elif logger is print:
        # Возвращаем саму функцию print
        return print

    elif hasattr(logger, 'info') and callable(logger.info):
        # Создаем адаптер для логгера
        def log_func(*args, sep=' ', end='\n', **kwargs):
            # Формируем сообщение как в print
            # message = sep.join(str(arg) for arg in args) + end
            message = sep.join(str(arg) for arg in args)
            if message.strip():  # Игнорируем сообщения из пробелов
                logger.info(message, **kwargs)
        #
        return log_func

    else:
        raise TypeError("Неподдерживаемый тип логгера. "
                        "Используйте None, print или объект логгера с методом info")
