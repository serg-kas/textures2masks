"""
Модуль работы с настройками
"""
import os
import json
from dotenv import load_dotenv

# При наличии файла .env загружаем из него переменные окружения
dotenv_path = os.path.join(os.path.dirname(__file__), 'cfg.env')
if os.path.exists(dotenv_path):
    print("Загружаем переменные окружения из файла: {}".format(dotenv_path))
    load_dotenv(dotenv_path)


def get_variable_name(variable):
    """
    Возвращает имя переменной как строку.
    Если переменная по имени не найдена, возвращает None

    :param variable: переменная
    :return: имя переменной
    """
    for name in globals():
        if id(globals()[name]) == id(variable):
            return name
    for name in locals():
        if id(locals()[name]) == id(variable):
            return name
    return None


def get_value_from_env(variable, default_value=None, prefix_name="APP_", verbose=True):
    """
    Ищет значение в переменных окружения.
    Если параметр variable - переменная и есть соответствующее значение в переменных окружения,
    то возвращает это значение. Если значения нет, то возвращает значение переменной variable.

    Если variable - имя переменной (строка) и есть соответствующее значение
    в переменных окружения, то возвращает это значение. Если значения нет, то возвращает default_value

    :param variable: существующая переменная или имя переменной (строка)
    :param default_value: значение по умолчанию
    :param prefix_name: префикс прибавляется к имени переменной
    :param verbose: выводить подробные сообщения
    :return: значение переменной
    """
    variable_name = get_variable_name(variable)
    if variable_name != 'variable':
        got_variable = True
    else:
        got_variable = False
        variable_name = variable

    # Переменная ищется с префиксом в верхнем регистре
    value = os.getenv(prefix_name + variable_name.upper())
    if value is not None:
        if type(default_value) is bool:
            value = bool(value)
        elif type(default_value) is int:
            value = int(value)
        elif type(default_value) is float:
            value = float(value)
        elif type(default_value) is list:
            try:
                value = json.loads(value)
            except ValueError as e:
                # При неудачном преобразовании в json останется тип str
                if verbose:
                    print("  Ошибка: {}".format(e))
                print("  Не удалось прочитать как список: {}".format(value))
        #
        if verbose:
            print("  Получили значение из переменной окружения: {}={}".format(variable_name, value))
            # print(variable_name, value, type(value)))
        return value
    else:
        if got_variable:
            if verbose:
                print("  Не найдено значения переменной {} в переменных окружения, "
                      "оставлено без изменения: {}".format(variable_name, variable))
            return variable
        else:
            if verbose:
                print("  Не найдено значения переменной {} в переменных окружения, "
                      "по умолчанию: {}".format(variable_name, default_value))
            return default_value


# #############################################################
#                        РЕЖИМЫ РАБОТЫ
# #############################################################
OPERATION_MODE_DICT = {
                       'help':'Справочная информация',
                       'self_test':'Тестовый запуск',
                       'workflow_baseline':'Базовый алгоритм через ресайз к разрешению 1024',
                       'workflow_tiling':'Алгоритм на основе тайлинга - В РАБОТЕ',
                       'new1':'Алгоритм новый (заготовка)'
}
OPERATION_MODE_LIST = list(OPERATION_MODE_DICT.keys())
DEFAULT_MODE = OPERATION_MODE_LIST[0]  # режим работы по умолчанию


# #############################################################
#                       ОБЩИЕ ПАРАМЕТРЫ
# #############################################################
# Флаг вывода подробных сообщений в консоль (уровень logging.DEBUG)
VERBOSE = get_value_from_env("VERBOSE", default_value=False)
CONS_COLUMNS = 0  # ширина консоли (0 - попытаться определить автоматически)

# Папки по умолчанию
SOURCE_PATH = 'source_files'
OUT_PATH = 'out_files'
MODELS_PATH = 'models'

# Допустимые форматы изображений для загрузки в программу
ALLOWED_IMAGES = ['.jpg', '.jpeg', '.png']
# Допустимые форматы файлов для загрузки в программу
ALLOWED_TYPES = ALLOWED_IMAGES + ['.pdf']


# #############################################################
#                    ПАРАМЕТРЫ МОДЕЛЕЙ
# #############################################################
# Параметры модели SAM2
# SAM2_config_file = "sam2_hiera_l.yaml"
# SAM2_checkpoint_file = "models/sam2_hiera_large.pt"
SAM2_config_file = "sam2.1_hiera_l.yaml"
SAM2_checkpoint_file = "models/sam2.1_hiera_large.pt"
SAM2_force_cuda = get_value_from_env("SAM2_FORCE_CUDA", default_value=False)
#
SAM2_iou_threshold = 0.25
SAM2_score_threshold = 0.95


# #############################################################
#           ПАРАМЕТРЫ ОБРАБОТКИ базового алгоритма
# #############################################################
# Фильтр масок в разрешении 1024 по размеру
AUTO_CALCULATE_AREAS = get_value_from_env("AUTO_CALCULATE_AREAS", default_value=True)
#
AREA_MIN = get_value_from_env("AREA_MIN", default_value=int(1024 * 1024 * 0.01))
AREA_MAX = get_value_from_env("AREA_MAX", default_value=int(1024 * 1024 * 0.80))

# Расщепление точки промта в заданном радиусе
PROMPT_POINT_RADIUS = get_value_from_env("PROMPT_POINT_RADIUS", default_value=0)
PROMPT_POINT_NUMBER = get_value_from_env("PROMPT_POINT_NUMBER", default_value=5)

# Фильтровать точки промпта по цвету
PROMPT_POINT_COLOR_FILTER = get_value_from_env("PROMPT_POINT_COLOR_FILTER", default_value=False)
PROMPT_POINT_COLOR_THRESH = get_value_from_env("PROMPT_POINT_COLOR_THRESH", default_value=20)


# #############################################################
#            ПАРАМЕТРЫ ОБРАБОТКИ алгоритма ТАЙЛИНГА
# #############################################################
# Разбиение на тайлы
TILING_SIZE = 1024
TILING_OVERLAP = get_value_from_env("TILING_OVERLAP", default_value=256)

# Инверсный режим тайлинга (обычный режим - предикт "плиток", инверсный режим - предикт швов)
TILING_INVERSE_MODE = get_value_from_env("TILING_INVERSE_MODE", default_value=False)

# Сколько точек брать вдоль контуров в тайле при подготовке промпта
TILING_PROMPTS_NUM_POINTS = get_value_from_env("TILING_PROMPTS_NUM_POINTS", default_value=1000)

# Построцессинг (удаление шума и заливка мелких деталей)
TILING_POST_PROCESS = get_value_from_env("TILING_POST_PROCESS", default_value=False)
TILING_POST_PROCESS_KERNEL = get_value_from_env("TILING_POST_PROCESS_KERNEL", default_value=9)


# #############################################################
#                           ПРОЧЕЕ
# #############################################################

# ####################### Цвета RGB #######################
black = (0, 0, 0)
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
yellow = (0, 255, 255)
purple = (255, 0, 255)
turquoise = (255, 255, 0)
white = (255, 255, 255)

# ################### Цвета для консоли ###################
BLACK_cons = '\033[30m'
RED_cons = '\033[31m'
GREEN_cons = '\033[32m'
YELLOW_cons = '\033[33m'
BLUE_cons = '\033[34m'
MAGENTA_cons = '\033[35m'
CYAN_cons = '\033[36m'
WHITE_cons = '\033[37m'
UNDERLINE_cons = '\033[4m'
RESET_cons = '\033[0m'
#
CR_CLEAR_cons = '\r\033[K'
