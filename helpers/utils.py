"""
Функции различного назначения
"""
import numpy as np
import math
# import re
#
import cv2 as cv
# from PIL import Image  # ImageDraw, ImageFont
# from imutils import perspective, auto_canny
# import matplotlib.pyplot as plt
#
# import io
import os
# import sys
import inspect
#
import settings as s
from settings import white


# #############################################################
#                       ФУНКЦИИ OpenCV
# #############################################################
def autocontrast_cv(img, clip_limit=2.0, called_from=False):
    """
    Функция автокоррекции контраста методом
    CLAHE Histogram Equalization через цветовую модель LAB

    :param img: изображение
    :param clip_limit: порог используется для ограничения контрастности
    :param called_from: сообщать откуда вызвана функция
    :return: обработанное изображение
    """
    #
    if called_from:
        # текущий фрейм объект
        current_frame = inspect.currentframe()
        # фрейм объект, который его вызвал
        caller_frame = current_frame.f_back
        # у вызвавшего фрейма исполняемый в нём объект типа "код"
        code_obj = caller_frame.f_code
        # и получи его имя
        code_obj_name = code_obj.co_name
        print("Функция autocontrast_cv вызвана из:", code_obj_name)

    # converting to LAB color space
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l_channel, a, b = cv.split(lab)

    # Applying CLAHE to L-channel
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color space
    result = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
    return result


def sharpen_image_cv(img, force_threshold=False):
    """
    Увеличивает резкость изображения
    с переходом в ч/б изображение и тресхолдом опционально.
    Использует фильтр (ядро) для улучшения четкости

    :param img: изображение
    :param force_threshold: тресхолд выходного изображение
    :return: обработанное изображение
    """
    img = img.copy()

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])

    img = cv.filter2D(img, -1, sharpen_kernel)

    if force_threshold:
        img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

    # show_image_cv(img, title="img sharpened")
    return img


def invert_image(image):
    """
    Инверсия изображения
    """
    inverted = cv.bitwise_not(image)
    return inverted


def img_resize_cv(image, img_size=1024):
    """
    Функция ресайза картинки через opencv к заданному размеру по наибольшей оси

    :param image: исходное изображение
    :param img_size: размер, к которому приводить изображение
    :return: resized image
    """
    curr_h = image.shape[0]
    curr_w = image.shape[1]
    # Рассчитаем коэффициент для изменения размера
    if curr_w > curr_h:
        scale_img = img_size / curr_w
    else:
        scale_img = img_size / curr_h
    # Новые размеры изображения
    new_width = int(curr_w * scale_img)
    new_height = int(curr_h * scale_img)
    # делаем ресайз к целевым размерам
    image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
    return image


def image_rotate_cv(image, angle, simple_way=False, resize_to_original=False):
    """
    Функция вращения картинки средствами opencv

    :param image: изображение
    :param angle: угол в градусах
    :param simple_way: упрощенный вариант с обрезкой углов повернутого изображения
    :param resize_to_original: ресайз результата к размерам исходного изображения при simple_way==False
    :return:
    """
    height, width = image.shape[:2]

    # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    image_center = (width / 2, height / 2)

    rotation_mat = cv.getRotationMatrix2D(image_center, angle, 1.)

    if simple_way:
        # rotate image and left border white
        result = cv.warpAffine(image,
                               rotation_mat,
                               (width, height),
                               flags=cv.INTER_LINEAR,
                               borderMode = cv.BORDER_CONSTANT,
                               borderValue = white)
        return result

    else:
        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to orig) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix and left border white
        result = cv.warpAffine(image,
                               rotation_mat,
                               (bound_w, bound_h),
                               flags=cv.INTER_LINEAR,
                               borderMode=cv.BORDER_CONSTANT,
                               borderValue=white)

        #
        if resize_to_original:
            result = cv.resize(result, (width, height), interpolation=cv.INTER_AREA)
            return result
        else:
            return result


def show_image_cv(img, title='Image '):
    """
    Выводит картинку на экран методом из opencv.
    Дожидается нажатия клавиши для продолжения

    :param img: изображение
    :param title: заголовок окна
    :return: none
    """
    if title == 'Image ':
        cv.imshow('Image ' + str(img.shape), img)
    else:
        cv.imshow(title, img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def get_optimal_font_scale_cv(text, width):
    """
    Подбор оптимального размера шрифта для метода cv.putText

    :param text: текст для подбора шрифта
    :param width: нужная ширина
    :return: пододранные fontScale
    """
    for scale in reversed(range(0, 60, 1)):
        textSize = cv.getTextSize(text, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=scale/10, thickness=3)
        new_width = textSize[0][0]
        if new_width <= width:
            return scale/10
    return 1


def fig2img_cv(fig):
    """
    Convert a Matplotlib figure to a CV Image and return it
    """
    import io
    buf = io.BytesIO()
    fig.savefig(buf, dpi=600)
    buf.seek(0)
    array = np.asarray(bytearray(buf.read()), dtype=np.uint8)
    img = cv.imdecode(array, cv.IMREAD_COLOR)
    # img = Image.open(buf)  # если нужно изображение PIL
    return img


def get_img_encoded_list(img_list, extension=".png"):
    """
    Создание списка изображений png в памяти

    :param img_list: список изображений
    :param extension: тип изображения по расширению
    :return: png_list
    """
    encoded_list = []
    for img in img_list:
        img_encoded = cv.imencode(extension, img)[1]
        # encoded_list.append(img_encoded)
        encoded_list.append(img_encoded.tobytes())
    return encoded_list


def get_blank_img_cv(height, width, rgb_color=(0, 0, 0), txt_to_put=None):
    """
    Возвращает изображение заданной размерности, цвета, с заданной надписью

    :param height: высота
    :param width: ширина
    :param rgb_color: цвет RGB
    :param txt_to_put: текст
    :return: изображение
    """
    blank_img = np.zeros((height, width, 3), dtype=np.uint8)
    bgr_color = tuple(reversed(rgb_color))
    blank_img[:] = bgr_color
    #
    if txt_to_put is not None:
        X = int(width * 0.1)
        Y = int(height * 0.1)
        font_size = get_optimal_font_scale_cv(txt_to_put, int(width * 0.9))
        cv.putText(blank_img, txt_to_put, (X, Y), cv.FONT_HERSHEY_SIMPLEX, font_size, s.black, 2, cv.LINE_AA)
    return blank_img


def hsv_color_filter_cv(image, light_hsv=(93, 75, 74), dark_hsv=(151, 255, 255)):
    """
    Фильтрация цвета в диапазоне цветов HSV
    Параметры по умолчанию фильтруют синий цвет (заменяет белым)

    :param image:
    :param light_hsv:
    :param dark_hsv:
    :return: обработанное изображение
    """
    #
    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask = cv.inRange(image_hsv, light_hsv, dark_hsv)
    #
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.erode(mask, kernel, iterations=1)
    mask = cv.dilate(mask, kernel, iterations=2)
    #
    image[:, :, 0] = np.where(mask == 0, image[:, :, 0], 255)
    image[:, :, 1] = np.where(mask == 0, image[:, :, 1], 255)
    image[:, :, 2] = np.where(mask == 0, image[:, :, 2], 255)

    return image


def match_template_cv(img, template, templ_threshold=0.5, templ_metric=cv.TM_CCOEFF_NORMED):
    """
    Поиск шаблона на изображении методом opencv

    :param img: изображение
    :param template: шаблон
    :param templ_threshold: тресхолд
    :param templ_metric: метрика (параметр opencv)
    :return: изображение с найденными шаблонами, список bb (координат привязки шаблонов)
    """
    img_parsed = img.copy()
    #
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h, w = template.shape[:2]
    #
    res = cv.matchTemplate(gray, template, templ_metric)
    loc = np.where(res >= templ_threshold)

    pt_list = []
    for pt in zip(*loc[::-1]):
        pt_list.append((pt[0], pt[1], pt[0] + w, pt[1] + h))
        cv.rectangle(img_parsed, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

    print("  Найдено привязок темплейта (bb) на изображении: {}".format(len(pt_list)))
    return img_parsed, pt_list


def image_is_close_to_white_cv(img, threshold=200, white_ratio=0.9):
    """
    Проверка близости изображения к белому цвету

    :param img: изображение для обработки
    :param threshold:
    :param white_ratio: доля белых пикселей на тресхолде
    :return: True если изображение близко белому, доля белых пикселей
    """
    image = img.copy()

    # Преобразование изображения в формат HSV
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Определение границ для белого цвета в формате HSV
    lower_white = np.array([0, 0, threshold], dtype=np.uint8)
    upper_white = np.array([180, 55, 255], dtype=np.uint8)

    # Создание маски для белых областей
    white_mask = cv.inRange(hsv_image, lower_white, upper_white)

    # Подсчет количества белых пикселей
    white_pixels = cv.countNonZero(white_mask)

    # Подсчет общего количества пикселей
    total_pixels = image.shape[0] * image.shape[1]

    # Вычисление доли белых пикселей
    white_ratio_actual = white_pixels / total_pixels

    # Проверка, превышает ли доля белых пикселей заданный порог
    return white_ratio_actual >= white_ratio, white_ratio_actual


# #############################################################
#                    ФУНКЦИИ МАТЕМАТИЧЕСКИЕ
# #############################################################
def get_distance_pp(p1, p2):
    """
    Вычисляет расстояние между двумя точками на плоскости

    :param p1: точка на плоскости tuple (x1, y1)
    :param p2: точка на плоскости tuple (x2, y2)
    :return: расстояние
    """
    distance = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return distance


def scale_to_01_range(x):
    """
    Scale and move the coordinates, so they fit [0; 1] range

    :param x: numpy array to process
    :return:
    """
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def rel_equal(x, y, e):
    """
    Возвращает истина, если числа совпадают с указанной относительной точностью

    :param x: число для сравнения
    :param y: числа для сравнения
    :param e: относительная точность
    :return: boolean
    """
    # print("rel_equality", x, y, e, "abs(abs(x / y) - 1) < e :", abs(abs(x / y) - 1) < e,
    #       "math.isclose(x, y, rel_tol=e) :", math.isclose(x, y, rel_tol=e))
    return math.isclose(x, y, rel_tol=e)


# #############################################################
#                      ФУНКЦИИ ПРОЧИЕ
# #############################################################
def always_true(*args, **kwargs):
    """
    Возвращает True при любых поданных аргументах
    """
    return True


def clean_folder(path):
    """
    Очистка папки с удалением только файлов внутри нее

    :param path: путь к папке
    """
    files_to_remove = []
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            files_to_remove.append(os.path.join(path, file))
    for file in files_to_remove:
        os.remove(file)


def get_files_by_type(source_files, type_list=None):
    """
    Получает список файлов и список типов (расширений)

    :param source_files: список файлов
    :param type_list: список типов (расширений файлов) или одно расширение как строка
    :return: список файлов, отобранный по типу (расширениям)
    """
    if type_list is None:
        type_list = []
    elif type(type_list) is str:
        type_list = [type_list]

    result_file_list = []
    for f in source_files:
        _, file_extension = os.path.splitext(f)
        if file_extension in type_list:
            result_file_list.append(f)
    return result_file_list


def txt_separator(string, n=80, txt='', txt_align='center'):
    """
    Формирует строку из символов и текстовой надписи.
    Текст накладывается в начале, в середине или в конце строки

    :param string: строка или символ, повторением которых формируется разделитель
    :param n: количество символов в стоке (если 0, то попробовать подогнать по консоли)
    :param txt: текст для печати (если слишком длинный, то обрезается)
    :param txt_align: как выравнивать текст в строке (left, center, right)
    :return:
    """
    assert type(string) != 'str', "Ожидается тип данных 'str'"
    n = 0 if n < 0 else n      # отрицательное n не имеет смысла
    n = 128 if n > 128 else n  # слишком большое n не имеет смысла

    if n == 0:  # пробуем получить ширину консоли
        stty_size = os.popen('stty size', 'r').read().split()
        if len(stty_size) == 2:
            n = int(stty_size[1])
        else:  # если не получилось присваиваем стандартную ширину
            n = 80

    # Выходная строка без текста
    if txt == '':
        k = n // len(string) + 1  # сколько раз брать строку
        out_string = string * k   # выходная строка
        out_string = out_string[:n] if len(out_string) > n else out_string  # ограничиваем длину строки параметром n
        return out_string
    # Выходная строка с наложением текста
    else:
        if len(txt) == n:  # если текст длинной n, то возвращаем текст
            return txt
        elif len(txt) > n:  # если текст слишком длинный, то возвращаем текст, обрезая до длинны n
            txt = txt[:n]
            return txt
        else:
            k = n // len(string) + 1  # сколько раз брать строку
            out_string = string * k   # выходная строка
            out_string = out_string[:n] if len(out_string) > n else out_string  # ограничиваем длину строки параметром n
            if txt_align == 'left':
                out_string = txt + out_string[len(txt):]
                return out_string
            elif txt_align == 'right':
                out_string = out_string[:-len(txt)] + txt
                return out_string
            elif txt_align == 'center':
                start_txt_pos = (n - len(txt)) // 2
                out_string = out_string[:start_txt_pos] + txt + out_string[-start_txt_pos:]
                return out_string


# #############################################################
#                  Функции ...
# #############################################################
