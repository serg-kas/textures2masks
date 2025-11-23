"""
Функции различного назначения
"""
import numpy as np
import math
import re
# from sklearn.cluster import KMeans
# from sklearn.mixture import GaussianMixture
# from scipy import stats
# import warnings

import cv2 as cv
from PIL import Image  # ImageDraw, ImageFont
# from imutils import perspective, auto_canny
import matplotlib.pyplot as plt

import io
import json
import base64
import os
# import sys
# import time
from datetime import datetime, timezone, timedelta
# import inspect
# import importlib

import settings as s


# #############################################################
#                       ФУНКЦИИ OpenCV
# #############################################################
def clahe_hist_cv(img,
                  clip_limit=2.0):
                  # called_from=False):
    """
    Функция автокоррекции контраста методом
    CLAHE Histogram Equalization через цветовую модель LAB.

    Принимает 3-х канальное изображение

    :param img: изображение
    :param clip_limit: порог используется для ограничения контрастности
    # :param called_from: сообщать откуда вызвана функция
    :return: обработанное изображение
    """
    #
    # if called_from:
    #     # текущий фрейм объект
    #     current_frame = inspect.currentframe()
    #     # фрейм объект, который его вызвал
    #     caller_frame = current_frame.f_back
    #     # у вызвавшего фрейма исполняемый в нём объект типа "код"
    #     code_obj = caller_frame.f_code
    #     # и получи его имя
    #     code_obj_name = code_obj.co_name
    #     print("Функция autocontrast_cv вызвана из:", code_obj_name)

    #
    assert len(img.shape) == 3, "Функция ожидает трехканальное изображения BGR"

    # TODO: предусмотреть работу с ЧБ (один канал)

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


def stretch_hist_cv(image,
                    clip_percent=2.0,
                    keep_dimension=True):
    """
    Растягивание гистограммы.

    Принимает 1-канальное или 3-х канальное изображение.
    Возвращает 1-канальное изображение или изображение канальности равной исходному с опцией keep_dimension

    :param image: изображение
    :param clip_percent: проценты для отсечения
    :param keep_dimension: сохранять количество каналов
    :return: обработанное изображение
    """
    # Проверяем количество каналов
    len_image_shape = len(image.shape)
    # print("Получено изображение: {}".format(image.shape))

    # Переходим к ЧБ
    if len_image_shape == 3:
        img_bw = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        img_bw = image.copy()

    # show_image_cv(img_bw, title=f"source {img_bw.shape}")
    assert len(img_bw.shape) == 2, "Для обработки требуется черно-белое изображение"

    # Рассчитываем гистограмму
    hist = cv.calcHist([img_bw], [0], None, [256], [0, 256])

    total = img_bw.shape[0] * img_bw.shape[1]
    threshold = (clip_percent / 100) * total

    min_val = 0
    max_val = 255
    sum_count = 0

    # Поиск min_val
    for i in range(256):
        sum_count += hist[i][0]
        if sum_count > threshold:
            min_val = i
            break

    # Поиск max_val
    sum_count = 0
    for i in range(255, -1, -1):
        sum_count += hist[i][0]
        if sum_count > threshold:
            max_val = i
            break

    if max_val > min_val:
        scale = 255.0 / (max_val - min_val)

        # Создаем матрицы с минимальным значением и масштабом
        min_val_mat = np.full(img_bw.shape, min_val, dtype=np.float32)
        scale_mat = np.full(img_bw.shape, scale, dtype=np.float32)

        # Конвертируем изображение в float32
        img_bw_float = img_bw.astype(np.float32)

        # Выполняем операции растяжения
        stretched = (img_bw_float - min_val_mat) * scale_mat

        # Обрезаем значения и конвертируем обратно в uint8
        stretched = np.clip(stretched, 0, 255).astype(np.uint8)
    else:
        stretched = img_bw.copy()

    # Сохраняем размерность
    if len_image_shape != 2 and keep_dimension:
        stretched = cv.cvtColor(stretched, cv.COLOR_GRAY2BGR)

    # show_image_cv(stretched, title=f"img stretched {stretched.shape}")
    return stretched


def equalize_hist_cv(image, keep_dimension=True) -> np.ndarray:
    """
    Выравнивание гистограммы методом equalizeHist.

    Не путать с адаптивным выравниванием алгоритмом CLAHE в функции clahe_hist_cv.
    Не путать с линейным растягиванием функцией stretch_hist_cv.
    """
    # Проверяем количество каналов
    len_image_shape = len(image.shape)
    # print("Получено изображение: {}".format(image.shape))

    # Переходим к ЧБ
    if len_image_shape == 3:
        img_bw = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        img_bw = image.copy()

    # show_image_cv(img_bw, title=f"source {img_bw.shape}")
    assert len(img_bw.shape) == 2, "Для обработки требуется черно-белое изображение"

    equalized = cv.equalizeHist(img_bw)

    # Сохраняем размерность
    if len_image_shape == 3 and keep_dimension:
        equalized = cv.cvtColor(equalized, cv.COLOR_GRAY2BGR)

    # show_image_cv(equalized, title=f"img equalized {stretched.shape}")
    return equalized


def adjust_contrast_gamma_image_cv(image,
                                   contrast_factor=1.25,
                                   gamma=0.75):
    """
    Увеличение контраста и уменьшение гаммы изображения.

    Используется cv.convertScaleAbs: каждый пиксель умножается на contrast_factor,
    что линейно усиливает разницу между темными и светлыми областями.

    Gamma < 1 (например, γ = 0.75) осветляет средние тона и тени.
    Gamma > 1 (например, γ = 1.5) затемняет средние тона, усиливает контраст в светлых областях.
    """
    # Повышаем контраст
    image = cv.convertScaleAbs(image, alpha=contrast_factor, beta=0)

    # Изменяем гамму
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    image = cv.LUT(image, lookUpTable)
    return image


def sharpen_image_cv(image,
                     force_threshold=False,
                     keep_dimension=True):
    """
    Увеличивает резкость изображения
    с тресхолдом опционально.
    Использует фильтр (ядро) для улучшения четкости.

    Принимает 1-канальное или 3-х канальное изображение.
    Возвращает 1-канальное изображение или изображение канальности равной исходному с опцией keep_dimension

    :param image: изображение
    :param force_threshold: тресхолд выходного изображение
    :param keep_dimension: сохранять количество каналов
    :return: обработанное изображение
    """
    # Проверяем количество каналов
    len_image_shape = len(image.shape)
    # print("Получено изображение: {}".format(image.shape))

    # Переходим к ЧБ
    if len_image_shape == 3:
        img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        img = image.copy()

    # show_image_cv(img_bw, title=f"source {img_bw.shape}")
    assert len(img.shape) == 2, "Для обработки требуется черно-белое изображение"

    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])

    img = cv.filter2D(img, -1, sharpen_kernel)

    if force_threshold:
        img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]

    # Сохраняем размерность
    if len_image_shape == 3 and keep_dimension:
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    # show_image_cv(img, title="img sharpened")
    return img


def invert_image_cv(image):
    """
    Инверсия изображения
    """
    inverted = cv.bitwise_not(image)
    return inverted


def resize_image_cv(image, img_size=1024, interpolation=cv.INTER_AREA):
    """
    Функция ресайза картинки через opencv к заданному размеру по наибольшей оси

    :param image: исходное изображение
    :param img_size: размер, к которому приводить изображение
    :param interpolation: интерполяция (константа opencv)
    :return: resized image
    """
    # Размеры исходного изображения
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
    image = cv.resize(image, (new_width, new_height), interpolation=interpolation)
    return image


def rotate_image_cv(image, angle, simple_way=False, resize_to_original=False):
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
                               borderValue = s.white)
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
                               borderValue=s.white)

        #
        if resize_to_original:
            result = cv.resize(result, (width, height), interpolation=cv.INTER_AREA)
            return result
        else:
            return result


def color_filter_hsv_cv(image, light_hsv=(93, 75, 74), dark_hsv=(151, 255, 255)):
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


def nlm_denoising_image_cv(
        image,
        h: float = 10,
        h_color: float = 10,
        template_window_size: int = 7,
        search_window_size: int = 21) -> np.ndarray:
    """
    Удаляет шум с изображения с использованием нелокального среднего фильтра (NLM)
    """
    # Определение типа изображения
    is_color = len(image.shape) == 3

    # Применение фильтра
    if is_color:
        denoised = cv.fastNlMeansDenoisingColored(
            src=image,
            h=h,
            hColor=h_color,
            templateWindowSize=template_window_size,
            searchWindowSize=search_window_size
        )
    else:
        denoised = cv.fastNlMeansDenoising(
            src=image,
            h=h,
            templateWindowSize=template_window_size,
            searchWindowSize=search_window_size
        )
    return denoised


def denoising_image_cv(image,
                       is_bgr=True,
                       simple_way=False,
                       keep_dimension=True):
    """
    Комплексная функция удаления шума

    Принимает 1-канальное или 3-х канальное изображение.
    Возвращает 1-канальное изображение или изображение канальности равной исходному с опцией keep_dimension

    :param image: изображение
    :param is_bgr: считать изображение BGR (иначе многоканальное изображение будет считаться RGB)
    :param simple_way: упрощенный вариант обработки
    :param keep_dimension: сохранять количество каналов
    :return: обработанное изображение
    """
    # Проверяем количество каналов
    len_image_shape = len(image.shape)
    # print("Получено изображение: {}".format(image.shape))

    # Переходим к ЧБ
    if len_image_shape == 3:
        if is_bgr:
            img_bw = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            img_bw = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    else:
        img_bw = image.copy()

    # show_image_cv(img_bw, title=f"source {img_bw.shape}")
    assert len(img_bw.shape) == 2, "Для обработки требуется черно-белое изображение"

    if simple_way:
        # Сглаживает шум, сохраняя резкие края QR-кода. Работает на уже частично очищенном изображении.
        img_bw = cv.bilateralFilter(img_bw, d=9, sigmaColor=75, sigmaSpace=75)

        # Легкое сглаживание для устранения микро-шумов. Используется аккуратно, чтобы не размыть QR-код.
        # img_bw = cv.GaussianBlur(img_bw, (5, 5), 0)
        img_bw = cv.GaussianBlur(img_bw, (3, 3), 0)


    else:
        # Удаляет сложные шумы, сохраняя детали. Идеально для начальной очистки без потери границ.
        img_bw = cv.fastNlMeansDenoising(img_bw, h=10, templateWindowSize=7, searchWindowSize=21)

        # Дополнительно сглаживает шум, сохраняя резкие края QR-кода. Работает на уже частично очищенном изображении.
        img_bw = cv.bilateralFilter(img_bw, d=9, sigmaColor=75, sigmaSpace=75)

        # Устраняет остаточный импульсный шум ("соль и перец") после предыдущих этапов.
        img_bw = cv.medianBlur(img_bw, 3)

        # Легкое сглаживание для устранения микро-шумов. Используется аккуратно, чтобы не размыть QR-код.
        img_bw = cv.GaussianBlur(img_bw, (5, 5), 0)

        # Финишная очистка, если после Гауссова размытия остались артефакты.
        img_bw = cv.medianBlur(img_bw, 3)

    # Сохраняем размерность
    if len_image_shape == 3 and keep_dimension:
        if is_bgr:
            img_denoised = cv.cvtColor(img_bw, cv.COLOR_GRAY2BGR)
        else:
            img_denoised = cv.cvtColor(img_bw, cv.COLOR_GRAY2RGB)
    else:
        # В остальных случаях оставляем ЧБ
        img_denoised = img_bw.copy()

    # show_image_cv(img_denoised, title=f"img denoised {img_denoised.shape}")
    return img_denoised

#
def adaptive_correction_hist_cv(image,
                                is_bgr=True,
                                threshold_hist=0.2,
                                keep_dimension=True,
                                logger_func=None):
    """
    Комплексная функция адаптивной коррекции гистограммы изображения

    Принимает 1-канальное или 3-х канальное изображение.
    Возвращает 1-канальное изображение или изображение канальности равной исходному с опцией keep_dimension

    :param image: изображение
    :param is_bgr: считать изображение BGR (иначе многоканальное изображение будет считаться RGB)
    :param threshold_hist: тресхолд изменения метрик изображения
    :param keep_dimension: сохранять количество каналов
    :param logger_func: функция для логгирования или печати
    :return: обработанное ч/б изображение, либо исходное если обработка не потребовалась
    """
    # Проверяем количество каналов
    len_image_shape = len(image.shape)
    # print("Получено изображение: {}".format(image.shape))

    # Переходим к ЧБ
    if len_image_shape == 3:
        if is_bgr:
            img_bw = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            img_bw = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    else:
        img_bw = image.copy()

    # show_image_cv(img_bw, title=f"source {img_bw.shape}")
    assert len(img_bw.shape) == 2, "Для обработки требуется черно-белое изображение"

    # Расчет метрик входного изображения
    metrics = calculate_image_metrics_cv(img_bw,
                                         percentile_low=5,
                                         percentile_high=95)
    #
    img_hist_processed = stretch_hist_cv(img_bw,
                                         clip_percent=1.0,
                                         keep_dimension=True)
    # Image.fromarray(img_bw).show()

    # Расчет метрик после выравнивания гистограммы
    metrics_processed = calculate_image_metrics_cv(img_hist_processed,
                                                   percentile_low=5,
                                                   percentile_high=95)

    # Рассчитываем изменения метрик
    contrast_change = (metrics_processed['contrast'] - metrics['contrast']) / metrics['contrast']
    dynamic_range_change = (metrics_processed['dynamic_range'] - metrics['dynamic_range']) / metrics[
        'dynamic_range']

    # Если контраст или динамический диапазон улучшились более чем на тресхолд, принимаем результат
    if (contrast_change > threshold_hist or
            dynamic_range_change > threshold_hist):
        #
        if logger_func:
            logger_func("Принимаем результат обработки (выравнивание гистограммы):")
            logger_func("  контраст:     {} -> {}".format(metrics['contrast'], metrics_processed['contrast']))
            logger_func("  дин.диапазон: {} -> {}".format(metrics['dynamic_range'], metrics_processed['dynamic_range']))
            logger_func("  резкость:     {} -> {}".format(metrics['sharpness'], metrics_processed['sharpness']))
            logger_func("  уровень шума: {} -> {}".format(metrics['noise'], metrics_processed['noise']))

        # Заменяем изображение img_bw на обработанное
        img_bw = img_hist_processed
    else:
        if logger_func:
            logger_func("Метрики исходного изображения, выравнивание гистограммы не требуется")
            logger_func("  контраст:     {}".format(metrics['contrast']))
            logger_func("  дин.диапазон: {}".format(metrics['dynamic_range']))
            logger_func("  резкость:     {}".format(metrics['sharpness']))
            logger_func("  уровень шума: {}".format(metrics['noise']))

    # Сохраняем размерность
    if len_image_shape == 3 and keep_dimension:
        if is_bgr:
            img_prep = cv.cvtColor(img_bw, cv.COLOR_GRAY2BGR)
        else:
            img_prep = cv.cvtColor(img_bw, cv.COLOR_GRAY2RGB)
    else:
        # В остальных случаях оставляем ЧБ
        img_prep = img_bw.copy()

    # show_image_cv(img_prep, title=f"img preprocessed {img_prep.shape}")
    return img_prep


def preprocessing_bw_cv(image,
                        is_bgr=True,
                        threshold_way='OTSU',
                        force_dilate=False,
                        force_invert=False,
                        keep_dimension=True):
    """
    Комплексная функция предобработки изображения: переход в ч/б, тресхолд опционально.

    Принимает 1-канальное или 3-х канальное изображение.
    Возвращает 1-канальное изображение или изображение канальности равной исходному с опцией keep_dimension

    :param image: изображение
    :param is_bgr: считать изображение BGR (иначе многоканальное изображение будет считаться RGB)
    :param threshold_way: вариант тресхолда изображения по умолчанию
    :param force_dilate: дилейт выходного изображение
    :param force_invert: инвертировать выходное изображение
    :param keep_dimension: сохранять количество каналов
    :return: обработанное ч/б изображение
    """
    # Проверяем количество каналов
    len_image_shape = len(image.shape)
    # print("Получено изображение: {}".format(len_image_shape))

    # Переходим к ЧБ
    if len_image_shape == 3:
        if is_bgr:
            img_bw = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            img_bw = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    else:
        img_bw = image.copy()

    # show_image_cv(img_bw, title=f"source {img_bw.shape}")
    assert len(img_bw.shape) == 2, "Для обработки требуется черно-белое изображение"

    # Инвертированное ЧБ изображение
    thresh = cv.bitwise_not(img_bw)

    # Опция тресхолда
    if threshold_way not in ['GLOBAL', 'OTSU', 'ADAPTIVE']:
        threshold_way = None

    if threshold_way is not None:
        # Гауссово размытие
        # img_bw = cv.GaussianBlur(img_bw, (3, 3), 0)

        if threshold_way == 'GLOBAL':
            # ВАРИАНТ-1: тресхолд глобальный
            thresh = cv.threshold(img_bw, 128, 255, cv.THRESH_BINARY_INV)[1]

        elif threshold_way == 'OTSU':
            # ВАРИАНТ-2: тресхолд OTSU
            thresh = cv.threshold(img_bw, 128, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

        elif threshold_way == 'ADAPTIVE':
            # ВАРИАНТ-3: тресхолд адаптивный
            thresh = cv.adaptiveThreshold(img_bw,255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV,11,2)

        # Применение морфологических операций
        kernel = np.ones((3, 3), np.uint8)
        if force_dilate:
            thresh = cv.dilate(thresh, kernel, iterations=2)
            thresh = cv.erode(thresh, kernel, iterations=1)
        else:
            # pass
            thresh = cv.erode(thresh, kernel, iterations=1)

    # Инвертировать изображение обратно если не указана опция force_invert
    if force_invert:
        img_bw = thresh
    else:
        img_bw = cv.bitwise_not(thresh)

    # Сохраняем размерность
    if len_image_shape == 3 and keep_dimension:
        if is_bgr:
            img_prep = cv.cvtColor(img_bw, cv.COLOR_GRAY2BGR)
        else:
            img_prep = cv.cvtColor(img_bw, cv.COLOR_GRAY2RGB)
    else:
        # В остальных случаях оставляем ЧБ
        img_prep = img_bw.copy()

    # show_image_cv(img_prep, title=f"img preprocessed {img_prep.shape}")
    return img_prep


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


def calculate_image_metrics_cv(image, percentile_low=5, percentile_high=95):
    """
    Рассчитывает ключевые метрики изображения, разделяя контраст и шум.

    Аргументы:
        image (numpy.ndarray): Входное изображение (цветное или в градациях серого).
        percentile_low (int): Нижний процентиль для диапазона яркости.
        percentile_high (int): Верхний процентиль для диапазона яркости.

    Возвращает:
        dict: Словарь с метриками:
            - 'contrast' (стандартное отклонение),
            - 'dynamic_range' (диапазон процентилей),
            - 'sharpness' (оценка резкости),
            - 'noise' (уровень шума через высокочастотные компоненты),
            - 'need_processing' (рекомендация для обработки).
    """
    # Преобразование в градации серого
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Контраст (стандартное отклонение)
    contrast = np.std(gray)

    # Динамический диапазон (процентили)
    low_val = np.percentile(gray, percentile_low)
    high_val = np.percentile(gray, percentile_high)
    dynamic_range = high_val - low_val

    # Резкость (энергия градиента)
    sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    sharpness = np.sqrt(sobel_x ** 2 + sobel_y ** 2).mean()

    # Уровень шума (стандартное отклонение высокочастотных компонент)
    laplacian = cv.Laplacian(gray, cv.CV_64F)
    noise = np.std(laplacian)

    # Решение о необходимости обработки
    # need_processing = (
    #         (contrast < 30) or       # Низкий контраст
    #         (dynamic_range < 80) or  # Узкий динамический диапазон
    #         (sharpness < 20) or      # Низкая резкость
    #         (noise > 15)             # Высокий уровень шума
    # )

    return {
        'contrast': round(contrast, 2),
        'dynamic_range': round(dynamic_range, 2),
        'sharpness': round(sharpness, 2),
        'noise': round(noise, 2),
        # 'need_processing': need_processing
    }


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


def get_encoded_image_list_cv(img_list, extension=".png"):
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


def get_blank_image_cv(height, width, rgb_color=(0, 0, 0), txt_to_put=None):
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


def safety_saving_image_cv(image_path, image, verbose=False):
    """
    Надежная запись файла изображения.
    Возможная ошибка перехватывается.
    Наличие записанного файла и его ненулевой размер проверяется.
    """
    try:
        # Пытаемся записать файл
        ret = cv.imwrite(image_path, image)
        #
        if ret:
            if verbose:
                print("  Файл '{}' сохранён методом imwrite".format(image_path))

            # Дополнительно проверим существование и размер файла
            if os.path.exists(image_path):
                if verbose:
                    print("  Файл '{}' присутствует на диске".format(image_path))
                #
                file_size = os.stat(image_path).st_size
                if verbose:
                    print("  Размер сохраненного файла, байт: {}".format(file_size))
                if file_size == 0:
                    if verbose:
                        print("  Размер сохраненного файла {} = 0".format(image_path))
                    return False
                else:
                    return True
            else:
                if verbose:
                    print("  Файл '{}' отсутствует на диске".format(image_path))
                return False
        else:
            if verbose:
                print("  Ошибка сохранения файла методом imwrite: {}".format(image_path))
            return False

    except Exception as e:
        if verbose:
            print("  Ошибка сохранения файла: {}".format(e))
        return False


def find_horizontal_cv(image,
                       region_coords,
                       blur_kernel_shape=(3, 3),
                       horizontal_kernel_shape=(40, 1),
                       horizontal_iterations=2,
                       min_y_step=30,
                       reverse_sort=False):
    """
    Функция поиска горизонтальных линий

    :param image: изображение
    :param region_coords: координаты региона для поиска горизонталей
    :param blur_kernel_shape: форма ядра для функции блюра
    :param horizontal_kernel_shape: форма ядра
    :param horizontal_iterations: итераций для поиска горизонталей
    :param min_y_step: минимальный шаг между линиями
    :param reverse_sort: обратная сортировка списка координат (нужна если ищем горизонтали снизу вверх)
    :return: список координат Y горизонтальных линий
    """
    # Регион где будем искать горизонтальные линии
    Rx1, Ry1, Rx2, Ry2 = region_coords
    region = image[Ry1:Ry2, Rx1:Rx2]
    #
    if 0 in region.shape:
        print("Некорректное изображение формы {}, горизонтали не могут быть найдены".format(region.shape))
        return []

    # Ищем горизонтальные линии
    gray = cv.cvtColor(region, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, blur_kernel_shape, 0)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    #
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, horizontal_kernel_shape)
    detect_horizontal = cv.morphologyEx(thresh, cv.MORPH_OPEN, horizontal_kernel, iterations=horizontal_iterations)
    cnts = cv.findContours(detect_horizontal, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    #
    y_list = []  # список координат Y горизонтальных линий
    for c in cnts:
        x, y, w, h = cv.boundingRect(c)
        # cv.rectangle(region, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv.drawContours(region, [c], -1, (36, 255, 12), 2)

        if len(y_list) == 0:
            y_list.append(y + Ry1)
            # cv.circle(image, (100, y + Ry1), 10, (255, 0, 0), -1)

        if (len(y_list) > 0) and (abs(y_list[-1] - (y + Ry1)) >= min_y_step):  # линии не могут быть слишком близко
            y_list.append(y + Ry1)
            # cv.circle(image, (100, y + Ry1), 10, (255, 0, 0), -1)
    if reverse_sort:
        y_list.sort(reverse=True)
    else:
        y_list.sort()
    # print("y_list", y_list)
    return y_list


def find_vertical_cv(image,
                     region_coords,
                     blur_kernel_shape=(3, 3),
                     vertical_kernel_shape=(1, 40),
                     vertical_iterations=2,
                     min_x_step=30,
                     reverse_sort=False):
    """
    Функция поиска вертикальных линий

    :param image: изображение
    :param region_coords: координаты региона для поиска вертикалей
    :param blur_kernel_shape: форма ядра для функции блюра
    :param vertical_kernel_shape: форма ядра для морфологии
    :param vertical_iterations: итераций
    :param min_x_step: минимальный шаг между линиями
    :param reverse_sort: обратная сортировка списка координат (нужна если ищем вертикали справа налево)
    :return: список координат X вертикальных линий
    """
    # Регион где будем искать вертикальные линии
    Rx1, Ry1, Rx2, Ry2 = region_coords
    region = image[Ry1:Ry2, Rx1:Rx2]
    #
    if 0 in region.shape:
        print("Некорректное изображение формы {}, вертикали не могут быть найдены".format(region.shape))
        return []

    # Ищем вертикальные линии
    gray = cv.cvtColor(region, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, blur_kernel_shape, 0)
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    #
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, vertical_kernel_shape)
    detect_vertical = cv.morphologyEx(thresh, cv.MORPH_OPEN, vertical_kernel, iterations=vertical_iterations)
    cnts = cv.findContours(detect_vertical, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    #
    x_list = []  # список координат X вертикальных линий
    for c in cnts:
        x, y, w, h = cv.boundingRect(c)
        # cv.rectangle(region, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv.drawContours(region, [c], -1, (36, 255, 12), 2)

        if len(x_list) == 0:
            x_list.append(x + Rx1)
            # cv.circle(image, (100, y + Ry1), 10, (255, 0, 0), -1)

        if (len(x_list) > 0) and (abs(x_list[-1] - (x + Rx1)) >= min_x_step):  # линии не могут быть слишком близко
            x_list.append(x + Rx1)
            # cv.circle(image, (100, y + Ry1), 10, (255, 0, 0), -1)
    if reverse_sort:
        x_list.sort(reverse=True)
    else:
        x_list.sort()
    # print("x_list", x_list)
    return x_list


# def align_by_outer_frame_cv(img,
#                             corner_regions,
#                             diagonals_equality=0.02,
#                             warped_shape=(0, 0),
#                             canny_edges=False,
#                             blur_kernel_shape=(5, 5),
#                             horizontal_kernel_shape=(60, 1),
#                             vertical_kernel_shape=(1, 60),
#                             horizontal_iterations=2,
#                             vertical_iterations=2,
#                             coord_correction=False,
#                             verbose=False):
#     """
#     Поиск прямоугольника по 4-м внешним углам и его выравнивание (трансформация)
#
#     :param img: изображение для обработки
#     :param corner_regions: список координат для поиска углов
#     :param diagonals_equality: метрика относительного равенства диагоналей
#     :param warped_shape: размер трансформируемой части
#     :param canny_edges: производить преобразование Canny перед поиском углов
#     :param blur_kernel_shape: размер ядра для функции блюра
#     :param horizontal_kernel_shape: размер ядра для горизонталей
#     :param horizontal_iterations: итераций для поиска горизонталей
#     :param vertical_kernel_shape: размер ядра для вертикалей
#     :param vertical_iterations: итераций для поиска вертикалей
#     :param coord_correction: пробовать исправлять координаты
#     :param verbose: печатать сопроводительные сообщения
#     :return: image_out: обработанное изображение
#              image_parsed: оригинальное изображение с пометками обработки
#              transformed: если трансформация удачна, то True
#              кортеж: 4 точки по которым делали трансформацию
#     """
#     # Копия исходного изображения
#     image = img.copy()
#     # Изображение для трансформации
#     image_orig = img.copy()
#
#     # ПРЕДОБРАБОТКА: удаление синего цвета
#     light_blue = (93, 75, 74)
#     dark_blue = (151, 255, 255)
#     image = color_filter_hsv_cv(image.copy(), light_hsv=light_blue, dark_hsv=dark_blue)
#     # show_image_cv(image, title="image")
#
#     # ПРЕДОБРАБОТКА: переход к контурам Canny
#     if canny_edges:
#         print("  Задана предобработка Canny edges")
#         gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#         gray = auto_canny(gray)
#         kernel = np.ones((3, 3), np.uint8)
#         gray = cv.dilate(gray, kernel, iterations=2)
#         gray = cv.erode(gray, kernel, iterations=1)
#         image = 255 - gray
#         image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
#
#     # show_image_cv(img_resize_cv(image, 700))
#     # show_image_cv(image)
#
#     # ##### ПО ГОРИЗОНТАЛИ ######
#     #
#     # ЛЕВЫЙ ВЕРХНИЙ УГОЛ
#     # Поиск горизонтальных линий
#     y_list = find_horizontal_cv(image,
#                                 corner_regions[0],
#                                 blur_kernel_shape=blur_kernel_shape,
#                                 horizontal_kernel_shape=horizontal_kernel_shape,
#                                 horizontal_iterations=horizontal_iterations,
#                                 min_y_step=10)
#     if verbose:
#         print("  ЛЕВЫЙ ВЕРХНИЙ y_list:", y_list)
#     if len(y_list) > 0:
#         Y1 = y_list[0]
#     else:
#         Y1 = -1
#
#     # ПРАВЫЙ ВЕРХНИЙ УГОЛ
#     # Поиск горизонтальных линий
#     y_list = find_horizontal_cv(image,
#                                 corner_regions[1],
#                                 blur_kernel_shape=blur_kernel_shape,
#                                 horizontal_kernel_shape=horizontal_kernel_shape,
#                                 horizontal_iterations=horizontal_iterations,
#                                 min_y_step=10)
#     if verbose:
#         print("  ПРАВЫЙ ВЕРХНИЙ y_list:", y_list)
#     if len(y_list) > 0:
#         Y2 = y_list[0]
#     else:
#         Y2 = -1
#
#     # ЛЕВЫЙ НИЖНИЙ УГОЛ
#     # Поиск горизонтальных линий
#     y_list = find_horizontal_cv(image,
#                                 corner_regions[2],
#                                 blur_kernel_shape=blur_kernel_shape,
#                                 horizontal_kernel_shape=horizontal_kernel_shape,
#                                 horizontal_iterations=horizontal_iterations,
#                                 min_y_step=30,
#                                 reverse_sort=True)
#     if verbose:
#         print("  ЛЕВЫЙ НИЖНИЙ y_list:", y_list)
#     if len(y_list) > 0:
#         Y3 = y_list[0]
#     else:
#         Y3 = -1
#
#     # ПРАВЫЙ НИЖНИЙ УГОЛ
#     # Поиск горизонтальных линий
#     y_list = find_horizontal_cv(image,
#                                 corner_regions[3],
#                                 blur_kernel_shape=blur_kernel_shape,
#                                 horizontal_kernel_shape=horizontal_kernel_shape,
#                                 horizontal_iterations=horizontal_iterations,
#                                 min_y_step=30,
#                                 reverse_sort=True)
#     if verbose:
#         print("  ПРАВЫЙ НИЖНИЙ y_list:", y_list)
#     if len(y_list) > 0:
#         Y4 = y_list[0]
#     else:
#         Y4 = -1
#
#     # ##### ПО ВЕРТИКАЛИ ######
#     #
#     # ЛЕВЫЙ ВЕРХНИЙ УГОЛ
#     # Поиск вертикальных линий
#     x_list = find_vertical_cv(image,
#                               corner_regions[4],
#                               blur_kernel_shape=blur_kernel_shape,
#                               vertical_kernel_shape=vertical_kernel_shape,
#                               vertical_iterations=vertical_iterations,
#                               min_x_step=10)
#     if verbose:
#         print("  ЛЕВЫЙ ВЕРХНИЙ x_list:", x_list)
#     if len(x_list) > 0:
#         X1 = x_list[0]
#     else:
#         X1 = -1
#
#     # ПРАВЫЙ ВЕРХНИЙ УГОЛ
#     # Поиск вертикальных линий
#     x_list = find_vertical_cv(image,
#                               corner_regions[5],
#                               blur_kernel_shape=blur_kernel_shape,
#                               vertical_kernel_shape=vertical_kernel_shape,
#                               vertical_iterations=vertical_iterations,
#                               min_x_step=10,
#                               reverse_sort=True)
#     if verbose:
#         print("  ПРАВЫЙ ВЕРХНИЙ x_list:", x_list)
#     if len(x_list) > 0:
#         X2 = x_list[0]
#     else:
#         X2 = -1
#
#     # ЛЕВЫЙ НИЖНИЙ УГОЛ
#     # Поиск вертикальных линий
#     x_list = find_vertical_cv(image,
#                               corner_regions[6],
#                               blur_kernel_shape=blur_kernel_shape,
#                               vertical_kernel_shape=vertical_kernel_shape,
#                               vertical_iterations=vertical_iterations,
#                               min_x_step=10)
#     if verbose:
#         print("  ЛЕВЫЙ НИЖНИЙ x_list:", x_list)
#     if len(x_list) > 0:
#         X3 = x_list[0]
#     else:
#         X3 = -1
#
#     # ПРАВЫЙ НИЖНИЙ УГОЛ
#     # Поиск вертикальных линий
#     x_list = find_vertical_cv(image,
#                               corner_regions[7],
#                               blur_kernel_shape=blur_kernel_shape,
#                               vertical_kernel_shape=vertical_kernel_shape,
#                               vertical_iterations=vertical_iterations,
#                               min_x_step=10,
#                               reverse_sort=True)
#     if verbose:
#         print("  ПРАВЫЙ НИЖНИЙ x_list:", x_list)
#     if len(x_list) > 0:
#         X4 = x_list[0]
#     else:
#         X4 = -1
#
#     # Получили КООРДИНАТЫ УГЛОВ
#     if verbose:
#         print("Получили координаты углов:", (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4))
#     # Обозначим полученные углы
#     if X1 != -1 and Y1 != -1:
#         cv.circle(image, (X1, Y1), 10, (255, 0, 255), -1)
#     if X2 != -1 and Y2 != -1:
#         cv.circle(image, (X2, Y2), 10, (255, 0, 255), -1)
#     if X3 != -1 and Y3 != -1:
#         cv.circle(image, (X3, Y3), 10, (255, 0, 255), -1)
#     if X4 != -1 and Y4 != -1:
#         cv.circle(image, (X4, Y4), 10, (255, 0, 255), -1)
#
#     # ##### ВОССТАНОВЛЕНИЕ ОТДЕЛЬНЫХ КООРДИНАТ #####
#     if -1 in [X1, Y1, X2, Y2, X3, Y3, X4, Y4]:
#         print("Отсутствует минимум одна координата")
#
#     # Восстановление ОДНОЙ отсутствующей КООРДИНАТЫ из [X1, Y1, X2, Y2, X3, Y3, X4, Y4]
#     if (X1 == -1) and (-1 not in [Y1, X2, Y2, X3, Y3, X4, Y4]):
#         print("Нет одной координаты X1, исправляем")
#         X1 = X3 + (X2 - X4)
#         print("Получили координаты углов:", (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4))
#
#     # Восстановление ОДНОЙ отсутствующей КООРДИНАТЫ из [X1, Y1, X2, Y2, X3, Y3, X4, Y4]
#     if (X2 == -1) and (-1 not in [X1, Y1, Y2, X3, Y3, X4, Y4]):
#         print("Нет одной координаты X2, исправляем")
#         X2 = X4 + (X1 - X3)
#         print("Получили координаты углов:", (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4))
#     #
#     if (X3 == -1) and (-1 not in [X1, Y1, X2, Y2, Y3, X4, Y4]):
#         print("Нет одной координаты X3, исправляем")
#         X3 = X1 + (X4 - X2)
#         print("Получили координаты углов:", (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4))
#     #
#     if (X4 == -1) and (-1 not in [X1, Y1, X2, Y2, X3, Y3, Y4]):
#         print("Нет одной координаты X4, исправляем")
#         X4 = X2 - (X1 - X3)
#         print("Получили координаты углов:", (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4))
#     #
#     if (Y1 == -1) and (-1 not in [X1, X2, Y2, X3, Y3, X4, Y4]):
#         print("Нет одной координаты Y1, исправляем")
#         Y1 = Y2 - (Y4 - Y3)
#         print("Получили координаты углов:", (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4))
#     #
#     if (Y2 == -1) and (-1 not in [X1, Y1, X2, X3, Y3, X4, Y4]):
#         print("Нет одной координаты Y2, исправляем")
#         Y2 = Y1 + (Y4 - Y3)
#         print("Получили координаты углов:", (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4))
#     #
#     if (Y3 == -1) and (-1 not in [X1, Y1, X2, Y2, X3, X4, Y4]):
#         print("Нет одной координаты Y3, исправляем")
#         Y3 = Y4 - (Y2 - Y1)
#         print("Получили координаты углов:", (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4))
#     #
#     if (Y4 == -1) and (-1 not in [X1, Y1, X2, Y2, X3, Y3, X4]):
#         print("Нет одной координаты Y4, исправляем")
#         Y4 = Y3 + (Y2 - Y1)
#         print("Получили координаты углов:", (X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4))
#
#     # TODO: Возможно восстановление любого отдельного Y и X отсутствующих одновременно
#
#     # Если нет хотя бы одного угла, то трансформация невозможна
#     if -1 in [X1, Y1, X2, Y2, X3, Y3, X4, Y4]:
#         if verbose:
#             print("Трансформация не возможна (не найден как минимум один угол)")
#         # Выходим из функции
#         return None, image, False, None
#
#     # ##### ИСПРАВЛЕНИЕ ОТДЕЛЬНЫХ КООРДИНАТ #####
#     if coord_correction:
#         # Загрубляем метрику diagonals_equality
#         prev_diagonals_equality = diagonals_equality
#         diagonals_equality = 0.02
#         print("Загрубляем метрику diagonals_equality: {} -> {}".format(prev_diagonals_equality, diagonals_equality))
#
#         # Вычисляем длины диагоналей четырёхугольника по полученным точкам
#         d1 = get_distance_pp((X1, Y1), (X4, Y4))
#         d2 = get_distance_pp((X2, Y2), (X3, Y3))
#         # Если диагонали разные, то пробуем исправить
#         if not rel_equal(d1, d2, diagonals_equality):
#             print("Диагонали разные: {}, {}".format(round(d1), round(d2)))
#             # Вычисляем стороны
#             # s1 = u.get_distance_pp((X1, Y1), (X2, Y2))
#             # s2 = u.get_distance_pp((X2, Y2), (X3, Y3))
#             # s3 = u.get_distance_pp((X3, Y3), (X4, Y4))
#             # s4 = u.get_distance_pp((X4, Y4), (X1, Y1))
#
#             # Восстановление ОДНОЙ НЕПРАВИЛЬНОЙ КООРДИНАТЫ из Y3 или Y4 при условии что остальные корректны
#             # print(Y1, Y2, Y3, Y4)
#             # print(X1, X2, X3, X4)
#             print("Пробуем исправить один нижний угол, при условии что остальные корректны")
#             dY = abs(Y1 - Y2)
#             if Y1 > Y2:
#                 print("Y3:", Y3, "->", Y4 + dY)
#                 Y3 = Y4 + dY
#                 cv.circle(image, (X3, Y3), 10, (0, 0, 255), -1)
#             else:
#                 print("Y4:", Y4, "->", Y3 + dY)
#                 Y4 = Y3 + dY
#                 cv.circle(image, (X4, Y4), 10, (0, 0, 255), -1)
#
#     # Вычисляем длины диагоналей четырёхугольника по полученным точкам
#     d1 = get_distance_pp((X1, Y1), (X4, Y4))
#     d2 = get_distance_pp((X2, Y2), (X3, Y3))
#     print("Диагонали прямоугольника из найденных координат: {}, {}.".format(round(d1), round(d2)))
#     # Если диагонали разные, то трансформация будет некорректна
#     if not rel_equal(d1, d2, diagonals_equality):
#         print("Трансформация не получается (разные диагонали), d1/d2=", round(min(d1, d2) / max(d1, d2), 3))
#         # Выходим из функции
#         return None, image, False, None
#
#     # #############################################################
#     # HOOK: Сдвиг на 5 пикселей чтобы не исчезала рамка справа и снизу
#     X1, Y1, X2, Y2, X3, Y3, X4, Y4 = X1, Y1, X2 + 5, Y2, X3, Y3 + 5, X4 + 5, Y4 + 5
#     # #############################################################
#
#     # ТРАНСФОРМАЦИЯ
#     points = np.array([(X1, Y1), (X2, Y2), (X3, Y3), (X4, Y4)])
#     warped = perspective.four_point_transform(image_orig, points)
#     if verbose and warped is not None:
#         print("Трансформация перспективы выполнена.")
#     # print("image_orig.shape", image_orig.shape)
#     # print("warped.shape", warped.shape)
#
#     # Приведем размер warped к единообразному размеру
#     X_b, Y_b = 0, 0
#     if 0 not in warped_shape:
#         warped = cv.resize(warped, warped_shape, interpolation=cv.INTER_AREA)
#         # Смещение чтобы разместить warped посередине image_out
#         X_b = int((image_orig.shape[1] - warped.shape[1]) / 2)
#         Y_b = int((image_orig.shape[0] - warped.shape[0]) / 2)
#     elif warped_shape[0] != 0 and warped_shape[1] == 0:
#         warped_shape = (warped_shape[0], int(((Y3 - Y1) + (Y4 - Y2)) / 2))
#         warped = cv.resize(warped, warped_shape, interpolation=cv.INTER_AREA)
#         #
#         X_b = int((image_orig.shape[1] - warped.shape[1]) / 2)
#         Y_b = 65
#     # print("warped.shape after resize", warped.shape)
#     # print("X_b, Y_b", X_b, Y_b)
#
#     # Переносим трансформированное изображение на выходную картинку
#     image_out = (np.ones((image_orig.shape[0], image_orig.shape[1], 3)) * 255).astype('uint8')
#     Xdest1, Ydest1 = X_b, Y_b
#     Xdest2, Ydest2 = X_b + warped.shape[1], Y_b + warped.shape[0]
#
#     # warped + Y_b может быть больше image_orig.shape[0]
#     Y_cut = Y_b + warped.shape[0] - image_orig.shape[0]
#     # print("Y_cut", Y_cut)
#     if Y_cut > 0:
#         image_out[Ydest1:Ydest2, Xdest1:Xdest2] = warped[:-Y_cut, :]
#     else:
#         image_out[Ydest1:Ydest2, Xdest1:Xdest2] = warped
#
#     # print("image_out.shape", image_out.shape)
#     # show_image_cv(img_resize_cv(image, img_size=500), title='image')
#     # show_image_cv(img_resize_cv(image_out, img_size=500), title='image_out')
#
#     # Возвращаем обработанное изображение, изображение с пометками, положительный статус операции,
#     # координаты перенесенного на image прямоугольника warped.
#     return image_out, image, True, (Xdest1, Ydest1, Xdest2, Ydest2)


def find_inner_rect_cv(image, line_thickness=2):
    """
    Функция поиска прямоугольника, вписанного во внешнюю границу (рамку) изображения.
    Может использоваться, чтобы получить изображение ячейки таблицы без окружающей рамки

    :param image: изображение
    :param line_thickness: толщина линии
    :return: координаты вписанного прямоугольника, распарсенное изображение
    """

    img = image.copy()
    img_parsed = image.copy()

    h, w = img.shape[:2]
    # print("h, w", h, w)

    #
    blur_kernel_shape = (3, 3)
    kernel_shape = (1, 20)
    iterations = 1
    min_step = 2

    # Зазор от рамки
    gap = line_thickness * 5                 # начальный зазор для морфологических операций
    default_gap = line_thickness * 3         # зазор по умолчанию, если не удалось найли линию
    line_thickness_gap = line_thickness * 2  # зазор, кратный толщине линий добавляется к найденной линии

    # Координаты вписанного прямоугольника
    X1, Y1, X2, Y2 = -1, -1, -1, -1

    # Находим левую стенку ячейки
    area_coords = [0, gap, gap, h-gap]
    # show_image_cv(img[gap:h-gap, 0:gap], title="left {}".format(area_coords))
    x_list = find_vertical_cv(img,
                              area_coords,
                              blur_kernel_shape=blur_kernel_shape,
                              vertical_kernel_shape=kernel_shape,
                              vertical_iterations=iterations,
                              min_x_step=min_step,
                              reverse_sort=True)
    # print("x_list", x_list)
    if len(x_list) > 0:
        X1 = x_list[0] + line_thickness_gap
    else:
        X1 = default_gap

    # Находим правую стенку ячейки
    area_coords = [w-gap, gap, w, h-gap]
    # show_image_cv(img[gap:h-gap, w-gap:w], title="right {}".format(area_coords))
    x_list = find_vertical_cv(img,
                              area_coords,
                              blur_kernel_shape=blur_kernel_shape,
                              vertical_kernel_shape=kernel_shape,
                              vertical_iterations=iterations,
                              min_x_step=min_step,
                              reverse_sort=False)
    # print("x_list", x_list)
    if len(x_list) > 0:
        X2 = x_list[0] - line_thickness_gap
    else:
        X2 = w - default_gap

    # Находим верхнюю стенку ячейки
    area_coords = [gap, 0, w-gap, gap]
    # show_image_cv(img[0:gap, gap:w-gap], title="top {}".format(area_coords))
    y_list = find_horizontal_cv(img,
                                area_coords,
                                blur_kernel_shape=blur_kernel_shape,
                                horizontal_kernel_shape=kernel_shape,
                                horizontal_iterations=iterations,
                                min_y_step=min_step,
                                reverse_sort=True)
    # print("y_list", y_list)
    if len(y_list) > 0:
        Y1 = y_list[0] + line_thickness_gap
    else:
        Y1 = default_gap

    # Находим нижнюю стенку ячейки
    area_coords = [gap, h-gap, w-gap, h]
    # show_image_cv(img[h-gap:h, gap:w-gap], title="bottom {}".format(area_coords))
    y_list = find_horizontal_cv(img,
                                area_coords,
                                blur_kernel_shape=blur_kernel_shape,
                                horizontal_kernel_shape=kernel_shape,
                                horizontal_iterations=iterations,
                                min_y_step=min_step,
                                reverse_sort=False)
    # print("y_list", y_list)
    if len(y_list) > 0:
        Y2 = y_list[0] - line_thickness_gap
    else:
        Y2 = h - default_gap

    #
    assert -1 not in [X1, Y1, X2, Y2]
    cv.rectangle(img_parsed, (X1, Y1), (X2, Y2), s.turquoise, thickness=1)
    return [X1, Y1, X2-w, Y2-h], [X1, Y1, X2, Y2], img_parsed


def scale_pad_image_cv(image, scale_img=2, scale_pad=2, pad_color=(0, 0, 0)):
    """
    Увеличивает изображение и накладывает его на заданный фон.
    Работает с трехканальным изображением
    """
    # Размер исходного изображения
    height, width, _ = image.shape

    # Новый размер изображения
    new_img_height = int(height * scale_img)
    new_img_width = int(width * scale_img)

    # Масштабирование изображения
    resized_image = cv.resize(image, (new_img_width, new_img_height), interpolation=cv.INTER_AREA)

    #
    if scale_pad < scale_img:
        scale_pad = scale_img

    # Создание фонового холста большего размера
    background_width = int(width * scale_pad)
    background_height = int(height * scale_pad)

    # Создаем фоновый холст
    background = np.full((background_height, background_width, 3), pad_color, dtype=np.uint8)

    # Размещение исходного изображения в центре фонового холста
    offset_x = (background_width - new_img_width) // 2
    offset_y = (background_height - new_img_height) // 2

    background[offset_y:offset_y + new_img_height, offset_x:offset_x + new_img_width] = resized_image
    return background


def remove_grid_cv(image,
                   kernel_h=(40,1),
                   kernel_v=(1,20),
                   return_denoised=True,
                   keep_dimension=True):
    """
    Удаление вертикальных и горизонтальных линий
    """
    image_parsed = image.copy()

    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # clahe = cv.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
    # equalized = clahe.apply(gray)
    # threshold = cv.threshold(equalized, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    # denoised = cv.fastNlMeansDenoising(threshold, h=10)

    #
    image_clahe = clahe_hist_cv(image, clip_limit=6)
    gray_equalized = cv.cvtColor(image_clahe, cv.COLOR_BGR2GRAY)
    threshold = cv.threshold(gray_equalized, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    denoised = cv.fastNlMeansDenoising(threshold, h=10)

    # Удаление горизонталей
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_h)
    detect_horizontal = cv.morphologyEx(threshold, cv.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv.findContours(detect_horizontal, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv.drawContours(image_parsed, [c], -1, (255, 255, 255), 5)
        cv.drawContours(threshold, [c], -1, (0, 0, 0), 5)
        cv.drawContours(denoised, [c], -1, (0, 0, 0), 5)

    # Удаление вертикалей
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, kernel_v)
    detect_vertical = cv.morphologyEx(threshold, cv.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv.findContours(detect_vertical, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv.drawContours(image_parsed, [c], -1, (255, 255, 255), 5)
        cv.drawContours(threshold, [c], -1, (0, 0, 0), 5)
        cv.drawContours(denoised, [c], -1, (0, 0, 0), 5)

    if keep_dimension:
        threshold = cv.cvtColor(threshold, cv.COLOR_GRAY2BGR)
        denoised = cv.cvtColor(denoised, cv.COLOR_GRAY2BGR)

    if return_denoised:
        return image_parsed, denoised
    else:
        return image_parsed, threshold


def mask_regions_cv(img_gray, region_list, coord_key='qr_coord'):
    """
    Маскирование регионов, заданных координатами [X1, Y1, X2, Y2]

    :param img_gray: ч/б изображение в формате numpy
    :param region_list: список регионов QR кодов (список словарей)
    :param coord_key: ключ к координатам в списке регионов
    :return: обработанное ч/б изображение
    """
    img_masked = img_gray.copy()

    # Определим среднее значение яркости
    img_corner = img_masked[20:100, 20:100]
    # avg_brightness = int(np.average(img_corner))
    # print("avg_brightness", avg_brightness)

    for r in region_list:
        X1, Y1, X2, Y2 = [-1, -1, -1, -1]

        if 'qr_coord' in r:
            # print("r['qr_coord']", r['qr_coord'])
            X1 = r[coord_key][0]
            Y1 = r[coord_key][1]
            X2 = r[coord_key][2]
            Y2 = r[coord_key][3]

        # Пропускаем если не нашли координаты
        if -1 in [X1, Y1, X2, Y2]:
            continue
        # print("X1, Y1, X2, Y2", X1, Y1, X2, Y2)

        # Создаем изображение для маскировки региона
        W_reg = X2 - X1
        H_reg = Y2 - Y1
        #
        img_reg = img_corner.copy()
        img_reg = cv.resize(img_reg, (W_reg, H_reg), interpolation=cv.INTER_AREA)
        #
        img_reg = cv.blur(img_reg, (3, 3))

        # Маскируем регион исходного изображения
        img_masked[Y1:Y2, X1:X2] = img_reg

    # show_image_cv(img_resize_cv(img_masked, img_size=900), title='img_masked')
    return img_masked


def four_point_transform(image, pts):
    """
    Трансформация перспективы по 4-м точкам

    :param image: изображение для обработки
    :param pts: список координат
    :return: обработанное трансформированное изображение
    """
    def order_points(pts):
        # initialize a list of coordinates that will be ordered
        # such that the first entry in the list is the top-left,
        # the second entry is the top-right, the third is the
        # bottom-right, and the fourth is the bottom-left
        rect = np.zeros((4, 2), dtype="float32")
        # the top-left point will have the smallest sum, whereas
        # the bottom-right point will have the largest sum
        pts = np.array(pts)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        # now, compute the difference between the points, the
        # top-right point will have the smallest difference,
        # whereas the bottom-left will have the largest difference
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        # return the ordered coordinates
        return rect  # TODO: аналогичная функция из find_document_by_grabcut_cv возвращает rect.astype('int').tolist()

    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def find_document_by_grabcut_cv(img, max_size=1080):
    """
    Ищет документ на изображении методом GrabCut
    Хорошо убирает фон если бланк не залазит на края изображения

    :param img: изображение для обработки
    :param max_size: максимальное разрешение, в котором проводить обработку
    :return: True если обработка успешна, обработанное изображение
    """
    def order_points(pts):
        """Rearrange coordinates to order:
           top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype='float32')
        pts = np.array(pts)
        s = pts.sum(axis=1)
        # Top-left point will have the smallest sum.
        rect[0] = pts[np.argmin(s)]
        # Bottom-right point will have the largest sum.
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        # Top-right point will have the smallest difference.
        rect[1] = pts[np.argmin(diff)]
        # Bottom-left will have the largest difference.
        rect[3] = pts[np.argmax(diff)]
        # return the ordered coordinates
        return rect.astype('int').tolist()

    def find_dest(pts):
        (tl, tr, br, bl) = pts
        # Finding the maximum width.
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        # Finding the maximum height.
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        # Final destination co-ordinates.
        destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
        return order_points(destination_corners)

    # Create a copy of original image for final transformation
    img_orig = img.copy()
    #
    H, W = img_orig.shape[:2]
    # print("  Размер оригинального изображения: {}".format((H, W)))

    # Resize image to workable size
    max_dim = max(img.shape)
    if max_dim > max_size:
        resize_scale = max_size / max_dim
        img = cv.resize(img, None, fx=resize_scale, fy=resize_scale)
    #
    h, w = img.shape[:2]
    # print("  Размер изображения после ресайза: {}".format((h, w)))

    # Repeated Closing operation to remove text from the document.
    kernel = np.ones((5, 5), np.uint8)
    img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=3)

    # Check image is almost white
    if image_is_close_to_white_cv(img, threshold=200, white_ratio=0.9)[0]:
        print("  Изображение близко к белому, выход")
        return False, None

    # GrabCut
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # rect = (20, 20, img.shape[1] - 40, img.shape[0] - 40)
    rect = (15, 15, img.shape[1] - 30, img.shape[0] - 30)
    cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    # Gray
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (11, 11), 0)

    # Edge Detection.
    canny = cv.Canny(gray, 0, 200)
    canny = cv.dilate(canny, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))

    # Finding contours for the detected edges.
    contours, hierarchy = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    # Keeping only the largest detected contour.
    page = sorted(contours, key=cv.contourArea, reverse=True)[:5]

    # If there weren't any contours
    if len(page) == 0:
        print("  Не найдено хотя бы 1-го контура, выход")
        return False, None

    # Detecting Edges through Contour approximation.
    # Loop over the contours.
    corners = []
    for c in page:
        # Approximate the contour.
        epsilon = 0.02 * cv.arcLength(c, True)
        corners = cv.approxPolyDP(c, epsilon, True)
        # If our approximated contour has four points.
        if len(corners) == 4:
            break
    # print("  Найдено углов: {}".format(len(corners)))

    # Sorting the corners and converting them to desired shape.
    corners = sorted(np.concatenate(corners).tolist())

    # For 4 corner points being detected.
    if len(corners) >= 4:
        corners = order_points(corners)
        # print("  corners", corners)

        # Recalculating corners in accordance with original image size
        for corner in corners:
            corner[0] = int(corner[0] / h * H)
            corner[1] = int(corner[1] / w * W)
        # print("  corners after recalculating", corners)

        # Отношение диагоналей и размеры четырехугольника из найденных углов
        X1 = corners[0][0]
        Y1 = corners[0][1]
        X2 = corners[1][0]
        Y2 = corners[1][1]
        X3 = corners[2][0]
        Y3 = corners[2][1]
        X4 = corners[3][0]
        Y4 = corners[3][1]
        #
        diag_ratio = get_distance_pp((X1, Y1), (X3, Y3)) / get_distance_pp((X2, Y2), (X4, Y4))
        if diag_ratio > 1.0:
            diag_ratio = 1.0 / diag_ratio
        #
        if X2 - X1 < W * 0.5 or Y4 - Y1 < H * 0.5 or diag_ratio < 0.85:
            print("  Cлишком маленький или искаженный объект: {} * {}".format(X2 - X1, Y4 - Y1))
            return False, None

        # Draw points
        # point_count = 0
        # for corner in corners:
        #     cv.circle(img_parsed, corner, 10, s.blue, -1)
        #     point_count += 1
        #     cv.putText(img_parsed, str(point_count), corner, cv.FONT_HERSHEY_SIMPLEX, 2, s.green, 2)
        # show_image_cv(img_resize_cv(img_parsed, img_size=900), title='img_parsed')

        #
        destination_corners = find_dest(corners)
        # print("  destination_corners", destination_corners)

        # Getting the homography.
        M = cv.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))

        # Perspective transform using homography.
        img_out = cv.warpPerspective(img_orig, M,
                                     (destination_corners[2][0], destination_corners[2][1]),
                                     flags=cv.INTER_LINEAR)

        # Resize результат к исходному изображению
        img_out = cv.resize(img_out, (W, H), interpolation=cv.INTER_AREA)
        # show_image_cv(img_resize_cv(img_out, img_size=900), title='img_perspective ' + str(img_out.shape))

        return True, img_out
    else:
        print("  Не найдено хотя бы 4-ре угла, выход")
        return False, None


# #############################################################
#                       ФУНКЦИИ Pillow
# #############################################################
def convert_opencv_to_pillow(img_cv):
    """
    Конвертация изображения OpenCV в формат Pillow
    """
    img_rgb = cv.cvtColor(img_cv, cv.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    return pil_img


def get_custom_jpeg(img_gray, region_list):
    """
    Создание JPEG с переменным качеством.
    Основное изображение сжимается с большей компрессией.
    Регионы сжимаются с меньшей компрессией (лучшим качеством)

    :param img_gray: ч/б изображение в формате numpy
    :param region_list: список регионов
    :return: обработанное ч/б изображение в формате PILLOW
    """
    img_gray = Image.fromarray(img_gray, mode="L")

    # Сжимаем основное изображение
    buffered = io.BytesIO()
    img_gray.save(buffered, format="JPEG", subsampling=0, quality=s.JPEG_LOW_QUALITY)
    img_poor = Image.open(buffered)
    # img_poor.show()

    for r in region_list:
        X1, Y1, X2, Y2 = [-1, -1, -1, -1]

        if 'boxes' in r:
            # print("r['boxes']", r['boxes'])
            X1 = r['qr_coord'][0]
            Y1 = r['qr_coord'][1]
            X2 = r['boxes'][2]
            Y2 = r['boxes'][3]

        elif 'private_area' in r:
            # print("r['private_area']", r['private_area'])
            X1 = r['private_area'][0]
            Y1 = r['private_area'][1]
            X2 = r['private_area'][2]
            Y2 = r['private_area'][3]

        # Пропускаем если не нашли координаты
        if -1 in [X1, Y1, X2, Y2]:
            continue
        # print("X1, Y1, X2, Y2", X1, Y1, X2, Y2)

        # Выбираем область изображения, которую оставить качественной
        box = (X1, Y1, X2, Y2)
        region = img_gray.crop(box)
        # region.show()

        # Сжимаем выбранную область минимально
        region_buffered = io.BytesIO()
        region.save(region_buffered, "JPEG", quality=s.JPEG_TOP_QUALITY)
        region_good = Image.open(region_buffered)
        # region_good.show()

        # Вставляем область в изображение
        img_poor.paste(region_good, box)

    # image_poor.save("out_files/poor.jpg", format="JPEG", subsampling=0, quality=s.JPEG_QUALITY)
    # img_poor.show()
    return img_poor


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


def get_cmap(n, name='hsv'):
    """
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct RGB color

    :param n: number of distinct color
    :param name: argument name must be a standard mpl colormap name
    :return: function
    """
    return plt.cm.get_cmap(name, n)


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
            else:
                return None


def digits_amount(string):
    """
    Подсчитывает количество цифр в строке

    :param string: строка
    :return: количество цифр
    """
    return sum(map(lambda x: x.isdigit(), string))


def frequency_sort(items):
    """
    Сортировка списка по частоте.
    При одинаковой частоте элементы встанут по возрастанию

    :param items: список
    :return: сортированный список
    """
    temp = []
    sort_dict = {x: items.count(x) for x in items}
    for k, v in sorted(sort_dict.items(), key=lambda x: x[1], reverse=True):
        temp.extend([k] * v)
    return temp


def most_frequent_item(items):
    """
    Определение самого часто встречающегося элемента списка
    """
    # Создаем словарь для хранения частот элементов списка
    frequency_dict = {}

    # Заполняем словарь частотами элементов списка
    for item in items:
        if item in frequency_dict:
            frequency_dict[item] += 1
        else:
            frequency_dict[item] = 1

    # Находим максимальное значение частоты
    max_frequency = max(frequency_dict.values())

    # Проверяем, сколько элементов имеют максимальную частоту
    frequent_items = [item for item, freq in frequency_dict.items() if freq == max_frequency]

    if len(frequent_items) > 1:
        return None  # Если таких элементов больше одного, возвращаем None
    else:
        return frequent_items[0]  # Иначе возвращаем единственное число


def check_string(string, pattern):
    """
    Проверяет соответствие стоки регулярному выражению

    :param string: строка
    :param pattern: паттерн регулярного выражения
    :return: True если строка соответствует регулярному выражению,
        False - если не соответствует,
        None - если паттерн ошибочный
    """
    try:
        compiled_regex = re.compile(pattern)
        return bool(compiled_regex.fullmatch(string))
    except re.error:
        return None


def remove_non_digits(text):
    """
    Оставляет в строке только цифры
    """
    return re.sub(r'\D+', '', text)


def get_deepest_element(lst, index):
    """
    Возвращает элемент с указанным индексом из самого глубоко вложенного списка
    """
    if not isinstance(lst, list):
        return None

    # Если первый элемент является списком, то вызываем функцию для него
    if isinstance(lst[index], list):
        return get_deepest_element(lst[index], index)

    # Иначе возвращаем первый элемент текущего списка
    return lst[index]


def is_base64(string):
    """
    Проверяет, являются ли данные закодированными в Base64
    """
    try:
        return base64.b64encode(base64.b64decode(string)) == string
    except Exception:
        return False


def time_conversion(time_start, time_end):
    """
    Форматирует интервал времени в дни, часы, минуты, секунды
    """
    seconds_in_day = 86400
    seconds_in_hour = 3600
    seconds_in_minute = 60
    seconds = time_end - time_start

    days = seconds // seconds_in_day
    seconds = seconds - (days * seconds_in_day)

    hours = seconds // seconds_in_hour
    seconds = seconds - (hours * seconds_in_hour)

    minutes = seconds // seconds_in_minute
    seconds = seconds - (minutes * seconds_in_minute)

    return days, hours, minutes, seconds


def datetime_formatted(timedelta_hours=3):
    """
    Возвращает форматированную строку даты/времени
    """
    offset = timezone(timedelta(hours=timedelta_hours))
    now = datetime.now(offset)
    now_formatted = now.strftime("%Y-%m-%d %H:%M:%S")

    return now_formatted


def read_json_file(file_path, verbose=False):
    """
    Чтение файла json
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        if verbose:
            print(f"  Файл {file_path} не найден.")
        return None
    except json.JSONDecodeError:
        if verbose:
            print("  Ошибка декодирования JSON. Проверьте правильность формата файла.")
        return None
    except Exception as e:
        if verbose:
            print(f"  Произошла непредвиденная ошибка: {e}")
        return None


def sort_polygons(polygons):
    # Функция для вычисления средней точки полигона
    def get_centroid(polygon):
        x_coords = [point[0] for point in polygon]
        y_coords = [point[1] for point in polygon]
        centroid_x = sum(x_coords) / len(polygon)
        centroid_y = sum(y_coords) / len(polygon)
        return centroid_x, centroid_y

    # Сортировка полигонов по средней точке: сначала по y, затем по x
    # polygons.sort(key=lambda poly: (get_centroid(poly)[1], get_centroid(poly)[0]))

    # Сортировка полигонов по средней точке: сначала по x, затем по y
    polygons.sort(key=lambda poly: (get_centroid(poly)[0], get_centroid(poly)[1]))
    return polygons


# #############################################################
#                  ЭКСПЕРИМЕНТАЛЬНЫЕ функции
# #############################################################

def add_random_distortion_cv(image):
    """
    Случайные по-пиксельные искажения
    """
    # Создадим матрицу случайных смещений
    rows, cols = image.shape[:2]
    dx = np.random.randint(-3, 3, size=(rows, cols)).astype(np.float32)
    dy = np.random.randint(-3, 3, size=(rows, cols)).astype(np.float32)

    # Применим смещения к каждому пикселю
    map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
    map_x = (map_x + dx).astype(np.float32)
    map_y = (map_y + dy).astype(np.float32)

    # Интерполируем изображение с учетом смещений
    distorted_image = cv.remap(image, map_x, map_y, cv.INTER_LINEAR)

    return distorted_image


def perspective_transform_with_white_border_cv(image, border_size=10, scale=1.05, max_shift=10):
    """
    Трансформация перспективны изображения с добавлением белой рамки и случайными искажениями.
    """
    # Расширяем изображение, добавляя белый фон
    padded_image = cv.copyMakeBorder(
        image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv.BORDER_CONSTANT,
        value=[255, 255, 255]
    )

    # Размеры расширенного изображения
    new_rows, new_cols = padded_image.shape[:2]

    # Выбираем четыре точки, представляющие вершины квадрата
    pts1 = np.float32([[0, 0], [new_cols, 0], [0, new_rows], [new_cols, new_rows]])

    # Генерируем случайные смещения для новых вершин
    shift_x = np.random.randint(-max_shift, max_shift, size=4)
    shift_y = np.random.randint(-max_shift, max_shift, size=4)

    # Масштабируем и добавляем смещение
    pts2 = np.float32([
        [shift_x[0], shift_y[0]],
        [scale * new_cols + shift_x[1], shift_y[1]],
        [shift_x[2], scale * new_rows + shift_y[2]],
        [scale * new_cols + shift_x[3], scale * new_rows + shift_y[3]]
    ])

    # Выполняем перспективную трансформацию
    M = cv.getPerspectiveTransform(pts1, pts2)
    transformed_image = cv.warpPerspective(padded_image, M, (int(scale * new_cols), int(scale * new_rows)))

    return transformed_image


def calculate_hist_chi_square_cv(image, threshold=10):
    """
    Рассчитывает метрику ровности гистограммы Хи-квадрат
    """
    # Преобразование в градации серого
    if len(image.shape) == 3:
        img_bw = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        img_bw = image.copy()

    # Вычисление гистограммы
    hist = cv.calcHist([img_bw], [0], None, [256], [0, 256])
    hist = hist / hist.sum()  # Нормализация

    # Идеальное равномерное распределение
    uniform_hist = np.ones_like(hist) / len(hist)

    # Критерий хи-квадрат (меньше значение = ближе к равномерному)
    chi_square = np.sum((hist - uniform_hist) ** 2 / (uniform_hist + 1e-10))

    # Если значение выше порога, требуется выравнивание
    return chi_square > threshold, chi_square


def get_points_in_radius(image_shape, center, radius):
    """
    Возвращает список точек (x, y) в радиусе от центра.

    Параметры:
    image_shape : tuple (height, width)
        Размеры изображения (высота, ширина).
    center : tuple (x, y)
        Координаты центральной точки.
    radius : int
        Радиус окружности в пикселях.

    Возвращает:
    list of tuples
        Список координат точек (x, y), находящихся внутри окружности.
    """
    if radius <= 0:
        return []

    x0, y0 = center
    height, width = image_shape[:2]
    r_squared = radius ** 2

    # Определяем границы области поиска
    x_min = max(0, x0 - radius)
    x_max = min(width, x0 + radius + 1)
    y_min = max(0, y0 - radius)
    y_max = min(height, y0 + radius + 1)

    points = []
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            dx = x - x0
            dy = y - y0
            distance_squared = dx * dx + dy * dy
            if distance_squared <= r_squared:
                points.append((x, y))

    return points


def calculate_average_color_with_outliers(image, points, color_threshold=30):
    """
    Вычисляет средний цвет точек, исключая выбросы, и возвращает отфильтрованные точки

    Параметры:
    image : numpy.ndarray
        Изображение в формате BGR
    points : list of tuples
        Список точек в формате [(x1, y1), (x2, y2), ...]
    color_threshold : int, optional
        Пороговое значение для определения выбросов (по умолчанию 30)

    Возвращает:
    tuple : (avg_bgr, filtered_points)
        Средний цвет в формате (b, g, r) и отфильтрованный список точек
    """
    if not points:
        return (0, 0, 0), []

    # Собираем цвета всех точек
    colors = []
    valid_points = []

    for x, y in points:
        # Проверяем, находится ли точка в пределах изображения
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            colors.append(image[y, x])
            valid_points.append((x, y))

    if not valid_points:
        return (0, 0, 0), []

    colors = np.array(colors)

    # Рассчитываем медиану и MAD для каждого канала
    median_bgr = np.median(colors, axis=0)
    mad_bgr = np.median(np.abs(colors - median_bgr), axis=0)

    # Создаем маску для точек без выбросов
    mask = np.all(np.abs(colors - median_bgr) <= (color_threshold + mad_bgr * 1.5), axis=1)

    # Применяем маску
    filtered_colors = colors[mask]
    filtered_points = [point for point, keep in zip(valid_points, mask) if keep]

    # Рассчитываем средний цвет по оставшимся точкам
    if len(filtered_colors) > 0:
        avg_bgr = np.mean(filtered_colors, axis=0).astype(int)
        avg_bgr_tuple = (int(avg_bgr[0]), int(avg_bgr[1]), int(avg_bgr[2]))
    else:
        avg_bgr_tuple = (0, 0, 0)

    return avg_bgr_tuple, filtered_points


# from scipy.spatial import cKDTree
# import random
#
# def downsample_points(points, target_count, method='grid'):
#     """
#     Прореживает список точек до заданного количества, сохраняя равномерное распределение
#
#     Параметры:
#     points : list of tuples
#         Список точек в формате [(x1, y1), (x2, y2), ...]
#     target_count : int
#         Желаемое количество точек после прореживания
#     method : str, optional
#         Метод прореживания ('grid', 'poisson', 'random')
#         По умолчанию 'grid'
#
#     Возвращает:
#     list of tuples
#         Прореженный список точек
#     """
#     if len(points) <= target_count:
#         return points
#
#     if method == 'random':
#         # Простейшая случайная выборка
#         return random.sample(points, target_count)
#
#     elif method == 'grid':
#         # Метод равномерной сетки
#         xs = [p[0] for p in points]
#         ys = [p[1] for p in points]
#
#         # Определяем границы области
#         min_x, max_x = min(xs), max(xs)
#         min_y, max_y = min(ys), max(ys)
#         width = max_x - min_x
#         height = max_y - min_y
#
#         if width == 0 or height == 0:
#             return random.sample(points, min(target_count, len(points)))
#
#         # Рассчитываем размеры сетки
#         aspect_ratio = width / height
#         cols = int(np.round(np.sqrt(target_count * aspect_ratio)))
#         rows = int(np.round(target_count / cols))
#         actual_count = cols * rows
#
#         # Создаем сетку
#         x_step = width / cols
#         y_step = height / rows
#
#         # Создаем копию для модификации
#         remaining_points = points.copy()
#         result_points = []
#
#         for i in range(rows):
#             for j in range(cols):
#                 # Центр текущей ячейки
#                 cell_center_x = min_x + (j + 0.5) * x_step
#                 cell_center_y = min_y + (i + 0.5) * y_step
#
#                 if not remaining_points:
#                     break
#
#                 # Находим ближайшую точку к центру ячейки
#                 closest_point = min(
#                     remaining_points,
#                     key=lambda p: (p[0] - cell_center_x) ** 2 + (p[1] - cell_center_y) ** 2
#                 )
#
#                 result_points.append(closest_point)
#                 remaining_points.remove(closest_point)
#
#         # Если получилось больше точек, чем нужно (из-за округления)
#         return result_points[:target_count]
#
#     elif method == 'poisson':
#         # Метод Пуассона - более равномерное распределение
#         points_array = np.array(points)
#
#         # Создаем KD-дерево для быстрого поиска соседей
#         tree = cKDTree(points_array)
#
#         # Выбираем случайную начальную точку
#         selected_indices = [random.randint(0, len(points) - 1)]
#         candidates = set(range(len(points))) - set(selected_indices)
#
#         # Рассчитываем минимальное расстояние между точками
#         bbox_diag = np.linalg.norm([max_x - min_x, max_y - min_y])
#         min_distance = bbox_diag / np.sqrt(target_count)
#
#         while candidates and len(selected_indices) < target_count:
#             # Выбираем случайного кандидата
#             candidate_idx = random.choice(list(candidates))
#             candidates.remove(candidate_idx)
#
#             # Проверяем расстояние до уже выбранных точек
#             distances, _ = tree.query(
#                 points_array[candidate_idx],
#                 k=min(len(selected_indices), 3),
#                 distance_upper_bound=min_distance * 1.5
#             )
#
#             if np.all(distances > min_distance):
#                 selected_indices.append(candidate_idx)
#
#         return [points[i] for i in selected_indices]
#
#     else:
#         raise ValueError(f"Unknown method: {method}. Use 'grid', 'poisson' or 'random'")


def filter_masks_by_area(masks,
                         scores,
                         min_area=100,
                         max_area=100000):
    """
    Фильтрует маски по площади в пикселях

    Args:
        masks: массив масок shape (N, H, W)
        scores: массив scores shape (N,)
        min_area: минимальная площадь в пикселях
        max_area: максимальная площадь в пикселях

    Returns:
        filtered_masks: отфильтрованные маски
        filtered_scores: соответствующие scores
        valid_indices: индексы валидных масок
    """
    filtered_masks = []
    filtered_scores = []
    valid_indices = []

    for i, mask in enumerate(masks):
        area = np.sum(mask)  # площадь в пикселях

        if min_area <= area <= max_area:
            filtered_masks.append(mask)
            filtered_scores.append(scores[i])
            valid_indices.append(i)

    if filtered_masks:
        return np.array(filtered_masks), np.array(filtered_scores), valid_indices
    else:
        return np.array([]), np.array([]), []


def filter_masks_by_area_relative(masks,
                                  scores,
                                  image_shape,
                                  min_area_ratio=0.001,
                                  max_area_ratio=0.8):
    """
    Фильтрует маски по относительной площади (доля от общего размера изображения)

    Args:
        masks: массив масок shape (N, H, W)
        scores: массив scores shape (N,)
        image_shape: размер изображения (H, W)
        min_area_ratio: минимальная доля площади (0.001 = 0.1%)
        max_area_ratio: максимальная доля площади (0.8 = 80%)
    """
    total_pixels = image_shape[0] * image_shape[1]
    min_area = int(total_pixels * min_area_ratio)
    max_area = int(total_pixels * max_area_ratio)

    return filter_masks_by_area(masks, scores, min_area, max_area)
