"""
Функции рабочего процесса
"""

import numpy as np
import cv2 as cv
# from PIL import Image, ImageDraw, ImageFont
# import supervision as sv
#
# import io
# import os
# import sys
import time
# import re
# import json
# import base64
# import requests
# from multiprocessing import Process, Queue
# from pprint import pprint, pformat
#
import settings as s
# import helpers.utils as u
#
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Цвета
black = s.black
blue = s.blue
green = s.green
red = s.red
yellow = s.yellow
purple = s.purple
turquoise = s.turquoise
white = s.white


def get_images_simple(source_files, verbose=False):
    """
    По списку файлов считывает изображения и возвращает их списком.
    Упрощенная функция без определения ротации листа.
    При ошибке чтения файла просто пропускает данное изображение

    :param source_files: список файлов с полными путями к ним
    :param verbose: выводить подробные сообщения
    :return: список изображений
    """
    time_0 = time.perf_counter()

    result = []
    if verbose:
        print("Загружаем изображения")

    for f in source_files:
        img = cv.imread(f)
        if img is None:
            print("  Ошибка чтения файла: {} ".format(f))
            continue
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        result.append(img)

    time_1 = time.perf_counter()
    if verbose:
        print("Загружено изображений: {}, время {:.2f} с.".format(len(result), time_1 - time_0))
    return result


def find_non_overlapping_masks(data, iou_threshold=0.5):
    """
    Находит элементы из списка словарей, маски которых не пересекаются друг с другом по критерию IoU

    :param data: исходный список словарей
    :param iou_threshold: порог
    :return: Новый список словарей, содержащие непересекающиеся маски
    """
    print("Оставляем только не пересекающиеся по тресхолду {} маски".format(iou_threshold))
    counter_overlapping = 0
    non_overlapping_list = []
    for item in data:
        # Извлекаем маску текущего элемента
        current_mask = item["segmentation"]

        # Флаг, показывающий, пересекается ли текущая маска с любой маской из нового списка
        overlaps = False

        for existing_item in non_overlapping_list:
            # Извлекаем маску существующего элемента
            existing_mask = existing_item["segmentation"]

            # Вычисляем IoU для текущей пары масок
            iou = calculate_mask_iou(current_mask, existing_mask)

            # Если IoU больше порога, значит маски пересекаются
            if iou >= iou_threshold:
                print("\r  Нашли пересекающихся масок: {}, IoU = {:.3f}".format(counter_overlapping, iou), end="")
                overlaps = True
                break
        # Если текущая маска не пересекается ни с одной из существующих, добавляем её в новый список
        if not overlaps:
            non_overlapping_list.append(item)
        else:
            counter_overlapping += 1
    print("{}Отбросили {} пересекающихся масок".format(s.CR_CLEAR_cons, counter_overlapping))
    print("{}Нашли {} не пересекающихся масок".format(s.CR_CLEAR_cons, len(non_overlapping_list)))
    return non_overlapping_list


def calculate_mask_iou(mask1, mask2):
    """
    IoU для бинарных масок

    :param mask1:
    :param mask2:
    :return:
    """
    # Убедимся, что маски имеют одинаковый размер
    assert mask1.shape == mask2.shape, "Маски должны иметь одинаковые размеры."

    # Пересечение (AND)
    intersection = np.logical_and(mask1, mask2)

    # Объединение (OR)
    union = np.logical_or(mask1, mask2)

    # Количество пикселей в пересечении и объединении
    intersection_count = np.sum(intersection)
    union_count = np.sum(union)

    # Вычисление IoU
    iou = intersection_count / union_count
    return iou


def convert_mask_to_image(mask):
    """
    Преобразует маску формата True/False в черно-белое изображение

    :param mask: бинарная маска True/False
    :return: ч/б маска
    """
    img = np.zeros((*mask.shape, 3), dtype=np.uint8)
    img[mask] = [255, 255, 255]
    return img


def compute_center_of_mass_cv(binary_mask):
    """
    Нахождение центра масс бинарной маски

    :param binary_mask: маска
    :return: моменты изображения
    """
    # Убедимся, что маска является numpy массивом
    # binary_mask = np.array(binary_mask, dtype=np.uint8)

    # Используем функцию moments для вычисления моментов изображения
    moments = cv.moments(binary_mask, 1)

    # Вычисляем координаты центра масс
    if moments['m00'] != 0:
        cX = int(moments['m10'] / moments['m00'])
        cY = int(moments['m01'] / moments['m00'])
    else:
        # Если m00 равно 0, значит маска пустая
        return None
    return cX, cY


def calculate_mask_dimensions(mask):
    """
    Вычисляет ширину и высоту объекта в бинарной маске.

    Параметры:
        mask: 2D NumPy array булевых значений (False/True)

    Возвращает:
        width, height (int, int). Если объект отсутствует, возвращает (0, 0).
    """
    # Найти координаты всех пикселей объекта (True)
    y_coords, x_coords = np.where(mask)

    # Если объект не найден
    if len(y_coords) == 0:
        return 0, 0

    # Вычислить границы bounding box
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    x_min, x_max = np.min(x_coords), np.max(x_coords)

    # Рассчитать ширину и высоту
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    return width, height


def calculate_mask_dimensions_cv(mask):
    """
    Маска должна быть бинарным изображением
    """
    contours, _ = cv.findContours(
        mask.astype(np.uint8),
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0, 0
    x, y, w, h = cv.boundingRect(contours[0])
    return w, h

