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


def resize_image(image, target_size):
    """
    Ресайз изображения к target_size по большей стороне

    :param image: изображение
    :param target_size: целевой размер
    :return: обработанное изображение
    """
    height, width = image.shape[:2]

    if width > height:
        new_width = target_size
        new_height = round(height * (target_size / width))
    else:
        new_height = target_size
        new_width = round(width * (target_size / height))

    resized_image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
    return resized_image


def find_non_overlapping_masks(data, iou_threshold=0.5):
    """
    Находит элементы из списка словарей, маски которых не пересекаются друг с другом по критерию IoU

    :param data: исходный список словарей
    :param iou_threshold: порог
    :return: Новый список словарей, содержащие непересекающиеся маски
    """
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

            # Если IoU больше нуля, значит маски пересекаются
            if iou >= iou_threshold:
                print("Нашли пересекающиеся маски, IoU = {:.2f}".format(iou))
                overlaps = True
                break

        # Если текущая маска не пересекается ни с одной из существующих, добавляем её в новый список
        if not overlaps:
            non_overlapping_list.append(item)

    return non_overlapping_list


# IoU для бинарных масок
def calculate_mask_iou(mask1, mask2):
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


# Преобразуем маску формата True/False в черно-белое изображение
def convert_mask_to_image(mask):
    img = np.zeros((*mask.shape, 3), dtype=np.uint8)
    img[mask] = [255, 255, 255]
    return img


# Нахождение центра масс
def compute_center_of_mass_cv(binary_mask):
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
    return (cX, cY)


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



