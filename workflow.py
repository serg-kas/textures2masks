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


def split_into_tiles(image, tile_size=1024, overlap=256):
    """
    Разбивает изображение на тайлы с перекрытием.

    Параметры:
        image: numpy array изображения (H, W, C) или (H, W)
        tile_size: размер тайла (квадратный)
        overlap: размер перекрытия между тайлами

    Возвращает:
        tiles: список тайлов
        coords: список координат (y1, x1, y2, x2) для каждого тайла
    """
    # Проверка входных данных
    if image.size == 0:
        return [], []
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= tile_size:
        raise ValueError("overlap must be less than tile_size")

    # Добавление размерности каналов при необходимости
    if image.ndim == 2:
        image = image[..., np.newaxis]
    elif image.ndim != 3:
        raise ValueError("Image must be 2D or 3D array")

    h, w, c = image.shape
    tiles = []
    coords = []
    stride = tile_size - overlap  # эффективный шаг

    # Расчет количества тайлов с корректным округлением
    n_y = (h - overlap + stride - 1) // stride if h > tile_size else 1
    n_x = (w - overlap + stride - 1) // stride if w > tile_size else 1

    # Главный цикл разбиения
    for i in range(n_y):
        for j in range(n_x):
            # Расчет координат без выхода за границы
            y1 = i * stride
            x1 = j * stride
            y2 = min(y1 + tile_size, h)
            x2 = min(x1 + tile_size, w)

            # Вычисление необходимого паддинга
            pad_y = max(0, tile_size - (y2 - y1))
            pad_x = max(0, tile_size - (x2 - x1))

            # Извлечение тайла
            tile = image[y1:y2, x1:x2]

            # Добавление паддинга при необходимости
            if pad_y > 0 or pad_x > 0:
                tile = np.pad(tile,
                              ((0, pad_y), (0, pad_x), (0, 0)),
                              mode='constant')

            tiles.append(tile)
            coords.append((y1, x1, y2, x2))

    return tiles, coords


def assemble_image(tiles, coords, original_shape, overlap):
    """
    Собирает обработанные тайлы в изображение с удалением перекрытий.

    Параметры:
        tiles: список обработанных тайлов
        coords: список координат (y1, x1, y2, x2)
        original_shape: размер исходного изображения (H, W[, C])
        overlap: размер перекрытия, использовавшийся при разбиении
    """
    h, w = original_shape[:2]
    c = tiles[0].shape[2] if tiles[0].ndim == 3 else 1
    result = np.zeros((h, w, c), dtype=tiles[0].dtype)
    weights = np.zeros((h, w), dtype=np.float32)

    for tile, (y1, x1, y2, x2) in zip(tiles, coords):
        # Определяем область обрезки перекрытия
        top = overlap // 2 if y1 > 0 else 0
        left = overlap // 2 if x1 > 0 else 0
        bottom = overlap // 2 if y2 < h else 0
        right = overlap // 2 if x2 < w else 0

        # Вырезаем полезную часть тайла
        tile_cropped = tile[top: tile.shape[0] - bottom,
                       left: tile.shape[1] - right]

        # Координаты для вставки в исходное изображение
        pos_y = y1 + top
        pos_x = x1 + left
        dh, dw = tile_cropped.shape[:2]

        # Убедимся, что не выходим за границы
        if pos_y + dh > h:
            dh = h - pos_y
            tile_cropped = tile_cropped[:dh]
        if pos_x + dw > w:
            dw = w - pos_x
            tile_cropped = tile_cropped[:, :dw]

        # Вставляем в результат с накоплением весов
        result[pos_y:pos_y + dh, pos_x:pos_x + dw] += tile_cropped
        weights[pos_y:pos_y + dh, pos_x:pos_x + dw] += 1

    # Нормализуем с учетом весов
    weights[weights == 0] = 1  # избегаем деления на 0
    return result / weights[..., np.newaxis]





