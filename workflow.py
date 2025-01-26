"""
Функции рабочего процесса
"""

# import numpy as np
# import cv2 as cv
# from PIL import Image, ImageDraw, ImageFont
# import supervision as sv
#
import io
import os
import sys
import time
import re
# import json
# import base64
# import requests
# from multiprocessing import Process, Queue
# from pprint import pprint, pformat
#
import settings as s
import helpers.utils as u
#
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
#
# import torch
# torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
#
# if torch.cuda.get_device_properties(0).major >= 8:
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True


# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(DEVICE)
# DEVICE = torch.device('cpu')
# CHECKPOINT_file = f"models/sam2_hiera_large.pt"
# CONFIG_file = "sam2_hiera_l.yaml"

# sam2_model = build_sam2(CONFIG, CHECKPOINT, device=DEVICE, apply_postprocessing=False)


# print(type(sam2_model))



# exit(77)







# Ресайз изображения к 1024 по большей стороне
def resize_image(image, target_size):
    height, width = image.shape[:2]

    if width > height:
        new_width = target_size
        new_height = round(height * (target_size / width))
    else:
        new_height = target_size
        new_width = round(width * (target_size / height))

    resized_image = cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)
    return resized_image

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


def find_non_overlapping_masks(data, iou_threshold=0.5):
    """
    Находит элементы из списка словарей, маски которых не пересекаются друг с другом по критерию IoU.

    :param data: Исходный список словарей.
    :return: Новый список словарей, содержащие непересекающиеся маски.
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


# Преобразуем маску формата True/False в черно-белое изображение
def convert_mask_to_image(mask):
    img = np.zeros((*mask.shape, 3), dtype=np.uint8)
    img[mask] = [255, 255, 255]
    return img







# Цвета
black = s.black
blue = s.blue
green = s.green
red = s.red
yellow = s.yellow
purple = s.purple
turquoise = s.turquoise
white = s.white


def get_images(source_files, autorotate=None, verbose=False):
    """
    По списку файлов считывает изображения и возвращает их списком.
    В случае необходимости изображения поворачивает на 90, 180 или 270 градусов.
    Для определения ориентации листа используется соответствующий режим Tesseract или функция,
    основанная на детекции QR кодов

    :param source_files: список файлов с полными путями к ним
    :param autorotate: вращать изображение при необходимости (возможные значения: None, TESS, QR)
    :param verbose: выводить подробные сообщения
    :return: список изображений
    """
    time_0 = time.perf_counter()

    if autorotate not in ['TESS', 'QR']:
        autorotate = None

    result = []
    if autorotate is None:
        if verbose:
            print("Загружаем изображения без авторотации")
        for f in source_files:
            img = cv.imread(f)
            if img is None:
                print("  Ошибка чтения файла: {} ".format(f))
                # Создаем фейковую картинку и сохраняем ее
                img_fake = u.get_blank_img_cv(s.SCAN_MAX_SIZE, s.SCAN_MIN_SIZE, s.white,
                                              "Missing file: {}".format(f))
                cv.imwrite(f, img_fake)
                result.append(img_fake)
                continue
            # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            result.append(img)

    elif autorotate == 'TESS':
        #
        print("Импорт модуля PYTESSERACT в функции get_images")
        import pytesseract
        assert 'pytesseract' in sys.modules, "Не удалось импортировать модуль pytesseract"
        #
        if verbose:
            print("Загружаем изображения с авторотацией:")
        for f in source_files:
            try:
                osd = pytesseract.image_to_osd(f)  # по умолчанию min_characters_to_try=50
                # osd = pytesseract.image_to_osd(f, config='--psm 0 -c min_characters_to_try=5')
                angle = re.search(r'(?<=Rotate: )\d+', osd).group(0)
            except pytesseract.pytesseract.TesseractError as e:
                print("  {}, ошибка {}".format(f, e.message[:65-len(f)] + '...'))
                angle = '0'
            #
            if verbose:
                print("  {}, угол поворота {}".format(f, angle))
            img = cv.imread(f)
            if img is None:
                print("  Ошибка чтения файла: {} ".format(f))
                # Создаем фейковую картинку и сохраняем ее
                img_fake = u.get_blank_img_cv(s.SCAN_MAX_SIZE, s.SCAN_MIN_SIZE, s.white,
                                              "Missing file: {}".format(f))
                cv.imwrite(f, img_fake)
                result.append(img_fake)
                continue
            # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            if angle != '0':
                # img = np.rot90(img, k=1, axes=(1,0))
                img = u.image_rotate_cv(img, -float(angle))

            result.append(img)

    else:  # QR
        if verbose:
            print("Загружаем изображения с авторотацией по QR кодам:")
        for f in source_files:
            img = cv.imread(f)
            if img is None:
                print("  Ошибка чтения файла: {} ".format(f))
                # Создаем фейковую картинку и сохраняем ее
                img_fake = u.get_blank_img_cv(s.SCAN_MAX_SIZE, s.SCAN_MIN_SIZE, s.white,
                                              "Missing file: {}".format(f))
                cv.imwrite(f, img_fake)
                result.append(img_fake)
                continue
            # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            # Получаем угол поворота листа по QR кодам
            angle, qrs_number = qr.pyzbar_find_image_orientation(img,
                                                                 qr_filter_function=qr.check_qrcode_type,
                                                                 qrcode_list=None,
                                                                 detector=None)
            if angle is None:
                angle = '0'
                if verbose:
                    print("  {}, угол поворота определить не удалось, загружаем без ротации".format(f))
            else:
                if verbose:
                    print("  {}, угол поворота {} определен по {} QR кодам".format(f, angle, qrs_number))

            if angle != '0':
                # img = np.rot90(img, k=1, axes=(1,0))
                img = u.image_rotate_cv(img, -float(angle))

            result.append(img)

    time_1 = time.perf_counter()
    if verbose:
        print("Загружено изображений: {}, время {:.2f} с.".format(len(result), time_1 - time_0))
    return result

