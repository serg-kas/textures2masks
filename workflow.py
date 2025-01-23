"""
Функции рабочего процесса
"""





import numpy as np
import cv2 as cv
# from PIL import Image, ImageDraw, ImageFont
#
import io
import os
import sys
import time
# from multiprocessing import Process, Queue
#
# import json
# import base64
# import requests
#
import re
from pprint import pprint, pformat
#
import settings as s
import helpers.utils as u

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

