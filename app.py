"""
Основной модуль программы.
Режимы работы см.operation_mode_list
"""
# import numpy as np
# import cv2 as cv
# import numpy as np
# from PIL import Image
#
import os
import sys
import time
# from datetime import datetime, timezone, timedelta
# import shutil
# import requests
# import magic
# import threading
# import io
# import json
# import base64
# from pprint import pprint, pformat
# from pprint import pprint

#
import config
# import log
import settings as s
import tool_case as t
import helpers.utils as u
import workflow as w
import sam2_model


# Рабочая директория: полный путь и локальная папка
working_folder_full_path = os.getcwd()
print("Рабочая директория (полный путь): {}".format(working_folder_full_path))


# ##################### РЕЖИМЫ РАБОТЫ #########################
operation_mode_list =s.OPERATION_MODE_LIST  # список режимов работы
default_mode = s.DEFAULT_MODE               # режим работы по умолчанию


def process(operation_mode, source_files, out_path):
    """
    Функция запуска рабочего процесса по выбранному
    режиму работы
    :param operation_mode: режим работы
    :param source_files: список файлов для обработки
    :param out_path: путь для сохранения результата
    :return:
    """
    time_start = time.time()

    # Выбор режима работы. Находит первый подходящий режим.
    # (например, по параметру 'tesT' будет выбран режим 'self_test')
    for curr_mode in operation_mode_list:
        if operation_mode.lower() in curr_mode:
            operation_mode = curr_mode
            break
    print('Режим работы, заданный в параметрах командной строки: {}'.format(operation_mode))

    # ###################### test #######################
    # TODO: Тестовый режим (самопроверка установки)
    # #########################################################
    if operation_mode == 'self_test':
        print(u.txt_separator('=', s.CONS_COLUMNS,
                              txt=' Тестовый режим ', txt_align='center'))
        #
        time_end = time.time()
        print("Общее время выполнения: {:.1f} с.".format(time_end - time_start))


    # #################### workflow_masks #####№№################
    # TODO: Рабочий режим обработки изображения с созданием выходной маски
    # #########################################################
    if operation_mode == 'workflow_masks':
        # Загружаем модель детектора текста

        # Загружаем только изображения
        img_file_list = u.get_files_by_type(source_files, s.ALLOWED_IMAGES)
        if len(img_file_list) < 1:
            print("Не нашли изображений для обработки")

        img_list = w.get_images_simple(img_file_list, verbose=s.VERBOSE)

        # #############################################
        # Загрузка МОДЕЛЕЙ и сохранение их в список
        # экземпляров КЛАССА Tool
        # #############################################
        print(u.txt_separator('=', s.CONS_COLUMNS,
                              txt=' Загрузка и сохранение моделей ', txt_align='center'))

        # Загружаем модель SAM2
        Tool_list = [t.Tool('model_sam2',
                            sam2_model.get_model_sam2(s.SAM2_config_file,
                                                      s.SAM2_checkpoint_file,
                                                      force_cuda=s.SAM2_force_cuda,
                                                      verbose=s.VERBOSE),
                            tool_type='model')]

        # #############################################
        # Обрабатываем файлы из списка
        # #############################################
        time_0 = time.perf_counter()

        counter = 0
        for img in img_list:
            #
            # start_time_qr = time.perf_counter()
            # img_parsed, _ = craft.detect_text(img,
            #                                   detector=detector)
            # print("{}Отработал детектор текста craft, время {:.3f} с.{}".format(s.MAGENTA_cons,
            #                                                                     time.perf_counter() - start_time_qr,
            #                                                                     s.RESET_cons))
            # if img_parsed is not None:
            #     Image.fromarray(img_parsed[:, :, ::-1]).show()
            #
            # out_file_name = os.path.join(out_path, 'crafted_' + str(counter) + '.png')
            # cv.imwrite(out_file_name, img_parsed)
            counter += 1
        #
        time_1 = time.perf_counter()
        print("Обработали изображений: {}, время {:.2f} с.".format(len(img_file_list),
                                                                   time_1 - time_0))

        # #############################################
        # Выведем статистику использования ИНСТРУМЕНТОВ
        # #############################################
        print(u.txt_separator('=', s.CONS_COLUMNS,
                              txt=' Статистика использования экземпляров'
                                  ' КЛАССА Tool (инструментов) ', txt_align='center'))
        for tool in Tool_list:
            print("Инструмент (модель) {}, вызывали, раз: {}".format(tool.name, tool.counter))

        # #############################################
        # Удалим экземпляры КЛАССОВ и ненужные файлы
        # #############################################
        print(u.txt_separator('=', s.CONS_COLUMNS,
                              txt=' Удаление экземпляров КЛАССОВ Tool и ненужных файлов ', txt_align='center'))
        #
        time_end = time.time()
        if s.VERBOSE:
            print("Общее время выполнения: {:.1f} с.".format(time_end - time_start))


if __name__ == '__main__':
    """
    В программу можно передать параметры командной строки:
    sys.argv[1] - operation_mode - режим работы 
    sys.argv[2] - source - путь к папке или отдельному файлу для обработки
    sys.argv[3] - out_path - путь к папке для сохранения результатов
    """

    # Проверим наличие и создадим рабочие папки если их нет
    config.check_folders([s.SOURCE_PATH, s.OUT_PATH],
                         verbose=s.VERBOSE)

    # Параметры командной строки
    OPERATION_MODE = default_mode if len(sys.argv) <= 1 else sys.argv[1]
    SOURCE = s.SOURCE_PATH if len(sys.argv) <= 2 else sys.argv[2]
    OUT_PATH = s.OUT_PATH if len(sys.argv) <= 3 else sys.argv[3]

    # Если в параметрах источник - это папка
    if os.path.isdir(SOURCE):
        SOURCE_FILES = []
        for file in sorted(os.listdir(SOURCE)):
            if os.path.isfile(os.path.join(SOURCE, file)):
                _, file_extension = os.path.splitext(file)
                # Берем только разрешенные типы файлы
                if file_extension in s.ALLOWED_TYPES:
                    SOURCE_FILES.append(os.path.join(SOURCE, file))
        # Отправляем в работу не проверяя, что source_files может быть пуст
        process(OPERATION_MODE, SOURCE_FILES, OUT_PATH)

    # Если в параметрах источник - это файл
    elif os.path.isfile(SOURCE):
        _, file_extension = os.path.splitext(SOURCE)
        # Берем только разрешенный типы файлов
        if file_extension in s.ALLOWED_TYPES:
            source_file = SOURCE
            print('Обрабатываем файл: {}'.format(source_file))
            # Отправляем файл в работу
            process(OPERATION_MODE, [source_file], OUT_PATH)

    # Иначе не нашли данных для обработки
    else:
        print("Не нашли данных для обработки: {}".format(SOURCE))
