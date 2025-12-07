"""
Основной модуль программы.
Режимы работы см.operation_mode_list
"""
import numpy as np
# import random
import cv2 as cv
# from PIL import Image

import os
import sys
import time
from time import perf_counter
# from datetime import datetime, timezone, timedelta
# import shutil
# import requests
# import threading
# import io
import json
# import base64
# from pprint import pprint, pformat
# from pprint import pprint

import config
import settings as s
import tool_case as t
# import helpers.log
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


    # ######################### help ##########################
    # Вывод списка режимов работы и другой информации
    # #########################################################
    if operation_mode == 'help':
        print(u.txt_separator('=', s.CONS_COLUMNS,
                              txt=' Справочная информация ', txt_align='center'))

        print("Предусмотренные режимы работы:")
        # Печатаем элементы с выравниванием
        max_key_length = max(len(key) for key in s.OPERATION_MODE_DICT)
        for key, value in s.OPERATION_MODE_DICT.items():
            print(f"{key:<{max_key_length}} : {value}")


    # ######################### test ##########################
    # Тестовый режим: самопроверка установки, тест скорости
    # #########################################################
    if operation_mode == 'self_test':
        print(u.txt_separator('=', s.CONS_COLUMNS,
                              txt=' Тестовый режим ', txt_align='center'))

        print("Пробная инициализация модели")

        import torch
        if torch.cuda.is_available() and s.SAM2_force_cuda:
            print("GPU доступен, получаем информацию:")
            CUDA_is_present = True

            # Получаем версию CUDA
            cuda_version = torch.version.cuda
            print(f"  Версия CUDA: {cuda_version}")

            # Получаем индекс первого доступного устройства
            index = torch.cuda.current_device()

            # Получаем свойства устройства
            props = torch.cuda.get_device_properties(index)
            # print("props", props)

            # Выводим информацию об устройстве
            print(f"  Название видеокарты: {props.name}")
            # print(f"  Количество мультипроцессоров: {props.multi_processor_count}")
            print(f"  Объем оперативной памяти: {props.total_memory // (1024 ** 2)} МБ")
            print(f"  CUDA compatibility: {props.major}.{props.minor}")

        else:
            print("GPU недоступен или запрещен в настройках")
            CUDA_is_present = False

        if CUDA_is_present:
            # Инициализируем модель на GPU
            model = None
            try:
                print("Запускаем функцию загрузки модели на GPU:")
                model = sam2_model.get_model_sam2(s.SAM2_config_file,
                                                  s.SAM2_checkpoint_file,
                                                  force_cuda=True,
                                                  verbose=s.VERBOSE)
            except:
                print("  Ошибка при загрузке модели на GPU")
                exit(-1)
            #
            if model is not None:
                print("  Модель успешно загрузилась на GPU")
            else:
                print("  Загрузка модели на GPU не удалась")

        else:
            # Инициализируем модель на CPU
            model = None
            try:
                print("Запускаем функцию загрузки модели на CPU:")
                model = sam2_model.get_model_sam2(s.SAM2_config_file,
                                                  s.SAM2_checkpoint_file,
                                                  force_cuda=False,
                                                  verbose=s.VERBOSE)
            except:
                print("  Ошибка при загрузке модели на CPU")
                exit(-1)
            #
            if model is not None:
                print("  Модель успешно загрузилась на CPU")
            else:
                print("  Загрузка модели на CPU не удалась")

        # Инициализация предиктора
        predictor = sam2_model.get_predictor(model, verbose=s.VERBOSE)
        if predictor is not None:
            print("Предиктор инициализирован успешно")
        else:
            print("Ошибка инициализации предиктора")
            exit(-1)

        # Загружаем изображение
        img_file_list = u.get_files_by_type(['images/bricks.png'], s.ALLOWED_IMAGES)
        if len(img_file_list) < 1:
            print("Не нашли изображение для тестовой обработки")
            exit(-1)
        #
        img_list = w.get_images_simple(img_file_list, verbose=s.VERBOSE)
        img = img_list[0]

        # Имя обрабатываемого файла изображения
        img_file = img_file_list[0]
        img_file_base_name = os.path.basename(img_file)
        print("\nОбрабатываем изображение из файла: {}".format(img_file_base_name))
        #
        image_bgr_original = img.copy()
        image_rgb_original = cv.cvtColor(image_bgr_original, cv.COLOR_BGR2RGB)
        print("  Загрузили изображение размерности: {}".format(image_bgr_original.shape))

        # Сохраним размеры оригинального изображения
        H, W = image_bgr_original.shape[:2]
        print("  Сохранили оригинальное разрешение: {}".format((H, W)))

        # Ресайз к номинальному разрешению 1024
        image_bgr = u.resize_image_cv(image_bgr_original, 1024)
        image_rgb = u.resize_image_cv(image_rgb_original, 1024)
        print("  Ресайз изображения: {} -> {}".format(image_bgr_original.shape, image_bgr.shape))

        # Сохраним размеры изображения номинального разрешения 1024
        # height, width = image_rgb.shape[:2]

        # Промт
        prompt_list = json.load(open('images/bricks_prompt.json'))
        if prompt_list is not None:
            print("Используем промт: {}".format(prompt_list))
        else:
            print("Не нашли файл промпта")
            exit(-1)

        # prompt_list [{'type': 'point', 'data': [350, 265], 'label': 1}]
        prompt = prompt_list[0]
        input_point = np.array([prompt['data']])
        # print("input_point", input_point)
        input_label = np.array([prompt['label']])
        # print("input_label", input_label)

        # Запуск с предварительной однократной загрузкой изображения в модель
        N = 5
        print("\nТестовый запуск предикта модели на однократно загруженном изображении, раз: {}".format(N))
        # Загружаем изображение в модель
        start_loading1 = perf_counter()
        predictor.set_image(image_rgb)
        stop_loading1 = perf_counter()
        print("  Изображение загружено в модель, время: {:.5f} с.".format(stop_loading1 - start_loading1))
        #
        time_list = []
        for n in range(N):
            start1 = perf_counter()
            try:
                _, _, _ = predictor.predict(point_coords=input_point,
                                            point_labels=input_label,
                                            multimask_output=True)
            except:
                print("  Ошибка выполнения в предикторе")
                exit(-1)
            stop1 = perf_counter()
            print("  {}.Предикт выполнен успешно, время: {:.5f} с.".format(n+1, stop1 - start1))
            time_list.append(stop1 - start1)
        print("Среднее время выполнения предикта: {:.2f} с.".format(sum(time_list) / len(time_list)))

        # Запуск с загрузкой изображения
        N = 5
        print("\nТестовый запуск предикта модели на загружаемом новом изображении, раз: {}".format(N))
        #
        time_list = []
        for n in range(N):
            # Загружаем изображение в модель
            start_loading2 = perf_counter()
            predictor.set_image(image_rgb)
            stop_loading2 = perf_counter()
            print("  {}.Изображение загружено в модель, время: {:.5f} с.".format(n+1, stop_loading2 - start_loading2))

            start2 = perf_counter()
            try:
                _, _, _ = predictor.predict(point_coords=input_point,
                                            point_labels=input_label,
                                            multimask_output=True)
            except:
                print("  Ошибка выполнения в предикторе")
                exit(-1)
            stop2 = perf_counter()
            print("    Предикт выполнен успешно, время: {:.5f} с.".format(stop2 - start2))
            time_list.append(stop2 - start2)
        print("Среднее время выполнения предикта: {:.2f} с.".format(sum(time_list) / len(time_list)))

        time_end = time.time()
        print("\nОбщее время выполнения: {:.1f} с.".format(time_end - time_start))


    # ################## workflow_baseline #####№№#############
    # Базовый режим обработки изображения с созданием выходной маски
    # промптами от центров масс масок полученных в разрешении 1024
    # #########################################################
    if operation_mode == 'workflow_baseline':
        # Загружаем только изображения
        img_file_list = u.get_files_by_type(source_files, s.ALLOWED_IMAGES)
        if len(img_file_list) < 1:
            print("Не нашли изображений для обработки")
        #
        img_list = w.get_images_simple(img_file_list, verbose=s.VERBOSE)

        # #############################################
        # Загрузка МОДЕЛЕЙ и сохранение их в список
        # экземпляров КЛАССА Tool
        # #############################################
        print(u.txt_separator('=', s.CONS_COLUMNS,
                              txt=' Загрузка и сохранение моделей ', txt_align='center'))

        # Загружаем модель SAM2 в класс Tool
        Tool_list = [t.Tool('model_sam2',
                            sam2_model.get_model_sam2(s.SAM2_config_file,
                                                      s.SAM2_checkpoint_file,
                                                      force_cuda=s.SAM2_force_cuda,
                                                      verbose=s.VERBOSE),
                            tool_type='model')]
        # tool_model_sam2 = t.get_tool_by_name('model_sam2', tool_list=Tool_list)

        # #############################################
        # Обрабатываем файлы из списка
        # #############################################
        print(u.txt_separator('=', s.CONS_COLUMNS,
                              txt=' Обрабатываем файлы из списка ', txt_align='center'))
        #
        time_0 = time.perf_counter()
        # counter_img = 0
        for img_idx, img in enumerate(img_list):
            # Имя обрабатываемого файла изображения
            img_file = img_file_list[img_idx]
            img_file_base_name = os.path.basename(img_file)
            print("Обрабатываем изображение из файла: {}".format(img_file_base_name))

            # Вызываем функцию обработки по базовому алгоритму
            result_dict = w.baseline(img,
                                     Tool_list,
                                     quick_exit=False,
                                     verbose=s.VERBOSE)

            # result_mask1024 = result_dict['result_mask1024']                              # маска в разрешении 1024
            result_mask1024_centers = result_dict['result_mask1024_centers']              # маска с визуализацией центров масс масок
            result_mask1024_original_size = result_dict['result_mask1024_original_size']  # маска в оригинальном разрешении полученная через разрешение 1024
            result_image_final = result_dict['result_image_final']                        #  выходная маска в оригинальном разрешении

            # Имя выходного файла маски в оригинальном разрешении, полученной через ресайз
            out_img_base_name_mask1024 = img_file_base_name[:-4] + "_mask_1024.jpg"
            # Полный путь к выходному файлу
            out_img_file_mask1024 = os.path.join(out_path, out_img_base_name_mask1024)
            # Запись изображения
            try:
                success = cv.imwrite(str(out_img_file_mask1024), result_mask1024_original_size)
                if success:
                    print("Сохранили в оригинальном разрешении маску, полученную через ресайз от 1024: {}".format(
                        out_img_file_mask1024))
                else:
                    print(f'Не удалось сохранить файл {out_img_file_mask1024}')
            except Exception as e:
                print(f'Произошла ошибка при сохранении файла: {e}')

            # Имя выходного файла центров масок в разрешении 1024
            out_img_base_name_mask1024_centers = img_file_base_name[:-4] + "_mask_centers_1024.jpg"
            # Полный путь к выходному файлу
            out_img_file_centers_mask1024 = os.path.join(out_path, out_img_base_name_mask1024_centers)
            # Запись изображения
            try:
                success = cv.imwrite(str(out_img_file_centers_mask1024), result_mask1024_centers)
                if success:
                    print(
                        "Сохранили визуализацию центров масс масок в разрешении 1024: {}".format(out_img_file_centers_mask1024))
                else:
                    print(f'Не удалось сохранить файл {out_img_file_centers_mask1024}')
            except Exception as e:
                print(f'Произошла ошибка при сохранении файла: {e}')

            # Имя выходного файла комбинированной маски в оригинальном разрешении
            out_img_base_name_original_size = img_file_base_name[:-4] + "_mask_W{}xH{}.jpg".format(result_image_final.shape[1],
                                                                                                   result_image_final.shape[0])
            # Полный путь к выходному файлу
            out_img_file_original_size = os.path.join(out_path, out_img_base_name_original_size)
            # Запись изображения
            try:
                success = cv.imwrite(str(out_img_file_original_size), result_image_final)
                if success:
                    print("Сохранили комбинированную маску, полученную в оригинальном разрешении: {}".format(out_img_file_original_size))
                else:
                    print(f'Не удалось сохранить файл {out_img_file_original_size}')
            except Exception as e:
                print(f'Произошла ошибка при сохранении файла: {e}')
            #
            # counter_img += 1
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


    # ################## workflow_tiling ######################
    # Режим обработки изображения на основе разбития на тайлы
    # #########################################################
    if operation_mode == 'workflow_tiling':
        # Загружаем только изображения
        img_file_list = u.get_files_by_type(source_files, s.ALLOWED_IMAGES)
        if len(img_file_list) < 1:
            print("Не нашли изображений для обработки")
        #
        img_list = w.get_images_simple(img_file_list, verbose=s.VERBOSE)

        # #############################################
        # Загрузка МОДЕЛЕЙ и сохранение их в список
        # экземпляров КЛАССА Tool
        # #############################################
        print(u.txt_separator('=', s.CONS_COLUMNS,
                              txt=' Загрузка и сохранение моделей ', txt_align='center'))

        # Загружаем модель SAM2 в класс Tool
        Tool_list = [t.Tool('model_sam2',
                            sam2_model.get_model_sam2(s.SAM2_config_file,
                                                      s.SAM2_checkpoint_file,
                                                      force_cuda=s.SAM2_force_cuda,
                                                      verbose=s.VERBOSE),
                            tool_type='model')]
        tool_model_sam2 = t.get_tool_by_name('model_sam2', tool_list=Tool_list)

        # #############################################
        # Обрабатываем файлы из списка
        # #############################################
        print(u.txt_separator('=', s.CONS_COLUMNS,
                              txt=' Обрабатываем файлы из списка ', txt_align='center'))
        #
        time_0 = time.perf_counter()
        # counter_img = 0
        for img_idx, img in enumerate(img_list):
            # Имя обрабатываемого файла изображения
            img_file = img_file_list[img_idx]
            img_file_base_name = os.path.basename(img_file)
            print("Обрабатываем изображение из файла: {}".format(img_file_base_name))

            # Вызываем функцию обработки по базовому алгоритму
            result_dict = w.baseline(img,
                                     Tool_list,
                                     quick_exit=True,
                                     verbose=s.VERBOSE)

            # result_mask1024 = result_dict['result_mask1024']                              # маска в разрешении 1024
            # result_mask1024_centers = result_dict['result_mask1024_centers']              # маска с визуализацией центров масс масок
            result_mask1024_original_size = result_dict['result_mask1024_original_size']  # маска в оригинальном разрешении полученная через разрешение 1024

            # Имя выходного файла маски в оригинальном разрешении, полученной через ресайз
            out_img_base_name_mask1024 = img_file_base_name[:-4] + "_mask_1024.jpg"
            # Полный путь к выходному файлу
            out_img_file_mask1024 = os.path.join(out_path, out_img_base_name_mask1024)
            # Запись изображения
            try:
                success = cv.imwrite(str(out_img_file_mask1024), result_mask1024_original_size)
                if success:
                    print("Сохранили в оригинальном разрешении маску, полученную через ресайз от 1024: {}".format(
                        out_img_file_mask1024))
                else:
                    print(f'Не удалось сохранить файл {out_img_file_mask1024}')
            except Exception as e:
                print(f'Произошла ошибка при сохранении файла: {e}')


            # Разбиение и обработка изображение на тайлы
            image_bgr_original = img.copy()
            tiles_list, coords_list = w.split_into_tiles(image_bgr_original,
                                                         tile_size=s.TILING_SIZE,
                                                         overlap=s.TILING_OVERLAP)
            print("Изображение {} разбито на фрагменты (тайлы): {}".format(img_file, len(tiles_list)))

            # Сохранение списка тайлов в файлы
            # for idx, tile in enumerate(tiles_list):
            #     # Имя выходного файла тайла
            #     out_tile_base_name = img_file_base_name[:-4] + f"_tile_{idx}.jpg"
            #     # Полный путь к выходному файлу
            #     out_tile_file = os.path.join(out_path, out_tile_base_name)
            #     if cv.imwrite(str(out_tile_file), tile):
            #         print("  Сохранили тайл: {}".format(out_tile_file))

            # TODO: Обработка каждого тайла
            # processed_tiles = [cv.cvtColor(tile, cv.COLOR_BGR2RGB) for tile in tiles_list]

            # Сборка выходного изображения
            # image_rgb_reconstructed = w.assemble_image(processed_tiles,
            #                                            coords_list,
            #                                            original_shape=image_bgr_original.shape,
            #                                            overlap=s.TILING_OVERLAP)
            # Сохранение собранного файла
            # out_new_base_name = img_file_base_name[:-4] + "_reconstructed.jpg"
            # Полный путь к выходному файлу
            # out_new_file = os.path.join(out_path, out_new_base_name)
            # image_bgr_reconstructed = cv.cvtColor(image_rgb_reconstructed, cv.COLOR_RGB2BGR)
            # if cv.imwrite(str(out_new_file), image_bgr_reconstructed):
            #     print("  Сохранили выходной файл: {}".format(out_new_file))


            # Разбиение маски, полученной через разрешение 1024 на тайлы
            mask_tiles_list, mask_coords_list = w.split_into_tiles(result_mask1024_original_size,
                                                                   tile_size=s.TILING_SIZE,
                                                                   overlap=s.TILING_OVERLAP)
            print("Маска в оригинальном разрешении разбита на фрагменты (тайлы): {}".format(len(mask_tiles_list)))
            # Сохранение списка тайлов масок в файлы
            # for idx, tile in enumerate(mask_tiles_list):
            #     # Имя выходного файла тайла
            #     out_tile_base_name = img_file_base_name[:-4] + f"_mask_tile_{idx}.jpg"
            #     # Полный путь к выходному файлу
            #     out_tile_file = os.path.join(out_path, out_tile_base_name)
            #     if cv.imwrite(str(out_tile_file), tile):
            #         print("  Сохранили тайл маски: {}".format(out_tile_file))

            # Инициализация предиктора
            predictor = sam2_model.get_predictor(tool_model_sam2.model, verbose=s.VERBOSE)
            if predictor is not None:
                if s.VERBOSE:
                    print("  Предиктор инициализирован успешно")
            else:
                pass
                # TODO: если нет предиктора, выйти с ошибкой или пустым результатом

            # Цикл обработки тайлов (сегментация в оригинальном разрешении)
            if s.TILING_INVERSE_MODE:
                print("Сегментация тайлов в оригинальном разрешении c инверсией")
            else:
                print("Сегментация тайлов в оригинальном разрешении без инверсии")

            processed_mask_list = []
            for idx, curr_mask in enumerate(mask_tiles_list):
                print("  Тайл {}/{}".format(idx+1, len(mask_tiles_list)))
                #
                curr_tile = tiles_list[idx]
                # u.show_image_cv(u.resize_image_cv(np.concatenate((curr_tile, curr_mask), axis=1), img_size=1024), title='tile | mask')

                if s.TILING_INVERSE_MODE:
                    # 1. Подготовка маски-промпта
                    custom_mask = cv.cvtColor(curr_mask, cv.COLOR_BGR2GRAY)
                    custom_mask = 255 - custom_mask

                    low_res_mask = cv.resize(custom_mask.astype(np.uint8), (256, 256), interpolation=cv.INTER_NEAREST)
                    # u.show_image_cv(low_res_mask, title='low_res_mask: {}'.format(low_res_mask.shape))

                    # 2. Нормализация: [0, 255] -> [0, 1]
                    mask_input = (low_res_mask > 128).astype(np.float32)
                    # u.show_image_cv(mask_input, title='mask_input: {}'.format(mask_input.shape))

                    # 3. Генерация точечных промптов
                    # point_coords, point_labels = sam2_model.prepare_prompts_from_mask(custom_mask, num_points=1000)

                    # 4. Нормализация координат точек к размеру тайла
                    # if len(point_coords) > 0:
                    #     height, width = curr_tile.shape[:2]
                    #     point_coords_normalized = point_coords / np.array([width, height])
                    # else:
                    #     point_coords_normalized = None
                    #     point_labels = None

                    # 5. Установка изображения
                    predictor.set_image(curr_tile)

                    # 6. Предсказание с комбинацией промптов
                    # TODO: Используем только промпт маской
                    masks, scores, _ = predictor.predict(
                        # point_coords=point_coords_normalized,
                        # point_labels=point_labels,
                        point_coords=None,
                        point_labels=None,
                        box=None,
                        mask_input=mask_input[None, :, :],
                        # mask_input=None,
                        multimask_output=True
                    )
                else:
                    # 1. Подготовка маски-промпта
                    custom_mask = cv.cvtColor(curr_mask, cv.COLOR_BGR2GRAY)

                    low_res_mask = cv.resize(custom_mask.astype(np.uint8), (256, 256), interpolation=cv.INTER_NEAREST)
                    # u.show_image_cv(low_res_mask, title='low_res_mask: {}'.format(low_res_mask.shape))

                    # 2. Нормализация: [0, 255] -> [0, 1]
                    mask_input = (low_res_mask > 128).astype(np.float32)
                    # u.show_image_cv(mask_input, title='mask_input: {}'.format(mask_input.shape))

                    # 3. Генерация точечных промптов
                    # TODO: собрать все точки - центры масс в пределах данного тайла или достаточно того что делает prepare_prompts_from_mask?
                    point_coords, point_labels = sam2_model.prepare_prompts_from_mask(custom_mask, num_points=1000)

                    # 4. Нормализация координат точек к размеру тайла
                    if len(point_coords) > 0:
                        height, width = curr_tile.shape[:2]
                        point_coords_normalized = point_coords / np.array([width, height])
                    else:
                        point_coords_normalized = None
                        point_labels = None

                    # 5. Установка изображения
                    predictor.set_image(curr_tile)

                    # 6. Предсказание с комбинацией промптов
                    masks, scores, _ = predictor.predict(
                        point_coords=point_coords_normalized,
                        point_labels=point_labels,
                        # point_coords=None,
                        # point_labels=None,
                        box=None,
                        mask_input=mask_input[None, :, :],
                        # mask_input=None,
                        multimask_output=True
                    )
                tool_model_sam2.counter += 1
                # print(masks.shape, scores.shape)

                # TODO: Фильтр масок по размеру
                # masks, filtered_scores, valid_indices = u.filter_masks_by_area(masks,
                #                                                                scores,
                #                                                                1000,
                #                                                                800000)
                # masks, filtered_scores, valid_indices = u.filter_masks_by_area_relative(masks,
                #                                                                         scores,
                #                                                                         curr_tile.shape,
                #                                                                         min_area_ratio=0.1,
                #                                                                         max_area_ratio=0.8)

                # TODO: ВАР-1. Выбор маски по максимальному score
                # mask_idx = np.argmax(scores)
                # print("scores", scores, mask_idx)

                # TODO: ВАР-2. Выбор маски по максимальному iou
                iou_list = [w.calculate_mask_iou(custom_mask, pred_mask) for pred_mask in masks]
                mask_idx = np.argmax(iou_list)
                # print("iou_list", iou_list, mask_idx)

                #
                masks_img = masks[mask_idx].astype(np.uint8) * 255
                # В инверсном режиме готовую маску надо инвертировать
                if s.TILING_INVERSE_MODE:
                    masks_img = 255 - masks_img

                masks_img = cv.cvtColor(masks_img, cv.COLOR_GRAY2BGR)
                processed_mask_list.append(masks_img)
                # u.show_image_cv(u.resize_image_cv(masks_img), title='masks_img')

            # Сборка выходной маски
            image_bgr_tiling = w.assemble_image(processed_mask_list,
                                                coords_list,
                                                original_shape=image_bgr_original.shape,
                                                overlap=s.TILING_OVERLAP)
            # u.show_image_cv(u.resize_image_cv(image_bgr_tiling), title='masks_img')

            image_bgr_tiling = cv.cvtColor(image_bgr_tiling, cv.COLOR_BGR2GRAY)
            # print(image_bgr_tiling.shape)

            # TODO Убираем шум
            kernel = np.ones((9, 9), np.uint8)
            mask_cleaned = cv.morphologyEx(image_bgr_tiling,
                                           cv.MORPH_OPEN, kernel)

            # TODO Заполняем небольшие отверстия
            image_bgr_tiling = cv.morphologyEx(mask_cleaned,
                                           cv.MORPH_CLOSE, kernel)
            # u.show_image_cv(u.resize_image_cv(image_bgr_tiling), title='masks_img_cleaned')

            image_bgr_tiling = cv.cvtColor(image_bgr_tiling, cv.COLOR_GRAY2BGR)


            # Имя выходного файла тайла
            out_new_base_name = img_file_base_name[:-4] + "_tiling_mask.jpg"
            # Полный путь к выходному файлу
            out_new_file = os.path.join(out_path, out_new_base_name)
            if cv.imwrite(str(out_new_file), image_bgr_tiling):
                print("  Сохранили выходной файл: {}".format(out_new_file))

        time_1 = time.perf_counter()
        print("Обработали изображений: {}, время {:.2f} с.".format(len(img_file_list),
                                                                   time_1 - time_0))
        #
        time_end = time.time()
        if s.VERBOSE:
            print("Общее время выполнения: {:.1f} с.".format(time_end - time_start))


    # ######################## new1 ###########################
    # Режим new1
    # #########################################################
    if operation_mode == 'new1':
        # Загружаем только изображения
        img_file_list = u.get_files_by_type(source_files, s.ALLOWED_IMAGES)
        if len(img_file_list) < 1:
            print("Не нашли изображений для обработки")
        #
        img_list = w.get_images_simple(img_file_list, verbose=s.VERBOSE)

        # #############################################
        # Загрузка МОДЕЛЕЙ и сохранение их в список
        # экземпляров КЛАССА Tool
        # #############################################
        print(u.txt_separator('=', s.CONS_COLUMNS,
                              txt=' Загрузка и сохранение моделей ', txt_align='center'))

        # Загружаем модель SAM2 в класс Tool
        Tool_list = [t.Tool('model_sam2',
                            sam2_model.get_model_sam2(s.SAM2_config_file,
                                                      s.SAM2_checkpoint_file,
                                                      force_cuda=s.SAM2_force_cuda,
                                                      verbose=s.VERBOSE),
                            tool_type='model')]
        # tool_model_sam2 = t.get_tool_by_name('model_sam2', tool_list=Tool_list)

        # #############################################
        # Обрабатываем файлы из списка
        # #############################################
        print(u.txt_separator('=', s.CONS_COLUMNS,
                              txt=' Обрабатываем файлы из списка ', txt_align='center'))
        #
        time_0 = time.perf_counter()
        # counter_img = 0
        for img_idx, img in enumerate(img_list):
            # Имя обрабатываемого файла изображения
            img_file = img_file_list[img_idx]
            img_file_base_name = os.path.basename(img_file)
            print("Обрабатываем изображение из файла: {}".format(img_file_base_name))

            # Вызываем функцию обработки по базовому алгоритму
            result_dict = w.baseline(img,
                                     Tool_list,
                                     quick_exit=False,
                                     verbose=s.VERBOSE)

            # result_mask1024 = result_dict['result_mask1024']                              # маска в разрешении 1024
            result_mask1024_centers = result_dict['result_mask1024_centers']              # маска с визуализацией центров масс масок
            result_mask1024_original_size = result_dict['result_mask1024_original_size']  # маска в оригинальном разрешении полученная через разрешение 1024
            result_image_final = result_dict['result_image_final']                        #  выходная маска в оригинальном разрешении

            # Имя выходного файла маски в оригинальном разрешении, полученной через ресайз
            out_img_base_name_mask1024 = img_file_base_name[:-4] + "_mask_1024.jpg"
            # Полный путь к выходному файлу
            out_img_file_mask1024 = os.path.join(out_path, out_img_base_name_mask1024)
            # Запись изображения
            try:
                success = cv.imwrite(str(out_img_file_mask1024), result_mask1024_original_size)
                if success:
                    print("Сохранили в оригинальном разрешении маску, полученную через ресайз от 1024: {}".format(
                        out_img_file_mask1024))
                else:
                    print(f'Не удалось сохранить файл {out_img_file_mask1024}')
            except Exception as e:
                print(f'Произошла ошибка при сохранении файла: {e}')

            # Имя выходного файла центров масок в разрешении 1024
            out_img_base_name_mask1024_centers = img_file_base_name[:-4] + "_mask_centers_1024.jpg"
            # Полный путь к выходному файлу
            out_img_file_centers_mask1024 = os.path.join(out_path, out_img_base_name_mask1024_centers)
            # Запись изображения
            try:
                success = cv.imwrite(str(out_img_file_centers_mask1024), result_mask1024_centers)
                if success:
                    print(
                        "Сохранили визуализацию центров масс масок в разрешении 1024: {}".format(out_img_file_centers_mask1024))
                else:
                    print(f'Не удалось сохранить файл {out_img_file_centers_mask1024}')
            except Exception as e:
                print(f'Произошла ошибка при сохранении файла: {e}')

            # Имя выходного файла комбинированной маски в оригинальном разрешении
            out_img_base_name_original_size = img_file_base_name[:-4] + "_mask_W{}xH{}.jpg".format(result_image_final.shape[1],
                                                                                                   result_image_final.shape[0])
            # Полный путь к выходному файлу
            out_img_file_original_size = os.path.join(out_path, out_img_base_name_original_size)
            # Запись изображения
            try:
                success = cv.imwrite(str(out_img_file_original_size), result_image_final)
                if success:
                    print("Сохранили комбинированную маску, полученную в оригинальном разрешении: {}".format(out_img_file_original_size))
                else:
                    print(f'Не удалось сохранить файл {out_img_file_original_size}')
            except Exception as e:
                print(f'Произошла ошибка при сохранении файла: {e}')
            #
            # counter_img += 1
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
    config.check_folders(
        [s.SOURCE_PATH,
         s.OUT_PATH],
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
