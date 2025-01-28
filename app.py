"""
Основной модуль программы.
Режимы работы см.operation_mode_list
"""
import numpy as np
import cv2 as cv
# from PIL import Image
#
import os
import sys
import time
# from datetime import datetime, timezone, timedelta
# import shutil
# import requests
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
    # Тестовый режим (самопроверка установки)
    # #########################################################
    if operation_mode == 'self_test':
        print(u.txt_separator('=', s.CONS_COLUMNS,
                              txt=' Тестовый режим ', txt_align='center'))

        import torch
        if torch.cuda.is_available():
            print("GPU is available")
            FORCE_CUDA = True

            # Получаем версию CUDA
            cuda_version = torch.version.cuda
            print(f"Версия CUDA: {cuda_version}")

            # Получаем индекс первого доступного устройства
            index = torch.cuda.current_device()

            # Получаем свойства устройства
            props = torch.cuda.get_device_properties(index)
            # print("props", props)

            # Выводим информацию об устройстве
            print(f"Название видеокарты: {props.name}")
            # print(f"Количество мультипроцессоров: {props.multi_processor_count}")
            print(f"Объем оперативной памяти: {props.total_memory // (1024 ** 2)} МБ")
            print(f"CUDA compatibility: {props.major}.{props.minor}")

        else:
            print("GPU is not available")
            FORCE_CUDA = False

        # Инициализируем модель
        model = None
        try:
            print("Запускаем функцию загрузки модели")
            model = sam2_model.get_model_sam2(s.SAM2_config_file,
                                              s.SAM2_checkpoint_file,
                                              force_cuda=FORCE_CUDA,
                                              verbose=s.VERBOSE)
        except:
            print("Ошибка при загрузке модели")
        #
        if model is not None:
            print("Проверка установки успешно завершена")
        else:
            print("Проверьте установку программного обеспечения")
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
            image_bgr_original = img.copy()
            image_rgb_original = cv.cvtColor(image_bgr_original, cv.COLOR_BGR2RGB)
            print("Загрузили изображение размерности: {}".format(image_bgr_original.shape))

            # Ресайз к номинальному разрешению
            image_bgr = w.resize_image(image_bgr_original, 1024)
            image_rgb = w.resize_image(image_rgb_original, 1024)
            print("Ресайз изображения: {} -> {}".format(image_bgr_original.shape, image_bgr.shape))

            # Инициализация генератора масок
            mask_generator = sam2_model.get_mask_generator(t.get_tool_by_name('model_sam2',
                                                                              tool_list=Tool_list).model,
                                                           verbose=s.VERBOSE)
            # Запуск генератора масок
            start_time = time.time()
            sam2_result = mask_generator.generate(image_rgb)
            end_time = time.time()
            print("Получили список масок: {}, время {:.2f} c.".format(len(sam2_result), end_time - start_time))

            # Сортируем результаты по увеличению площади маски
            sam2_result_sorted = sorted(sam2_result, key=lambda x: x['area'], reverse=False)

            # Отбираем не пересекающиеся маски (по установленному порогу)
            non_overlapping_result = w.find_non_overlapping_masks(sam2_result_sorted,
                                                                  iou_threshold=s.SAM2_iou_threshold)

            # Отбираем маски площадью не менее заданной
            mask_list = []
            area_min = 100
            area_max = 1024 * 1024 * 0.8

            for res in non_overlapping_result:
                # print("\nsegmentation shape:", res['segmentation'].shape)
                # sv.plot_image(res['segmentation'], size=(6, 6))

                if area_min < res['area'] < area_max:
                    mask_list.append(res['segmentation'])
                    print("Взяли маску площадью: {}".format(res['area']))
                else:
                    print("Пропустили маску площадью: {}".format(res['area']))
            print("Собрали список масок: {}".format(len(mask_list)))

            # Формирование выходной маски объединением отобранных
            height, width = image_rgb.shape[:2]
            combined_mask = np.zeros((height, width), dtype=bool)
            #
            for mask in mask_list:
                combined_mask = np.logical_or(combined_mask, mask)
            print("Сформировали выходную маску размерности {}".format(combined_mask.shape))
            # sv.plot_image(combined_mask, size=(12, 12))

            # Сохраняем результат в локальное хранилище в оригинальном разрешении
            H, W = image_bgr_original.shape[:2]
            print("Оригинальное разрешение: {}".format((H, W)))

            result_image = w.convert_mask_to_image(combined_mask)

            # result_image_original = cv2.resize(result_image, (W, H), interpolation=cv.INTER_CUBIC)
            result_image_original = cv.resize(result_image, (W, H), interpolation=cv.INTER_LANCZOS4)

            out_file_name = os.path.join(out_path, 'result_mask_1024.jpg')
            cv.imwrite(out_file_name, result_image_original)

            # Проверим имеющийся набор масок в разрешении 1024
            mask1024_list = mask_list.copy()

            if len(mask1024_list) > 0:
                print("Имеем набор {} масок в разрешении {}".format(len(mask1024_list), mask1024_list[0].shape))
            else:
                print("Список масок пуст, необходимо выполнить предикт в разрешении 1024")

            # Рассчитаем центры масс масок
            center_of_mass_list = []
            for mask1024 in mask1024_list:
                binary_mask = mask1024.astype(np.uint8)
                center_of_mass_list.append(w.compute_center_of_mass_cv(binary_mask))

            # Визуализируем рассчитанные центры масс в разрешении 1024
            image1024 = result_image.copy()
            for center in center_of_mass_list:
                X, Y = center
                X = int(X)
                Y = int(Y)
                image1024 = cv.circle(image1024, (X, Y), 5, (0, 0, 255), -1)
            # u.show_image_cv(image1024, title=str(image1024.shape))

            # Пересчитываем центры масс к оригинальному разрешению и визуализируем
            H, W = image_bgr_original.shape[:2]
            print("Оригинальное разрешение: {}".format((H, W)))

            hh, ww = image_bgr.shape[:2]
            print("Разрешение приведенное к 1024: {}".format((hh, ww)))

            image_original_parsed = result_image_original.copy()

            center_of_mass_original_list = []
            for center in center_of_mass_list:
                X_1024, Y_1024 = center
                X = X_1024 * W / ww
                Y = Y_1024 * H / hh
                X = int(X)
                Y = int(Y)
                center_of_mass_original_list.append((X, Y))

                radius = int(5 * W / 1024)
                image_original_parsed = cv.circle(image_original_parsed, (X, Y), radius, (255, 0, 0), -1)
            # u.show_image_cv(u.img_resize_cv(image_original_parsed, img_size=1024), title=str(image_original_parsed.shape))

            # ПАРАМЕТРЫ
            SCORE_TRESHOLD = s.SAM2_score_threshold  # Ниже какого уровня score применять алгоритм выбора

            # Инициализация предиктора
            predictor = sam2_model.get_predictor(t.get_tool_by_name('model_sam2',
                                                                    tool_list=Tool_list).model,
                                                 verbose=s.VERBOSE)

            #
            H, W = image_bgr_original.shape[:2]
            print("Оригинальное разрешение: {}".format((H, W)))

            hh, ww = image_bgr.shape[:2]
            print("Разрешение приведенное к 1024: {}".format((hh, ww)))

            gap = int(512 * max(hh, ww) / max(H, W))
            print("Размер шага, на который отступать от центра масс масок низкого разрешения: {}".format(gap))

            # Получим список кропов масок около центров масс в разрешении 1024
            c1 = 0
            mask1024_center_list = []
            for idx, center in enumerate(center_of_mass_list[:]):
                Xc, Yc = center
                X1 = Xc - gap if Xc > gap else 0
                Y1 = Yc - gap if Yc > gap else 0
                X2 = Xc + gap if Xc < ww - gap else ww
                Y2 = Yc + gap if Yc < hh - gap else hh
                # print((Xc, Yc), (X1, Y1), (X2, Y2))

                # Берем изображение фрагмента маски вокруг центра
                mask1024 = mask1024_list[idx]
                mask1024_center_img = mask1024[Y1:Y2, X1:X2].copy()
                mask1024_center_list.append(mask1024_center_img)
                # print(mask1024_center_img.shape)
                c1 += 1
            print("Обработали {} масок".format(c1))

            #
            ccc = 0
            mask_promted_list = []
            mask_promted_coord_list = []
            for idx, center_original in enumerate(center_of_mass_original_list[:]):
                Xc, Yc = center_original
                X1 = Xc - 512 if Xc > 512 else 0
                Y1 = Yc - 512 if Yc > 512 else 0
                X2 = Xc + 512 if Xc < W - 512 else W
                Y2 = Yc + 512 if Yc < H - 512 else H
                mask_promted_coord_list.append([X1, Y1, X2, Y2])

                # Берем изображение фрагмента текстуры вокруг центра
                image_center = image_rgb_original[Y1:Y2, X1:X2].copy()
                # print(image_center.shape)

                # Заносим изображение в модель
                predictor.set_image(image_center)

                Xp = Xc - X1
                Yp = Yc - Y1
                promt_point = (Xp, Yp)
                # print(promt_point)

                input_point = np.array([promt_point])
                # print(input_point)

                input_label = np.ones(input_point.shape[0])
                # print(input_label)

                masks, scores, logits = predictor.predict(point_coords=input_point,
                                                          point_labels=input_label,
                                                          multimask_output=True)
                # print(masks.shape)
                # print(scores.shape)

                # Определяем лучший sacore в предикте
                best_idx = np.argmax(scores)
                best_score = scores[best_idx]

                if best_score >= SCORE_TRESHOLD:
                    # Выше порога SCORE_TRESHOLD выбираем маску автоматически
                    mask_promted_list.append(masks[best_idx])
                    # print(mask_promted_list[-1].shape)

                    #
                    ccc += 1
                    print("По score выбрана маска {} из {}; центр: {}, размер: {}, score: {:.2f}".format(ccc,
                                                                                                         len(center_of_mass_original_list),
                                                                                                         (Xc, Yc),
                                                                                                         mask_promted_list[-1].shape,
                                                                                                         best_score))
                else:
                    # Ниже порога SCORE_TRESHOLD отбираем по сходству с маской в разрешении 1024
                    print(
                        "\n=============================================== BEST SCORE: {:.2f} ================================================".format(
                            best_score))

                    # Изображение центра на фрагменте текстуры
                    image_center_parsed = image_center.copy()
                    image_center_parsed = cv.cvtColor(image_center_parsed, cv.COLOR_RGB2BGR)
                    image_center_parsed = cv.circle(image_center_parsed, (Xc - X1, Yc - Y1), 15, (0, 0, 255), -1)
                    # sv.plot_image(image_center_parsed, size=(3,3))

                    # Берем изображение фрагмента маски в разрешении 1024
                    mask1024_center = mask1024_center_list[idx]
                    # Делаем ресайз к размеру маски в оригинальном изображении
                    Hc, Wc = image_center.shape[:2]
                    mask_center_resized = mask1024_center.astype(np.uint8) * 255
                    mask_center_resized = cv.resize(mask_center_resized, (Wc, Hc), interpolation=cv.INTER_LANCZOS4)

                    # Изображение центра на фрагменте маски, полученной ресайзом
                    mask_center_parsed = mask_center_resized.copy()
                    mask_center_parsed = cv.cvtColor(mask_center_parsed, cv.COLOR_GRAY2BGR)
                    mask_center_parsed = cv.circle(mask_center_parsed, (Xc - X1, Yc - Y1), 15, (0, 0, 255), -1)
                    # sv.plot_image(image_center_parsed, size=(3,3))

                    # Три сгенерированные маски, исходное и распарсенное изображения
                    # sv.plot_images_grid(
                    #     images=[image_center_parsed, mask_center_parsed] + list(masks),
                    #     titles=["(Xc, Yc)={}".format((Xc, Yc)), "Low res mask"] + [f"score: {score:.2f}" for score in
                    #                                                                scores],
                    #     grid_size=(1, 5),
                    #     size=(12, 6))

                    #
                    time.sleep(1)

                    # Сравниваем маски
                    IoU_list = []
                    for mask in masks:
                        IoU_list.append(w.calculate_mask_iou(mask_center_resized, mask))
                    print("Выбор лучшей маски по максимальному IoU: {}".format(IoU_list))
                    best_iou_idx = IoU_list.index(max(IoU_list))

                    mask_promted_list.append(masks[best_iou_idx])
                    # print(mask_promted_list[-1].shape)

                    #
                    ccc += 1
                    # sv.plot_image(mask_promted_list[-1], size=(3, 3))
                    print("По IoU выбрана маска {} из {}; центр: {}, размер: {}, score: {:.2f}".format(ccc, len(center_of_mass_original_list), (Xc, Yc), mask_promted_list[-1].shape, scores[best_iou_idx]))


            # Собираем выходную маску в оригинальном разрешении
            combined_original_mask = np.zeros((H, W), dtype=bool)

            for idx, mask in enumerate(mask_promted_list):
                X1, Y1, X2, Y2 = mask_promted_coord_list[idx]
                # print(X1, Y1, X2, Y2)

                mask_temp = np.zeros((H, W), dtype=bool)
                mask_temp[Y1:Y2, X1:X2] = mask

                combined_original_mask = np.logical_or(combined_original_mask, mask_temp)

            print(combined_original_mask.shape)
            # sv.plot_image(combined_original_mask, size=(12, 12))

            # Сохраняем результат в локальное хранилище в оригинальном разрешении
            result_image_final = w.convert_mask_to_image(combined_original_mask)
            # u.show_image_cv(u.img_resize_cv(result_image_final, img_size=1024), title=str(result_image_final.shape))

            file_name = "result_mask_{}x{}.jpg".format(result_image.shape[0], result_image.shape[1])
            out_file_name = os.path.join(out_path, file_name)
            if cv.imwrite(out_file_name, result_image_final):
                print("Успешно записан файл: {}".format(out_file_name))

            # start_time_process = time.perf_counter()
            # print("{}Отработал детектор текста craft, время {:.3f} с.{}".format(s.MAGENTA_cons,
            #                                                                     time.perf_counter() - start_time_qr,
            #                                                                     s.RESET_cons))
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
