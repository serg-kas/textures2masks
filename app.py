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
            print("GPU доступен, получаем информацию:")
            TEST_CUDA = True

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
            print("GPU недоступен")
            TEST_CUDA = False

        if TEST_CUDA:
            # Инициализируем модель на GPU
            model = None
            try:
                print("Запускаем функцию загрузки модели на GPU:")
                model = sam2_model.get_model_sam2(s.SAM2_config_file,
                                                  s.SAM2_checkpoint_file,
                                                  force_cuda=True,
                                                  verbose=s.VERBOSE)
            except:
                print("  Ошибка при загрузке модели")
            #
            if model is not None:
                print("  Модель успешно загрузилась на GPU")
            else:
                print("  Загрузка модели на GPU не удалась")

            # Инициализируем модель на CPU
            model = None
            try:
                print("Запускаем функцию загрузки модели на CPU:")
                model = sam2_model.get_model_sam2(s.SAM2_config_file,
                                                  s.SAM2_checkpoint_file,
                                                  force_cuda=False,
                                                  verbose=s.VERBOSE)
            except:
                print("  Ошибка при загрузке модели")
            #
            if model is not None:
                print("  Модель успешно загрузилась на CPU")
            else:
                print("  Загрузка модели на CPU не удалась")

        # TODO: тестовый запуск модели на фейковых данных ?
        #
        time_end = time.time()
        print("Общее время выполнения: {:.1f} с.".format(time_end - time_start))


    # #################### workflow_masks #####№№################
    # Рабочий режим обработки изображения с созданием выходной маски
    # #########################################################
    if operation_mode == 'workflow_masks':
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
        #
        tool_model_sam2 = t.get_tool_by_name('model_sam2', tool_list=Tool_list)

        # #############################################
        # Обрабатываем файлы из списка
        # #############################################
        print(u.txt_separator('=', s.CONS_COLUMNS,
                              txt=' Обрабатываем файлы из списка ', txt_align='center'))
        #
        time_0 = time.perf_counter()
        counter_img = 0
        for img_idx, img in enumerate(img_list):
            # Имя обрабатываемого файла изображения
            img_file = img_file_list[img_idx]
            img_file_base_name = os.path.basename(img_file)
            print("Обрабатываем изображение из файла: {}".format(img_file_base_name))
            #
            image_bgr_original = img.copy()
            image_rgb_original = cv.cvtColor(image_bgr_original, cv.COLOR_BGR2RGB)
            print("Загрузили изображение размерности: {}".format(image_bgr_original.shape))

            # Сохраним размеры оригинального изображения
            H, W = image_bgr_original.shape[:2]
            print("Сохранили оригинальное разрешение: {}".format((H, W)))

            # Ресайз к номинальному разрешению 1024
            image_bgr = u.img_resize_cv(image_bgr_original, 1024)
            image_rgb = u.img_resize_cv(image_rgb_original, 1024)
            print("Ресайз изображения: {} -> {}".format(image_bgr_original.shape, image_bgr.shape))

            # Сохраним размеры изображения номинального разрешения 1024
            height, width = image_rgb.shape[:2]

            # Инициализация генератора масок
            mask_generator = sam2_model.get_mask_generator(tool_model_sam2.model, verbose=s.VERBOSE)
            if mask_generator is not None:
                print("Успешно инициализирован генератор масок")

            # Запуск генератора масок на изображении 1024
            start_time = time.time()
            sam2_result = mask_generator.generate(image_rgb)
            end_time = time.time()
            tool_model_sam2.counter += 1
            print("Получили список масок: {}, время {:.2f} c.".format(len(sam2_result), end_time - start_time))

            # Сортируем результаты по увеличению площади маски
            sam2_result_sorted = sorted(sam2_result, key=lambda x: x['area'], reverse=False)

            # Отбираем не пересекающиеся маски (по установленному порогу)
            non_overlapping_result = w.find_non_overlapping_masks(sam2_result_sorted,
                                                                  iou_threshold=s.SAM2_iou_threshold)

            # Отбираем маски площадью в заданных пределах
            area_min = s.AREA_MIN
            area_max = s.AREA_MAX
            print("Фильтруем маски по площади от {} до {}".format(area_min, area_max))

            mask_list = []
            for res in non_overlapping_result:
                if area_min < res['area'] < area_max:
                    mask_list.append(res['segmentation'])
                    print("\r  Взяли маску площадью: {}".format(res['area']), end="")
                else:
                    print("\r  Пропустили маску площадью: {}".format(res['area']), end="")
            print("{}Собрали список масок в разрешении 1024: {}".format(s.CR_CLEAR_cons, len(mask_list)))

            # Формируем выходную маску объединением отобранных
            print("Формируем выходную маску объединением отобранных")
            combined_mask = np.zeros((height, width), dtype=bool)
            for mask in mask_list:
                combined_mask = np.logical_or(combined_mask, mask)
            print("Сформировали выходную маску размерности: {}".format(combined_mask.shape))

            # Ресайз к оригинальному разрешению и сохранение маски, полученной через раз
            result_mask1024 = w.convert_mask_to_image(combined_mask)
            result_mask1024_original_size = cv.resize(result_mask1024, (W, H), interpolation=cv.INTER_LANCZOS4)  # cv.INTER_CUBIC

            # Имя выходного файла в оригинальном разрешении
            out_img_base_name_mask1024 = img_file_base_name[:-4] + "_mask_1024.jpg"
            # Полный путь к выходному файлу
            out_img_file_mask1024 = os.path.join(out_path, out_img_base_name_mask1024)

            # if cv.imwrite(result_mask1024_original_size, out_img_file_mask1024):
            #     print("Сохранили в оригинальном разрешении маску, полученную через ресайз от 1024: {}".format(out_img_file_mask1024))

            # Запись изображения
            try:
                success = cv.imwrite(out_img_file_mask1024, result_mask1024_original_size)
                if not success:
                    print(f'Не удалось сохранить файл {out_img_file_mask1024}')
            except Exception as e:
                print(f'Произошла ошибка при сохранении файла: {e}')

            # Имеющийся набор масок в разрешении 1024
            mask1024_list = mask_list.copy()
            # if len(mask1024_list) > 0:
            #     print("Имеем набор {} масок в разрешении {}".format(len(mask1024_list), mask1024_list[0].shape))

            # Рассчитаем центры масс масок
            center_of_mass_list = []
            for mask1024 in mask1024_list:
                binary_mask = mask1024.astype(np.uint8)
                center_of_mass_list.append(w.compute_center_of_mass_cv(binary_mask))
            print("Рассчитали центры масс в разрешении 1024: {}".format(len(mask1024_list)))

            # Визуализируем рассчитанные центры масс в разрешении 1024
            # result_mask1024_centers = result_mask1024.copy()
            # for center in center_of_mass_list:
            #     X, Y = center
            #     X = int(X)
            #     Y = int(Y)
            #     result_mask1024_centers = cv.circle(result_mask1024_centers, (X, Y), 5, s.red, -1)
            # u.show_image_cv(result_mask1024_centers, title=str(result_mask1024_centers.shape))

            # Пересчитываем центры масс к оригинальному разрешению
            print("Пересчитываем центры масс к оригинальному разрешению")
            result_mask_original_centers = result_mask1024_original_size.copy()
            center_of_mass_original_list = []
            for center in center_of_mass_list:
                X_1024, Y_1024 = center
                X = X_1024 * W / width
                Y = Y_1024 * H / height
                X = int(X)
                Y = int(Y)
                center_of_mass_original_list.append((X, Y))
                #
                radius = int(5 * W / 1024)
                result_mask_original_centers = cv.circle(result_mask_original_centers, (X, Y), radius, s.blue, -1)
            # u.show_image_cv(u.img_resize_cv(result_mask_original_centers, img_size=1024), title=str(result_mask_original_centers.shape))

            # Инициализация предиктора
            predictor = sam2_model.get_predictor(tool_model_sam2.model, verbose=s.VERBOSE)
            if predictor is not None:
                print("Предиктор инициализирован успешно")

            # Получим список кропов масок около центров масс в разрешении 1024
            print("Готовим список кропов масок вокруг центров в разрешении 1024")
            #
            gap = int(512 * max(height, width) / max(H, W))
            print("  Размер шага, на который отступать от центра масс масок разрешения 1024: {}".format(gap))
            #
            counter_crop_1024 = 0
            mask1024_center_list = []
            for idx, center in enumerate(center_of_mass_list[:]):
                Xc, Yc = center
                X1 = Xc - gap if Xc > gap else 0
                Y1 = Yc - gap if Yc > gap else 0
                X2 = Xc + gap if Xc < width - gap else width
                Y2 = Yc + gap if Yc < height - gap else height
                # print((Xc, Yc), (X1, Y1), (X2, Y2))

                # Берем кроп маски вокруг центра
                mask1024 = mask1024_list[idx]
                mask1024_center_img = mask1024[Y1:Y2, X1:X2].copy()
                mask1024_center_list.append(mask1024_center_img)
                counter_crop_1024 += 1
            print("Собрали кропов вокруг центров в разрешении 1024: {}".format(counter_crop_1024))

            # Ниже какого уровня score применять алгоритм выбора
            SCORE_TRESHOLD = s.SAM2_score_threshold
            print("Тресхолд score, ниже которого применяется алгоритм отбора масок по IoU: {}".format(SCORE_TRESHOLD))

            # Цикл обработки кропов (сегментация в оригинальном разрешении вокруг центров)
            print("Сегментация в оригинальном разрешении вокруг рассчитанных центров")
            counter_crop = 0
            mask_promted_list = []        # список получаемых масок
            mask_promted_coord_list = []  # список координат кропов на текстуре оригинального разрешения
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

                # Промт
                Xp = Xc - X1
                Yp = Yc - Y1
                promt_point = (Xp, Yp)
                # print(promt_point)
                input_point = np.array([promt_point])
                # print(input_point)
                input_label = np.ones(input_point.shape[0])  # TODO: ?
                # print(input_label)

                masks, scores, logits = predictor.predict(point_coords=input_point,
                                                          point_labels=input_label,
                                                          multimask_output=True)
                tool_model_sam2.counter += 1
                # print(masks.shape, scores.shape)

                # Определяем лучший score в предикте
                best_idx = np.argmax(scores)
                best_score = scores[best_idx]

                # Если score выше порога SCORE_TRESHOLD, то выбираем маску автоматически
                if best_score >= SCORE_TRESHOLD:
                    mask_promted_list.append(masks[best_idx])
                    # print(mask_promted_list[-1].shape)

                    print("{}  По score выбрана маска {} из {}; центр: {}, размер: {}, score: {:.3f}".format(s.CR_CLEAR_cons,
                                                                                                             counter_crop,
                                                                                                             len(center_of_mass_original_list),
                                                                                                             (Xc, Yc),
                                                                                                             mask_promted_list[-1].shape,
                                                                                                             best_score),
                          end="")
                    #
                    counter_crop += 1

                # Если score ниже порога SCORE_TRESHOLD, отбираем маску по IoU с маской в разрешении 1024
                else:
                    # Берем изображение фрагмента маски в разрешении 1024
                    mask1024_center = mask1024_center_list[idx]

                    # Делаем ресайз к размеру маски в оригинальном изображении
                    Hc, Wc = image_center.shape[:2]
                    mask_center_resized = mask1024_center.astype(np.uint8) * 255
                    mask_center_resized = cv.resize(mask_center_resized, (Wc, Hc), interpolation=cv.INTER_LANCZOS4)

                    # Сравниваем маски по IoU
                    IoU_list = []
                    for mask in masks:
                        IoU_list.append(w.calculate_mask_iou(mask_center_resized, mask))

                    # print("Выбор лучшей маски по максимальному IoU: {}".format(IoU_list))
                    best_iou_idx = IoU_list.index(max(IoU_list))
                    mask_promted_list.append(masks[best_iou_idx])
                    # print(mask_promted_list[-1].shape)

                    print("{}  По IoU выбрана маска {} из {}; центр: {}, размер: {}, score: {:.3f}".format(s.CR_CLEAR_cons,
                                                                                                           counter_crop,
                                                                                                           len(center_of_mass_original_list),
                                                                                                           (Xc, Yc),
                                                                                                           mask_promted_list[-1].shape,
                                                                                                           scores[best_iou_idx]))
                    #
                    counter_crop += 1

            # Собираем выходную маску в оригинальном разрешении
            combined_original_mask = np.zeros((H, W), dtype=bool)

            for idx, mask in enumerate(mask_promted_list):
                X1, Y1, X2, Y2 = mask_promted_coord_list[idx]
                # print(X1, Y1, X2, Y2)

                mask_temp = np.zeros((H, W), dtype=bool)
                mask_temp[Y1:Y2, X1:X2] = mask

                combined_original_mask = np.logical_or(combined_original_mask, mask_temp)
            print("\nСобрали выходную маску в оригинальном разрешении: {}".format(combined_original_mask.shape))

            # Сохраняем результат в локальное хранилище в оригинальном разрешении
            result_image_final = w.convert_mask_to_image(combined_original_mask)
            # u.show_image_cv(u.img_resize_cv(result_image_final, img_size=1024), title=str(result_image_final.shape))

            # Имя выходного файла в оригинальном разрешении
            out_img_base_name_original_size = img_file_base_name[:-4] + "_mask_W{}xH{}.jpg".format(result_image_final.shape[1],
                                                                                                   result_image_final.shape[0])
            # Полный путь к выходному файлу
            out_img_file_original_size = os.path.join(out_path, out_img_base_name_original_size)

            # Запись изображения
            try:
                success = cv.imwrite(out_img_file_original_size, result_image_final)
                if not success:
                    print(f'Не удалось сохранить файл {out_img_file_original_size}')
            except Exception as e:
                print(f'Произошла ошибка при сохранении файла: {e}')
            #
            counter_img += 1
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
