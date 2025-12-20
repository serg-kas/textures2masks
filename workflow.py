"""
Функции рабочего процесса
"""
import numpy as np
import cv2 as cv
import random
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

# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# import config
import settings as s
import tool_case as t
# import helpers.log_utils as l
import helpers.utils as u
import helpers.adaptive_mask_filter as m
import sam2_model

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


def find_non_overlapping_masks(data,
                               iou_threshold=0.5):
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
    print("{}  Отбросили {} пересекающихся масок".format(s.CR_CLEAR_cons, counter_overlapping))
    print("{}  Нашли {} не пересекающихся масок".format(s.CR_CLEAR_cons, len(non_overlapping_list)))
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


def baseline(img,
             tool_list=None,
             quick_exit=False,
             verbose=False):
    """
    Алгоритм обработки изображения через центры масс.
    Если quick_exit=True, то выходит после формирования маски в оригинальном разрешении через ресайз от 1024

    TODO: возвращать не только изображения, но и маски и/или другие данные

    """
    if verbose:
        print("Алгоритм BaseLine обработки через центры масс")

    # TODO: Получаем модель (если модели нет поднимать ошибку?)
    tool_model_sam2 = t.get_tool_by_name('model_sam2', tool_list=tool_list)

    # Исходное изображение
    image_bgr_original = img.copy()
    image_rgb_original = cv.cvtColor(image_bgr_original, cv.COLOR_BGR2RGB)
    if verbose:
        print("Обрабатываем изображение размерности: {}".format(image_bgr_original.shape))

    # Сохраним размеры оригинального изображения
    H, W = image_bgr_original.shape[:2]
    if verbose:
        print("Сохранили оригинальное разрешение: Н={}, W={}".format(H, W))

    # Ресайз к номинальному разрешению 1024
    image_bgr = u.resize_image_cv(image_bgr_original, 1024)
    image_rgb = u.resize_image_cv(image_rgb_original, 1024)
    if verbose:
        print("Ресайз изображения: {} -> {}".format(image_bgr_original.shape, image_bgr.shape))

    # Сохраним размеры изображения номинального разрешения 1024
    height, width = image_rgb.shape[:2]

    # Инициализация генератора масок
    mask_generator = sam2_model.get_mask_generator(tool_model_sam2.model,
                                                   verbose=s.VERBOSE)
    if mask_generator is not None:
        if verbose:
            print("  Успешно инициализирован генератор масок")
    else:
        pass
        # TODO: если нет модели, выйти с ошибкой или пустым результатом

    # Запуск генератора масок на изображении разрешения 1024
    if verbose:
        print("Запускаем генератор масок на изображении в разрешении 1024")
    start_time = time.time()
    sam2_result = mask_generator.generate(image_rgb)
    end_time = time.time()
    tool_model_sam2.counter += 1  # счетчик использования модели
    if verbose:
        print("  Получили список масок: {}, время {:.2f} c.".format(len(sam2_result), end_time - start_time))

    # Сортируем результаты по увеличению площади маски
    sam2_result_sorted = sorted(sam2_result,
                                key=lambda x: x['area'],
                                reverse=False)

    # Отбираем не пересекающиеся маски по установленному порогу
    non_overlapping_result = find_non_overlapping_masks(sam2_result_sorted,  #  TODO: проверить алгоритм
                                                        iou_threshold=s.SAM2_iou_threshold)

    # Отбираем маски по площади
    if verbose:
        print("Отбираем маски по площади")
    area_min, area_max = s.AREA_MIN, s.AREA_MAX  # пределы из настроек
    if s.AUTO_CALCULATE_AREAS:
        areas = [res['area'] for res in non_overlapping_result]
        if areas:
            # Статистика площадей
            if verbose:
                print("  Статистика площадей: min={:.2f}, max={:.2f}, median={:.2f}".format(min(areas),
                                                                                            max(areas),
                                                                                            np.median(areas)))
            # area_min, area_max = u.calculate_area_thresholds(areas,iqr_multiplier=1.5)
            # area_min, area_max = u.calculate_area_thresholds_quantile(areas,lower_quantile=0.05,upper_quantile=0.95)
            area_min, area_max = m.adaptive_area_thresholds(areas, sensitivity='medium')

            if verbose:
                # Показываем распределение
                hist, bins = np.histogram(areas, bins=min(10, len(areas)))
                print("  Распределение площадей:")
                for i in range(len(hist)):
                    print(f"    {bins[i]:.1f}-{bins[i + 1]:.1f}: {hist[i]} масок")
                print("  Автоматически рассчитаны пороги: от {:.2f} до {:.2f}".format(area_min, area_max))
                print("  Будет отобрано масок в диапазоне: {} из {}".format(sum(area_min <= a <= area_max for a in areas),
                                                                            len(areas)))
    #
    else:
        if verbose:
            print("  Берем пределы из настроек: min={}, max={}".format(area_min, area_max))
    #
    mask_list = []
    for res in non_overlapping_result:
        if area_min <= res['area'] <= area_max:
            mask_list.append(res['segmentation'])
            print("\r  Взяли маску площадью: {}".format(res['area']), end="")
        else:
            print("\r  Пропустили маску площадью: {}".format(res['area']), end="")
    print("{}  Собрали список масок в разрешении 1024: {}".format(s.CR_CLEAR_cons, len(mask_list)))

    # Формируем выходную маску объединением отобранных
    if verbose:
        print("Формируем выходную маску в разрешении 1024 объединением отобранных")
    combined_mask = np.zeros((height, width), dtype=bool)
    for mask in mask_list:
        combined_mask = np.logical_or(combined_mask, mask)
    if verbose:
        print("  Сформировали выходную маску размерности: {}".format(combined_mask.shape))

    # Ресайз к оригинальному разрешению маски, полученной через разрешение 1024
    result_mask1024 = convert_mask_to_image(combined_mask)
    result_mask1024_original_size = cv.resize(result_mask1024, (W, H),
                                              interpolation=cv.INTER_LANCZOS4)  # cv.INTER_CUBIC

    # Имеющийся набор масок в разрешении 1024
    mask1024_list = mask_list.copy()

    # Рассчитаем центры масс масок
    center_of_mass_list = []
    for mask1024 in mask1024_list:
        binary_mask = mask1024.astype(np.uint8)
        center_of_mass_list.append(compute_center_of_mass_cv(binary_mask))
    if verbose:
        print("Рассчитали центры масс в разрешении 1024: {}".format(len(center_of_mass_list)))

    # Максимальный размер маски в разрешении 1024, которая поместится в 1024 пиксела в оригинальном разрешении
    max_mask_size_1024 = int(1024 * max(height, width) / max(H, W))
    print("  Максимальный размер маски в разрешении 1024, "
          "которая поместится в окно 1024 пиксела в оригинальном разрешении: {}".format(max_mask_size_1024))

    # Визуализируем рассчитанные центры масс в разрешении 1024
    result_mask1024_centers = result_mask1024.copy()
    for idx, center in enumerate(center_of_mass_list[:]):
        X, Y = center
        X = int(X)
        Y = int(Y)
        mask_w, mask_h = calculate_mask_dimensions(mask1024_list[idx])
        mask_size = max(mask_w, mask_h)
        if mask_size <= max_mask_size_1024:
            result_mask1024_centers = cv.circle(result_mask1024_centers, (X, Y), 5, s.green, -1)
        else:
            result_mask1024_centers = cv.circle(result_mask1024_centers, (X, Y), 5, s.red, -1)
    # u.show_image_cv(result_mask1024_centers, title=str(result_mask1024_centers.shape))

    # Пересчитываем центры масс к оригинальному разрешению
    # result_mask_original_centers = result_mask1024_original_size.copy() # TODO: Визуализацию не используем
    center_of_mass_original_list = []
    for center in center_of_mass_list:
        X_1024, Y_1024 = center
        X = X_1024 * W / width
        Y = Y_1024 * H / height
        X = int(X)
        Y = int(Y)
        center_of_mass_original_list.append((X, Y))
        # radius = int(5 * W / 1024)
        # result_mask_original_centers = cv.circle(result_mask_original_centers, (X, Y), radius, s.blue, -1)
    # u.show_image_cv(u.resize_image_cv(result_mask_original_centers, img_size=1024), title=str(result_mask_original_centers.shape))
    if verbose:
        print("  Пересчитали центры масс к оригинальному разрешению: {}".format(len(center_of_mass_original_list)))

    # Быстрое окончание без формирования финальной маски
    if quick_exit:
        if verbose:
            print("Быстрый выход без формирования финальной маски в оригинальном разрешении")
        return {
            "result_mask1024": result_mask1024,                              # маска в разрешении 1024
            "result_mask1024_centers": result_mask1024_centers,              # визуализация центров масс в разрешении 1024
            "result_mask1024_original_size": result_mask1024_original_size,  # маска в оригинальном разрешении, полученная ресайзом из 1024
            'center_of_mass_original_list': center_of_mass_original_list,    # центры масс масок, пересчитанные в оригинальное разрешение
            "result_image_final": None                                       # финальная маска в оригинальном разрешении НЕ СОЗДАВАЛАСЬ
        }

    # Инициализация предиктора
    predictor = sam2_model.get_predictor(tool_model_sam2.model, verbose=s.VERBOSE)
    if predictor is not None:
        if verbose:
            print("  Предиктор инициализирован успешно")
    else:
        pass
        # TODO: если нет предиктора, выйти с ошибкой или пустым результатом

    # Получим список кропов масок около центров масс в разрешении 1024
    if verbose:
        print("Готовим список кропов масок вокруг центров в разрешении 1024")
    #
    gap = int(512 * max(height, width) / max(H, W))
    if verbose:
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
    if verbose:
        print("  Собрали кропов вокруг центров в разрешении 1024: {}".format(counter_crop_1024))

    # Сегментация в оригинальном разрешении вокруг центров
    if verbose:
        print("Сегментация в оригинальном разрешении вокруг рассчитанных центров")

    # Ниже какого уровня score применять алгоритм выбора
    SCORE_TRESHOLD = s.SAM2_score_threshold
    if verbose:
        print("  Тресхолд score, ниже которого применяется алгоритм отбора масок по IoU: {}".format(SCORE_TRESHOLD))

    # Цикл обработки кропов
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

        # Заносим изображение в модель
        predictor.set_image(image_center)

        # Координаты центра масс, пересчитанные к image_center (кроп изображения)
        Xp = Xc - X1
        Yp = Yc - Y1

        # Промт - одна точка в центре масс маски
        if s.PROMPT_POINT_RADIUS == 0:
            # if verbose:
            #     print("\n  Промт - одна точка в центре масс маски: {}".format((Xc, Yc)))

            #
            promt_point = (Xp, Yp)
            #
            input_point = np.array([promt_point])
            input_label = np.ones(input_point.shape[0])

        # Промт - несколько точек в заданном радиусе от центра масс
        else:
            # if verbose:
            #     print("\n  Промт - {} точек в радиусе {} от центра масс маски: {}".format(s.PROMPT_POINT_NUMBER,
            #                                                                               s.PROMPT_POINT_RADIUS,
            #                                                                               (Xc, Yc)))

            # Фильтруем точки в заданном радиусе от центра
            radius_points_list = u.get_points_in_radius(image_center.shape,
                                                        (Xp, Yp),
                                                        s.PROMPT_POINT_RADIUS)

            # Фильтруем точки по среднему цвету
            if s.PROMPT_POINT_COLOR_FILTER:
                avg_color_bgr, radius_points_list = u.calculate_average_color_with_outliers(image_center,
                                                                                            radius_points_list,
                                                                                            color_threshold=s.PROMPT_POINT_COLOR_THRESH)
                # if verbose:
                #     print("\n  Точки отфильтрованы по цвету вокруг цвета, (b, g, r): {}".format(avg_color_bgr))

            # Оставляем заданное количество точек
            promt_point_list = random.sample(radius_points_list, s.PROMPT_POINT_NUMBER)
            #
            if len(promt_point_list) == 0:
                promt_point_list = [(Xp, Yp)]
            #
            input_point = np.array(promt_point_list)
            input_label = np.ones(input_point.shape[0])

        # Предикт
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

            if verbose:
                print(
                    "{}  Центр {}/{}. Промт точек: {}. По score выбрана маска: центр: {}, размер: {}, score: {:.3f}".format(
                        s.CR_CLEAR_cons,
                        idx + 1,
                        len(center_of_mass_original_list),
                        input_point.shape[0],
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
                IoU_list.append(calculate_mask_iou(mask_center_resized, mask))

            # print("Выбор лучшей маски по максимальному IoU: {}".format(IoU_list))
            best_iou_idx = IoU_list.index(max(IoU_list))
            mask_promted_list.append(masks[best_iou_idx])
            # print(mask_promted_list[-1].shape)

            if verbose:
                print(
                    "{}  Центр {}/{}. Промт точек: {}. По IoU выбрана маска: центр: {}, размер: {}, score: {:.3f}".format(
                        s.CR_CLEAR_cons,
                        idx + 1,
                        len(center_of_mass_original_list),
                        input_point.shape[0],
                        (Xc, Yc),
                        mask_promted_list[-1].shape,
                        scores[best_iou_idx]),
                end="")
            #
            counter_crop += 1
    if verbose:
        print("\n  Всего обработано кропов: {}".format(counter_crop))

    # Собираем выходную маску в оригинальном разрешении
    combined_original_mask = np.zeros((H, W), dtype=bool)

    for idx, mask in enumerate(mask_promted_list):
        X1, Y1, X2, Y2 = mask_promted_coord_list[idx]
        # print(X1, Y1, X2, Y2)

        mask_temp = np.zeros((H, W), dtype=bool)
        mask_temp[Y1:Y2, X1:X2] = mask

        combined_original_mask = np.logical_or(combined_original_mask, mask_temp)
    if verbose:
        print("Собрали комбинированную выходную маску в оригинальном разрешении: {}".format(combined_original_mask.shape))

    # Результат обработки в оригинальном разрешении
    result_image_final = convert_mask_to_image(combined_original_mask)
    # u.show_image_cv(u.img_resize_cv(result_image_final, img_size=1024), title=str(result_image_final.shape))

    return {
        "result_mask1024": result_mask1024,                             # маска в разрешении 1024
        "result_mask1024_centers": result_mask1024_centers,             # визуализация центров масс в разрешении 1024
        "result_mask1024_original_size": result_mask1024_original_size, # маска в оригинальном разрешении, полученная ресайзом из 1024
        'center_of_mass_original_list': center_of_mass_original_list,   # центры масс масок, пересчитанные в оригинальное разрешение
        "result_image_final": result_image_final                        # финальная маска в оригинальном разрешении
    }


def split_into_tiles(image,
                     tile_size=1024,
                     overlap=256):
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


def assemble_image(tiles,
                   coords,
                   original_shape,
                   overlap):
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
