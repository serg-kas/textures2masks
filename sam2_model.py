"""
Функции, связанные с вызовом моделей
и обработкой результатов
"""
import numpy as np
import cv2 as cv
import time
#
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
#
# import helpers.utils as u
import settings as s


# #########################################################
# Функции для работы с моделью SAM2
# #########################################################
def get_model_sam2(config_file,
                   model_file,
                   force_cuda=False,
                   verbose=False):
    """
    Загрузка модели из указанного файла

    :param config_file: путь к файлу конфигурации модели для загрузки
    :param model_file: путь к файлу чекпоинта модели для загрузки
    :param force_cuda: использовать CUDA
    :param verbose: выводить дополнительную информацию
    :return: model: загруженная модель
    """
    if verbose:
        print("Загружаем файл конфигурации модели: {}".format(config_file))
        print("Загружаем файл чекпоинта модели: {}".format(model_file))
    time_0 = time.perf_counter()

    #
    DEVICE = torch.device('cpu')
    if force_cuda:
        if torch.cuda.is_available():
            if verbose:
                print("  Found CUDA, trying to use it")
            #
            DEVICE = torch.device('cuda')
            #
            torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
            #
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        else:
            if verbose:
                print("  No CUDA available, using CPU")
    else:
        if verbose:
            print("  Ignoring CUDA, using CPU")
    #
    model = build_sam2(config_file, model_file, device=DEVICE, apply_postprocessing=False)
    #
    time_1 = time.perf_counter()
    if verbose:
        print("  Время загрузки модели, с: {:.2f}".format(time_1 - time_0))
    return model


def get_mask_generator(model, verbose=False):
    """
    Инициализация генератора масок

    :param model: модель
    :param verbose: выводить дополнительную информацию
    :return: model: загруженная модель
    """
    if verbose:
        print("Инициализируем генератор масок:")
    time_0 = time.perf_counter()
    #
    mask_generator = SAM2AutomaticMaskGenerator(model)
    #
    time_1 = time.perf_counter()
    if verbose:
        print("  Время подготовки генератора масок, с: {:.2f}".format(time_1 - time_0))
    return mask_generator


def get_predictor(model, verbose=False):
    """
    Инициализация предиктора

    :param model: модель
    :param verbose: выводить дополнительную информацию
    :return: model: загруженная модель
    """
    if verbose:
        print("Инициализируем предиктор:")
    time_0 = time.perf_counter()
    #
    predictor = SAM2ImagePredictor(model)
    #
    time_1 = time.perf_counter()
    if verbose:
        print("  Время подготовки предиктора, с: {:.2f}".format(time_1 - time_0))
    return predictor


def prepare_prompts_from_mask(mask,
                              num_points=20,
                              min_contour_area=1000,
                              max_contours=10,
                              foreground=True):
    """
    Генерация точечных промптов из маски
    """
    # Находим контуры в маске
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Фильтруем контуры по площади и берем только самые крупные
    filtered_contours = []

    # Вычисляем площади всех контуров
    contour_areas = [(i, cv.contourArea(contour)) for i, contour in enumerate(contours)]

    # Сортируем контуры по площади (по убыванию)
    contour_areas.sort(key=lambda x: x[1], reverse=True)

    # Отбираем контуры, удовлетворяющие критериям
    for i, area in contour_areas:
        if area >= min_contour_area and len(filtered_contours) < max_contours:
            filtered_contours.append(contours[i])

    print(f"    Найдено контуров: {len(contours)}, отфильтровано: {len(filtered_contours)}")
    print(f"    Площади контуров (первые 5): {[f'{area:.0f}' for _, area in contour_areas][:5]}")

    # Создаем копию маски для визуализации
    if len(mask.shape) == 2:  # Если маска одноканальная
        mask_parsed = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
    else:
        mask_parsed = mask.copy()

    point_coords = []
    point_labels = []

    # for contour in contours:
    for contour in filtered_contours:
        # Добавляем точки вдоль контура
        for i in range(0, len(contour), max(1, len(contour) // num_points)):
            point = contour[i][0]
            point_coords.append([point[0], point[1]])
            #
            if foreground:
                point_labels.append(1)
                cv.circle(mask_parsed, (point[0], point[1]), 3, s.red, -1)
            else:
                point_labels.append(0)
                cv.circle(mask_parsed, (point[0], point[1]), 3, s.blue, -1)

        # Добавляем точки внутри области (центроиды)
        if len(contour) > 0:
            M = cv.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                point_coords.append([cx, cy])
                #
                if foreground:
                    point_labels.append(1)
                    cv.circle(mask_parsed, (cx, cy), 5, s.red, -1)
                else:
                    point_labels.append(0)
                    cv.circle(mask_parsed, (cx, cy), 5, s.blue, -1)

    return np.array(point_coords), np.array(point_labels), mask_parsed


# #########################################################
# Функции для работы с моделями DNN opencv - универсальные
# #########################################################
def get_model_dnn(model_file, force_cuda=False, verbose=False):
    """
    Загрузка модели из указанного файла
    :param model_file: путь к модели для загрузки
    :param force_cuda: использовать CUDA
    :param verbose: выводить дополнительную информацию
    :return: model: загруженная модель
    """
    if verbose:
        print('Загружаем модель: {}'.format(model_file))
    time_0 = time.perf_counter()
    #
    model = cv.dnn.readNet(model_file)
    if force_cuda:
        print("  Try to use CUDA")
        model.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        model.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
    else:
        print("  Running on CPU")
        model.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        model.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    #
    time_1 = time.perf_counter()
    if verbose:
        print('  Время загрузки модели, с: {:.2f}'.format(time_1 - time_0))
    return model


def get_blob_dnn(source_img, input_shape=(640, 640), to_gray=False, subtract_mean=False):
    """
    Подготовка изображения для передачи в модель
    :param source_img: исходное изображение
    :param input_shape: размер к которому приводить изображение
    :param to_gray: переходить к ч/б
    :param subtract_mean: центрировать без скейлинга
    :return: обработанное изображение
    """
    # ресайз к img_size, scale factor = 1, subtract mean BGR
    # result = cv.dnn.blobFromImage(source_img, 1, input_shape, (104, 117, 123))

    if to_gray:
        # переход к ч/б, ресайз к img_size и нормализация
        gray = cv.cvtColor(source_img, cv.COLOR_BGR2GRAY)
        # gray = cv.threshold(grey, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
        result = cv.dnn.blobFromImage(gray, 1 / 255.0, input_shape)
        return result

    if subtract_mean:
        # ресайз к img_size, scale factor = 1, subtract mean BGR
        result = cv.dnn.blobFromImage(source_img, 1, input_shape, (104, 117, 123))  # BGR
        # result = cv.dnn.blobFromImage(source_img, 1, input_shape, (123, 117, 103))  # RGB
        return result

    # ресайз к img_size, нормализация и замена между собой R и B каналов (переход к RGB)
    result = cv.dnn.blobFromImage(source_img, 1 / 255.0, input_shape, swapRB=True)
    return result


def get_pred_dnn(model, blob):
    """
    Получение предикта из модели и подготовленного изображения
    :param model: модель
    :param blob: подготовленное изображение
    :return: обработанное изображение
    """
    model.setInput(blob)
    predictions = model.forward()
    return predictions
