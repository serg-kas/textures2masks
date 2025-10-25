"""
Функции различного назначения
"""
import numpy as np
# import math
# import re
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy import stats
# import warnings

# import cv2 as cv
# from PIL import Image  # ImageDraw, ImageFont
# from imutils import perspective, auto_canny
# import matplotlib.pyplot as plt

# import io
# import json
# import base64
# import os
# import sys
# import time
# from datetime import datetime, timezone, timedelta
# import inspect
# import importlib

# import settings as s


# #############################################################
#                       ФУНКЦИИ адаптивной фильтрации масок по размеру кластеризации др.
# #############################################################
# def calculate_area_thresholds(areas, iqr_multiplier=1.5):
#     """
#     Автоматически рассчитывает пороги для фильтрации масок по площади
#     используя межквартильный размах (IQR) для отсечения выбросов
#     """
#     if not areas:
#         return 0, float('inf')
#
#     areas_array = np.array(areas)
#     Q1 = np.percentile(areas_array, 25)
#     Q3 = np.percentile(areas_array, 75)
#     IQR = Q3 - Q1
#
#     # Расчет порогов
#     lower_bound = Q1 - iqr_multiplier * IQR
#     upper_bound = Q3 + iqr_multiplier * IQR
#
#     # Гарантируем, что нижний порог не отрицательный
#     area_min = max(0, lower_bound)
#     area_max = upper_bound
#
#     return area_min, area_max


# def calculate_area_thresholds_quantile(areas,
#                                        lower_quantile=0.05,
#                                        upper_quantile=0.95):
#     """
#     Рассчитывает пороги используя квантили
#     """
#     if not areas:
#         return 0, float('inf')
#
#     areas_array = np.array(areas)
#     area_min = np.quantile(areas_array, lower_quantile)
#     area_max = np.quantile(areas_array, upper_quantile)
#
#     return area_min, area_max


def find_optimal_clusters_elbow(areas, max_clusters=10):
    """Метод локтя для определения оптимального числа кластеров"""
    if len(areas) <= 1:
        return 1

    areas_array = np.array(areas).reshape(-1, 1)
    max_clusters = min(max_clusters, len(areas))

    inertias = []
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(areas_array)
        inertias.append(kmeans.inertia_)

    # Находим "локоть" - точку максимального изгиба
    differences = []
    for i in range(1, len(inertias)):
        differences.append(inertias[i - 1] - inertias[i])

    if differences:
        # Ищем наибольшее изменение (можно улучшить более сложным алгоритмом)
        optimal_k = np.argmax(differences) + 2  # +2 потому что начинаем с k=2
        return min(optimal_k, max_clusters)
    return 1


def cluster_based_thresholds(areas, method='gmm'):
    """
    Определение порогов на основе кластеризации
    methods: 'gmm' (Gaussian Mixture), 'kmeans', 'largest_cluster'
    """
    if len(areas) < 3:
        return min(areas), max(areas)

    areas_array = np.array(areas).reshape(-1, 1)

    if method == 'kmeans':
        # K-means с автоматическим подбором кластеров
        n_clusters = find_optimal_clusters_elbow(areas)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(areas_array)

    elif method == 'gmm':
        # Gaussian Mixture Model - лучше для размеров
        n_components = min(5, len(areas) // 3)  # Не больше 5 кластеров
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        labels = gmm.fit_predict(areas_array)

    elif method == 'largest_cluster':
        # Простая кластеризация по квантилям - берем самый большой кластер
        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(areas_array)

    # Находим основной кластер (самый многочисленный)
    unique_labels, counts = np.unique(labels, return_counts=True)
    main_cluster_label = unique_labels[np.argmax(counts)]

    # Берем площади из основного кластера
    main_cluster_areas = areas_array[labels == main_cluster_label].flatten()

    # Расширяем границы на 10% для надежности
    cluster_min = np.min(main_cluster_areas)
    cluster_max = np.max(main_cluster_areas)
    margin = (cluster_max - cluster_min) * 0.1

    area_min = max(0, cluster_min - margin)
    area_max = cluster_max + margin

    return area_min, area_max


def adaptive_area_thresholds(areas, sensitivity='medium'):
    """
    Адаптивный метод, комбинирующий несколько подходов
    sensitivity: 'high' (строгий), 'medium', 'low' (мягкий)
    """
    if len(areas) < 5:
        return min(areas), max(areas)

    # Пробуем разные методы кластеризации
    methods = ['gmm', 'kmeans']
    results = []

    for method in methods:
        try:
            area_min, area_max = cluster_based_thresholds(areas, method)
            results.append((area_min, area_max))
        except:
            continue

    if not results:
        # Fallback на квантильный метод
        area_min = np.quantile(areas, 0.1)
        area_max = np.quantile(areas, 0.9)
        return area_min, area_max

    # Выбираем результат в зависимости от чувствительности
    if sensitivity == 'high':
        # Берем самый строгий (самые узкие границы)
        idx = np.argmin([max_val - min_val for min_val, max_val in results])
    elif sensitivity == 'low':
        # Берем самый мягкий (самые широкие границы)
        idx = np.argmax([max_val - min_val for min_val, max_val in results])
    else:  # medium
        # Берем средний по ширине
        ranges = [max_val - min_val for min_val, max_val in results]
        idx = np.argsort(ranges)[len(ranges) // 2]

    return results[idx]

