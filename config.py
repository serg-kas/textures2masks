"""
Модуль проверки и настройки конфигурации.
Проверит наличие и при необходимости создаст рабочие папки.
"""
import os


def check_folders(folders_to_check, verbose=False):
    """
    Проверяет наличие и при необходимости создает рабочие папки
    :param folders_to_check: список папок для проверки
    :param verbose: выводить сообщения
    :return: список созданных папок
    """
    if verbose:
        print("Проверяем наличие рабочих папок: {}".format(folders_to_check))
    #
    folders_created = []
    for folder in folders_to_check:
        if not (folder in os.listdir('.')):
            os.mkdir(folder)
            if verbose:
                print("  Создали отсутствующую папку: '{}'".format(folder))

    return folders_created
