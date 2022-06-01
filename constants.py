# https://datahacker.rs/019-siamese-network-in-pytorch-with-application-to-face-similarity/
"""
Created on Wed Feb  9 16:36:25 2022

@author: ma2sevich
"""
##################################################################################
# Copyright © 2021-2099 Ekosphere. All rights reserved
# Author: Evgeny Matusevich
# Contacts: <ma2sevich222@gmail.com>
##################################################################################

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""" Parameters Block """""""""""""""""""""""""""
SOURCE_ROOT = "source_root"
DESTINATION_ROOT = "outputs"
FILENAME = "GC_2020_2022_60min.csv"


"""Основные параметры"""
EXTR_WINDOW = 140  # то, на каком окне слева и вправо алгоритм размечает экстремумы
PATTERN_SIZE = 86  # размер паттерна
OVERLAP = 10 # сдвигаем паттерн на n шагов вперед от локального минимума
profit_value = 0.003
TRAIN_WINDOW = 4500 # Количество баров для обучения


"""Для тестирования модели"""
START_TEST = "2021-03-25 00:00:00" #04.01.2021 0:01:25
END_TEST = "2022-04-01 21:00:00"  #01.04.2022 6:01:43