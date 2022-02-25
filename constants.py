# https://datahacker.rs/019-siamese-network-in-pytorch-with-application-to-face-similarity/
"""
Created on Wed Feb  9 16:36:25 2022

@author: ma2sevich
"""

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""" Parameters Block """""""""""""""""""""""""""
SOURCE_ROOT = "source_root/1min"
DESTINATION_ROOT = "outputs"
FILENAME = "TSLA_1_Minute_(with_indicators).txt"
# out_filename ='test_results.csv'
# buy_patterns_save = 'buy_patterns.txt'
eval_dates_save = 'eval_dates.txt'
eval_data_df = 'Eval_df.csv'
train_data_df = 'train_df.csv'

"""Основыен параметры"""
TRESHHOLD_DISTANCE = 100
num_classes = 2
EXTR_WINDOW = 60  # то, на каком окне слева и вправо алгоритм размечает экстремумы
PATTERN_SIZE = 15  # размер паттерна
"""Параметры обучения"""
latent_dim = 50
BATCH_SIZE = 600
epochs = 100

"""Для обучения модели"""
START_TRAIN = "2021-06-01 09:00:00"
END_TRAIN = "2021-07-31 23:00:00"
"""Для тестирования модели"""
START_TEST = "2021-08-01 09:00:00"
END_TEST = "2021-08-26 16:00:00"  # last date is 2021-08-26 16:00:00