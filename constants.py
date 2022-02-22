# https://datahacker.rs/019-siamese-network-in-pytorch-with-application-to-face-similarity/
"""
Created on Wed Feb  9 16:36:25 2022

@author: ma2sevich
"""

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""" Parameters Block """""""""""""""""""""""""""
SOURCE_ROOT = "source_root/1min"
DESTINATION_ROOT = "outputs"
MODELS_ROOT = 'models'
FILENAME = "TSLA_1_Minute_(with_indicators).txt"
TICKER = 'TSLA'
# out_filename ='test_results.csv'
# buy_patterns_save = 'buy_patterns.txt'
eval_dates_save = 'eval_dates.txt'
eval_data_df = 'Eval_df.csv'
train_data_df = 'train_df.csv'

"""Основыен параметры"""
TRESHHOLD_DISTANCE = 100
num_classes = 2
EXTR_WINDOW = 15  # то, на каком окне слева и вправо алгоритм размечает экстремумы
PATTERN_SIZE = 20  # размер паттерна
"""Параметры обучения"""
latent_dim = 5
BATCH_SIZE = 1000
epochs = 5000
learning_rate = 1e-3
norm = 'l1'

"""Для обучения модели"""
START_TRAIN = "2021-07-01 09:00:00"
END_TRAIN = "2021-07-31 23:00:00"
"""Для тестирования модели"""
START_TEST = "2021-08-01 09:00:00"
END_TEST = "2021-08-02 23:00:00"  # last date is 2021-08-26 16:00:00