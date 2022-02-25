

import numpy as np
import pandas as pd
import plotly.io as pio

from constants import *
from utilits.data_transforms import patterns_to_df, evdata_for_visualisation
from utilits.functions_for_train_nn import get_locals
from utilits.visualisation_functios import calculate_cos_dist, pattern_samples_plot, plot_nearlist_patterns, \
    patterns_heatmap
from utilits.data_load import data_load

pio.renderers.default = "browser"

pd.pandas.set_option("display.max_columns", None)
pd.set_option("expand_frame_repr", False)
pd.set_option("precision", 2)


patterns_file_name = 'buy_patterns_extr_window15_latent_dim5_pattern_size20.csv'
results_file_name = 'test_results_extr_window15_latent_dim5_pattern_size20.csv'
pattern_num = 112  # номер паспознаваемого паттерна

# загрузка данных, которые использовались для обучение и теста модели
Train_df, Eval_df, Eval_dates_str = data_load(SOURCE_ROOT, FILENAME)

# загружаем массив размечанных паттернов и результаты тестирования модели
loader = np.loadtxt(f'{DESTINATION_ROOT}/{patterns_file_name}')
results = pd.read_csv(f'{DESTINATION_ROOT}/{results_file_name}', index_col=[0])
patterns = loader.reshape(-1, PATTERN_SIZE, len(Eval_df.columns.to_list()))  # возвращаем исходный размер
neighbor_patterns = calculate_cos_dist(patterns, pattern_num)  # ближайшие соседи паттерна
column_list = Eval_df.columns.to_list()

paterns_df = patterns_to_df(patterns, column_list)
eval_samples_df = evdata_for_visualisation(Eval_df, PATTERN_SIZE)

# График визуального сравнения паттерна и предсказаний
pattern_samples_plot(paterns_df, eval_samples_df, results, pattern_num)

# График визуального сравнения паттерна и ближайших к нему размеченных паттернов
plot_nearlist_patterns(paterns_df, neighbor_patterns)

# Heatmap размеченных паттернов
patterns_heatmap(patterns)

# функция отображения локальных эемтремумов
get_locals(Eval_df, EXTR_WINDOW)  # функция отображения локальных эемтремумов

