import plotly.io as pio
import json
pio.renderers.default = "browser"
from utilits.visualisation_functios import *
from utilits.data_transforms import *
from constants import *

destination_root = "outputs"
patterns_file_name = 'buy_patterns.txt'
results_file_name = 'pattern_model_test.csv'
eval_data_df = 'Eval_df.csv'
train_data_df = 'train_df.csv'
eval_dates_save = 'eval_dates.txt'


# загружаем массив размечанных паттернов и результаты тестирования модели
loader = np.loadtxt(f'{destination_root}/{patterns_file_name}')
patterns = loader.reshape(-1,20,13)  # возвращаем исходный размер
results = pd.read_csv(f'{destination_root}/{results_file_name}', index_col=[0])
results = results.rename(columns={"pattern No.": "pattern"})
neighbor_patterns = calculate_cos_dist(patterns, pattern)  # ближайшие соседи паттерна
Eval_df = pd.read_csv(f"{destination_root}/{eval_data_df}")
Eval_df = Eval_df.drop("Unnamed: 0", axis=1)
Train_df=pd.read_csv(f"{destination_root}/{train_data_df}")

with open(f'{destination_root}/{eval_dates_save}', 'r') as f:
    Eval_dates = json.loads(f.read())
column_list = Eval_df.columns.to_list()

paterns_df = patterns_to_df(patterns, column_list)
eval_samples_df = evdata_for_visualisation(Eval_df, batch)

# График визуального сравнения паттерна и предсказаний
pattern_samples_plot(paterns_df, eval_samples_df, results, 2)


# График визуального сравнения паттерна и ближайших к нему размеченных паттернов
plot_nearlist_patterns(paterns_df, neighbor_patterns)

# Heatmap размеченных паттернов
patterns_heatmap(patterns)

# функция отображения локальных эемтремумов
get_locals(Eval_df, extrema_window)  # функция отображения локальных эемтремумов

# функция показывает на данных, где были определены заданные паттерны с учетом  границ дистанции
predictions_plotting(results, list_of_trashholds, list_of_patterns)
