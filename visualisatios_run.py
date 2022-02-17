import json
import plotly.io as pio
from utilits.visualisation_functios import *
from utilits.data_transforms import *
from constants import *
pio.renderers.default = "browser"


patterns_file_name = 'buy_patterns_extr_window60_pattern_size15.csv'
results_file_name = 'test_results_extr_window60_pattern_size15.csv'
pattern = 26

# загружаем массив размечанных паттернов и результаты тестирования модели
loader = np.loadtxt(f'{DESTINATION_ROOT}/{patterns_file_name}')
results = pd.read_csv(f'{DESTINATION_ROOT}/{results_file_name}', index_col=[0])
results = results.rename(columns={"pattern No.": "pattern"})
Eval_df = pd.read_csv(f"{DESTINATION_ROOT}/{eval_data_df}")
Eval_df = Eval_df.drop("Unnamed: 0", axis=1)
Train_df = pd.read_csv(f"{DESTINATION_ROOT}/{train_data_df}")
patterns = loader.reshape(-1, PATTERN_SIZE, len(Eval_df.columns.to_list()))# возвращаем исходный размер
neighbor_patterns = calculate_cos_dist(patterns, pattern)  # ближайшие соседи паттерна

with open(f'{DESTINATION_ROOT}/{eval_dates_save}', 'r') as f:
    Eval_dates = json.loads(f.read())
column_list = Eval_df.columns.to_list()

paterns_df = patterns_to_df(patterns, column_list)
eval_samples_df = evdata_for_visualisation(Eval_df, BATCH_SIZE)

# График визуального сравнения паттерна и предсказаний
pattern_samples_plot(paterns_df, eval_samples_df, results, pattern)

# График визуального сравнения паттерна и ближайших к нему размеченных паттернов
plot_nearlist_patterns(paterns_df, neighbor_patterns)

# Heatmap размеченных паттернов
patterns_heatmap(patterns)

# функция отображения локальных эемтремумов
get_locals(Eval_df, EXTR_WINDOW)  # функция отображения локальных эемтремумов

