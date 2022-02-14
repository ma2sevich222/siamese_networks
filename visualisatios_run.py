import plotly.io as pio

pio.renderers.default = "browser"
from utilits.visualisation_functios import *
from utilits.data_transforms import *
from constants import *


data_file_name = 'VZ_15_Minutes_(with_indicators)_2018_18012022.txt'
patterns_file_name = 'buy_patterns.txt'
results_file_name = 'test_results_latentdim10.csv'

# Загрузка  и отчиска данных на которых тестировалась модель
df = pd.read_csv(f'{SOURCE_ROOT}/{data_file_name}', delimiter=",")
raw_eval_data = df[
    ["<Date>", " <Time>", " <Open>", " <High>", " <Low>", " <Close>", " <Volume>"]
].copy()
column_names = raw_eval_data.columns.to_list()
raw_eval_data = raw_eval_data[~(raw_eval_data == 0).any(axis=1)]
raw_data_dates = pd.to_datetime(
    raw_eval_data["<Date>"] + raw_eval_data[" <Time>"], format="%d.%m.%Y%H:%M:%S"
)
raw_eval_data.drop(["<Date>", " <Time>"], axis=1, inplace=True)
raw_eval_data.insert(0, "Date", raw_data_dates)
raw_eval_data.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
raw_eval_data = raw_eval_data[["Date", "Open", "High", "Low", "Close", "Volume"]]

# готовим данные, на которых проверялась модель, для визуализации
eval_df = raw_eval_data.loc[10000:]
eval_dates = eval_df[["Date"]]
eval_df.drop(["Date"], axis=1, inplace=True)
eval_df = eval_df.reset_index(drop=True)

# загружаем массив размечанных паттернов и результаты тестирования модели
loader = np.loadtxt(f'{DESTINATION_ROOT}/{patterns_file_name}')
patterns = loader.reshape(data_shape)  # возвращаем исходный размер
results = pd.read_csv(f'{DESTINATION_ROOT}/{results_file_name}', index_col=[0])
results = results.rename(columns={"pattern No.": "pattern"})
neighbor_patterns = calculate_cos_dist(patterns, pattern)  # ближайшие соседи паттерна

paterns_df = patterns_to_df(patterns)
eval_samples_df = evdata_for_visualisation(eval_df, batch)

# График визуального сравнения паттерна и предсказаний
pattern_samples_plot(paterns_df, eval_samples_df, results, 2)


# График визуального сравнения паттерна и ближайших к нему размеченных паттернов
plot_nearlist_patterns(paterns_df, neighbor_patterns)

# Heatmap размеченных паттернов
patterns_heatmap(patterns)

# функция отображения локальных эемтремумов
get_locals(raw_eval_data, extrema_window)  # функция отображения локальных эемтремумов

# функция показывает на данных, где были определены заданные паттерны с учетом  границ дистанции
predictions_plotting(results, list_of_trashholds, list_of_patterns)
