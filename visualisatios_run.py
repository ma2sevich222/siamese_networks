import plotly.io as pio

pio.renderers.default = "browser"
from utilits.visualisation_functios import *
from utilits.data_transforms import *
from constants import *

destination_root =  "outputs"
source_root = "source_root/15min"
filename = "VZ_15_Minutes_(with_indicators).txt"
patterns_file_name = 'buy_patterns.txt'
results_file_name = 'pattern_model_test.csv'

# Загрузка  и отчиска данных на которых тестировалась модель
indices = [
    i for i, x in enumerate(filename) if x == "_"
]  # находим индексы вхождения '_'
ticker = filename[: indices[0]]

"""Загрузка и подготовка данных"""
df = pd.read_csv(f"{source_root}/{filename}")
df.rename(columns=lambda x: x.replace(">", ""), inplace=True)
df.rename(columns=lambda x: x.replace("<", ""), inplace=True)
df.rename(columns=lambda x: x.replace(" ", ""), inplace=True)
del df["ZeroLine"]
columns = df.columns.tolist()

"""Формат даты в Datetime"""
print(df)
new_df = df["Date"].str.split(".", expand=True)
df["Date"] = new_df[2] + "-" + new_df[1] + "-" + new_df[0] + " " + df["Time"]
df.Date = pd.to_datetime(df.Date)
df.dropna(axis=0, inplace=True)  # Удаляем наниты
df["Datetime"] = df["Date"]
df.set_index("Datetime", inplace=True)
df.sort_index(ascending=True, inplace=False)
df = df.rename(columns={"<Volume>": "Volume"})
del df["Time"], df["Date"]

"""Добавление фич"""
df["SMA"] = df.iloc[:, 3].rolling(window=10).mean()
df["CMA30"] = df["Close"].expanding().mean()
df["SMA"] = df["SMA"].fillna(0)
print(df)

"""Для обучения модели"""
START_TRAIN = "2018-01-01 09:00:00"
END_TRAIN = "2020-12-31 23:00:00"
"""Для тестирования модели"""
START_TEST = "2021-01-01 09:00:00"
END_TEST = "2021-12-31 23:00:00"
"""Отберем данные по максе"""
mask_train = (df.index >= START_TRAIN) & (df.index <= END_TRAIN)
Train_df = df.loc[mask_train]
mask_test = (df.index >= START_TEST) & (df.index <= END_TEST)
Eval_df = df.loc[mask_test]
"""Сохраняем даты, удаляем из основынх датафрэймов"""
Train_dates = Train_df.index.to_list()
Eval_dates = Eval_df.index.astype(str)
Train_df = Train_df.reset_index(drop=True)
Eval_df = Eval_df.reset_index(drop=True)
Eval_dates_str=[str(i) for i in Eval_dates]

# загружаем массив размечанных паттернов и результаты тестирования модели
loader = np.loadtxt(f'{destination_root}/{patterns_file_name}')
patterns = loader.reshape(-1,20,13)  # возвращаем исходный размер
results = pd.read_csv(f'{destination_root}/{results_file_name}', index_col=[0])
results = results.rename(columns={"pattern No.": "pattern"})
neighbor_patterns = calculate_cos_dist(patterns, pattern)  # ближайшие соседи паттерна

paterns_df = patterns_to_df(patterns)
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
