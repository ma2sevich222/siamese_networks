from scipy.signal import argrelmin, argrelmax
from sklearn.preprocessing import StandardScaler
import numpy as np

""" Получаем паттерны:
    Функция принимает на вход:
    data_df - data frame  данных для обучения
    profit_value - требуемы уровень профита в долях
    EXTR_WINDOW - размер окна поисков экстремумов
    PATTERN_SIZE - размер паттерна
    OVERLAP - кол-во баров от экстремума"""


def get_train_data(data_df, profit_value, EXTR_WINDOW, PATTERN_SIZE, OVERLAP):  #
    close_price = data_df["Close"].values.tolist()
    # находим индексы локальных минимумов
    min_indexes = argrelmin(np.array(close_price), order=EXTR_WINDOW)[0]
    min_indexes = [i for i in min_indexes if i + EXTR_WINDOW <= len(data_df)]
    # находим индексы локальных максимумов
    max_indexes = argrelmax(np.array(close_price), order=EXTR_WINDOW)[0]
    max_indexes = [i for i in max_indexes if i + EXTR_WINDOW <= len(data_df)]
    # фильтруем согласно профиту
    indexes_with_profit = []
    for i in min_indexes:
        if i + (EXTR_WINDOW) <= len(close_price):

            if (
                close_price[int(i) : int(i + (EXTR_WINDOW))][-1] - close_price[i]
                >= close_price[i] * profit_value
            ):
                indexes_with_profit.append(i)

    indexes_lost_profit = []
    for i in max_indexes:
        if i + (EXTR_WINDOW) <= len(close_price):
            if close_price[int(i) : int(i + (EXTR_WINDOW))][-1] - close_price[i] <= -(
                close_price[i] * profit_value
            ):
                indexes_lost_profit.append(i)

    patterns = []  # отбираем паттерны buy

    for ind in indexes_with_profit:
        if ind - PATTERN_SIZE >= 0:
            patt = data_df[(ind - PATTERN_SIZE + OVERLAP) : ind + OVERLAP].to_numpy()
            patterns.append(patt)

    sell_patterns = []  # отбираем паттерны sell
    for ind in indexes_lost_profit:
        if ind - PATTERN_SIZE >= 0:
            patt = data_df[(ind - PATTERN_SIZE + OVERLAP) : ind + OVERLAP].to_numpy()
            sell_patterns.append(patt)
    assert (
        len(patterns) > 0
    ), "На данном участке  не обнаруженно паттернов класса 'buy'."
    assert (
        len(sell_patterns) > 0
    ), "На данном участке  не обнаруженно паттернов класса 'sell'."
    if len(patterns) < 2:
        patterns.append(patterns[0])
    if len(sell_patterns) < 2:
        sell_patterns.append(sell_patterns[0])

    print(f"Найдено паттернов класса buy = {len(patterns)}")
    print(f"Найдено паттернов класса sell = {len(sell_patterns)}")
    n_samples_to_train = ((len(patterns) - 1) * len(sell_patterns)) + (
        (len(sell_patterns) - 1) * len(patterns)
    )
    print(f"Количество уникальных триплетов = {n_samples_to_train}")

    scaler = StandardScaler()  # стандартизируем данные по каждой оси
    std_patterns = np.array([scaler.fit_transform(i) for i in patterns]).reshape(
        -1, PATTERN_SIZE, len(data_df.columns.to_list()), 1
    )
    std_sell_patterns = np.array(
        [scaler.fit_transform(i) for i in sell_patterns]
    ).reshape(-1, PATTERN_SIZE, len(data_df.columns.to_list()), 1)

    train_x = [std_patterns, std_sell_patterns]

    if len(patterns) == len(sell_patterns):
        train_x = np.array(train_x)

    else:
        train_x = np.array(train_x, dtype=object)

    return train_x, n_samples_to_train


""" Генерим триплеты:
    n_samples_to_train - количество требуемых триплетов, если подавать через data_loader он равен бач сайзу
    n_classes- количество классов ( сейчас их 2 но может быть и больше)
    train_x - np массив паттернов"""


def get_triplet_random(n_samples_to_train, n_classes, train_x):
    """
    Генерим триплеты рандомным сэмплингом
    """

    X = train_x

    w, h, c = X[0][0].shape

    triplets = [np.zeros((n_samples_to_train, w, h, c)) for i in range(int(3))]

    for i in range(n_samples_to_train):
        # берем рандомно класс
        anchor_class = np.random.randint(0, n_classes)

        nb_sample_available_for_class_AP = X[anchor_class].shape[0]

        # опеделеляем якорь и позитивный пример

        idx_A = 0
        idx_P = np.random.choice(
            range(1, nb_sample_available_for_class_AP), size=1, replace=False
        )

        # берем другой класс
        negative_class = (anchor_class + np.random.randint(1, n_classes)) % n_classes
        nb_sample_available_for_class_N = X[negative_class].shape[0]

        # берем негативный пример
        idx_N = np.random.randint(0, nb_sample_available_for_class_N)

        triplets[0][i, :, :, :] = X[anchor_class][idx_A, :, :, :]
        triplets[1][i, :, :, :] = X[anchor_class][idx_P, :, :, :]
        triplets[2][i, :, :, :] = X[negative_class][idx_N, :, :, :]

    return np.array(triplets)
