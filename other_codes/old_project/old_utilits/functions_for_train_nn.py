import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema


#  функция создания пар
def create_pairs(x, digit_indices, num_classes):
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [0, 1]
    return np.array(pairs), np.array(labels)


#  Получаем индексы локальных минимумов и максимумов
def get_locals(data_df, n):  # данные подаются в формате df

    data_df["index"] = data_df.index
    data_df["min"] = data_df.iloc[
        argrelextrema(data_df.Close.values, np.less_equal, order=n)[0]
    ]["Close"]
    data_df["max"] = data_df.iloc[
        argrelextrema(data_df.Close.values, np.greater_equal, order=n)[0]
    ]["Close"]

    # f = plt.figure()
    # f.set_figwidth(80)
    # f.set_figheight(65)
    # plt.scatter(data_df.index, data_df["min"], c="r")
    # plt.scatter(data_df.index, data_df["max"], c="g")
    # plt.plot(data_df.index, data_df["Close"])
    # plt.show()

    Min_ = data_df.loc[data_df["min"].isnull() == False]
    Min_.reset_index(inplace=True)
    Min_.drop(["level_0", "max"], axis=1, inplace=True)

    Max_ = data_df.loc[data_df["max"].isnull() == False]
    Max_.reset_index(inplace=True)
    Max_.drop(["level_0", "min"], axis=1, inplace=True)

    data_df.drop(["index", "min", "max"], axis=1, inplace=True)

    return Min_, Max_


#  Получаем паттерны
def get_patterns(
    data, min_indexes, max_indexes, n_backwatch
):  # подаем дата как нумпи, индексы как лист

    negative_patterns = []
    positive_patterns = []
    for ind in min_indexes:
        if ind - n_backwatch >= 0:
            neg = data[(ind - n_backwatch) : ind]
            negative_patterns.append(neg)
    for ind in max_indexes:
        if ind - (2 * n_backwatch) >= 0:
            pos = data[(ind - n_backwatch) : ind]
            positive_patterns.append(pos)
    negative_patterns = np.array(negative_patterns)
    positive_patterns = np.array(positive_patterns)
    return negative_patterns, positive_patterns


# Формируем X и Y для обучения сети
def get_train_samples(negative, positive):
    train_samples = []
    train_labels = []

    for neg in negative:
        train_samples.append(neg)
        train_labels.append(0)
    for pos in positive:
        train_samples.append(pos)
        train_labels.append(1)
    train_samples = np.array(train_samples)
    train_labels = np.array(train_labels)
    return train_samples, train_labels
