<<<<<<< HEAD
from __future__ import absolute_import
from __future__ import print_function
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema
from sklearn.preprocessing import normalize
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.layers import Conv2D, Activation, AveragePooling2D
from keras import backend as K
from tqdm import tqdm
import GPUtil as GPU
import psutil

# сделаем так, чтобы tf не резервировал под себя сразу всю память
# https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

# Assume that you have 12GB of GPU memory and want to allocate ~4GB:
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
def gpu_usage():
    GPUs = GPU.getGPUs()
    # XXX: only one GPU on Colab and isn’t guaranteed
    if len(GPUs) == 0:
        return False
    gpu = GPUs[0]
    process = psutil.Process(os.getpid())
    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util: {2:3.0f}% | Total: {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))

seed = 347
# https://coderoad.ru/51249811/Воспроизводимые-результаты-в-Tensorflow-с-tf-set_random_seed
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

pd.pandas.set_option("display.max_columns", None)
pd.set_option("expand_frame_repr", False)
pd.set_option("precision", 2)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""" Main Block """""""""""""""""""""""""""""""""
source_root = "source_root/15min"
destination_root = "outputs"
filename = "VZ_15_Minutes_(with_indicators).txt"
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
print(f'Исходный датасет:\n{df}\n')
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

"""Для обучения модели"""
START_TRAIN = "2018-01-01 09:00:00"
END_TRAIN = "2021-09-30 23:00:00"
"""Для тестирования модели"""
START_TEST = "2021-10-01 09:00:00"
END_TEST = "2021-12-31 23:00:00"
"""Отберем данные по максе"""
mask_train = (df.index >= START_TRAIN) & (df.index <= END_TRAIN)
Train_df = df.loc[mask_train]
print(f'Источник патернов:\n{Train_df}\n')
mask_test = (df.index >= START_TEST) & (df.index <= END_TEST)
Eval_df = df.loc[mask_test]
print(f'Датасет test с фичами:\n{Eval_df}\n')
"""Сохраняем даты, удаляем из основынх датафрэймов"""
Train_dates = Train_df.index.to_list()
Eval_dates = Eval_df.index.to_list()
Train_df = Train_df.reset_index(drop=True)
Eval_df = Eval_df.reset_index(drop=True)

"""Основыен параметры"""
num_classes = 2
extr_window = 40
n_size = 7  # размер паттерна ??
"""Параметры обучения"""
batch_size = 100
epochs = 700
treshhold = 0.01  # граница уверености
latent_dim = 100


def get_locals(data_df, n):  # данные подаются в формате df

    data_df["index"] = data_df.index
    data_df["min"] = data_df.iloc[
        argrelextrema(data_df.Close.values, np.less_equal, order=n)[0]
    ]["Close"]
    data_df["max"] = data_df.iloc[
        argrelextrema(data_df.Close.values, np.greater_equal, order=n)[0]
    ]["Close"]

    f = plt.figure()
    f.set_figwidth(80)
    f.set_figheight(65)
    plt.scatter(data_df.index, data_df["min"], c="r")
    plt.scatter(data_df.index, data_df["max"], c="g")
    plt.plot(data_df.index, data_df["Close"])
    plt.show()

    Min_ = data_df.loc[data_df["min"].isnull() == False]
    Min_.reset_index(inplace=True)
    Min_.drop(["level_0", "max"], axis=1, inplace=True)

    Max_ = data_df.loc[data_df["max"].isnull() == False]
    Max_.reset_index(inplace=True)
    Max_.drop(["level_0", "min"], axis=1, inplace=True)

    data_df.drop(["index", "min", "max"], axis=1, inplace=True)
    return Min_, Max_


def get_patterns(data, min_indexes, max_indexes, n_backwatch):  # подаем дата как нумпи, индексы как лист

    negative_patterns = []
    positive_patterns = []
    for ind in min_indexes:
        if ind - n_backwatch >= 0:
            neg = data[(ind - n_backwatch): ind]
            negative_patterns.append(neg)
    for ind in max_indexes:
        if ind - (2 * n_backwatch) >= 0:
            pos = data[(ind - n_backwatch): ind]
            positive_patterns.append(pos)
    negative_patterns = np.array(negative_patterns)
    positive_patterns = np.array(positive_patterns)
    return negative_patterns, positive_patterns


Min_train_locals, Max_train__locals = get_locals(Train_df, extr_window)

neg_patern, pos_patern = get_patterns(
    Train_df.to_numpy(),
    Min_train_locals["index"].values.tolist(),
    Max_train__locals["index"].values.tolist(),
    n_size,
)

print(f"Число патернов, найденных алгоритмом для разметки:\n"
      f"neg_patern.shape: {neg_patern.shape}\t|\tpos_patern.shape: {pos_patern.shape}")

"""Функции сети"""


def euclid_dis(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    y_true = tf.dtypes.cast(y_true, tf.float64)
    y_pred = tf.dtypes.cast(y_pred, tf.float64)
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def create_pairs(x, digit_indices):
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
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def create_base_net(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(16, (2, 2), activation="tanh", padding="same")(input)
    x = AveragePooling2D(pool_size=(2, 2), padding="same")(x)
    x = Conv2D(16, (2, 2), activation="tanh", padding="same")(x)
    x = Conv2D(16, (3, 2), activation="tanh", padding="same")(x)
    x = Conv2D(32, (2, 2), activation="tanh", padding="same")(x)
    # x = AveragePooling2D(pool_size = (2,2))(x)
    x = Flatten()(x)
    x = Dense(latent_dim, activation="tanh")(x)
    model = Model(input, x)
    # model.summary()
    return model


def compute_accuracy(y_true, y_pred):
    """Compute classification accuracy with a fixed threshold on distances.
    """
    gpu_usage()
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)


def accuracy(y_true, y_pred):
    """Compute classification accuracy with a fixed threshold on distances.
    """
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def get_train_samples(negative, positive):
    train_samples = []
    train_labels = []

    gpu_usage()

    for neg in negative:
        train_samples.append(neg)
        train_labels.append(0)
    for pos in positive:
        train_samples.append(pos)
        train_labels.append(1)
    train_samples = np.array(train_samples)
    train_labels = np.array(train_labels)
    return train_samples, train_labels


"""Получаем Xtrain и Ytrain для обучения сети"""
Xtrain, Ytrain = get_train_samples(neg_patern, pos_patern)
"""Нормализуем Xtrain"""
X_norm = [normalize(i, axis=0, norm="max") for i in Xtrain]
"""решейпим для подачи в сеть"""
X_norm = np.array(X_norm).reshape(
    -1, neg_patern[0].shape[0], neg_patern[0][0].shape[0], 1
)
Ytrain = Ytrain.reshape(-1, 1)

"""Получаем пары"""
digit_indices = [np.where(Ytrain == i)[0] for i in range(num_classes)]
tr_pairs, tr_y = create_pairs(X_norm, digit_indices)

"""Создаем сеть"""
input_shape = (neg_patern[0].shape[0], neg_patern[0][0].shape[0], 1)
base_network = create_base_net(input_shape)

"""Запускаем обучение"""
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclid_dis, output_shape=eucl_dist_output_shape)(
    [processed_a, processed_b]
)

model = Model([input_a, input_b], distance)
model.compile(loss=contrastive_loss, optimizer="adam", metrics=[accuracy])
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, batch_size=batch_size, epochs=epochs)

"""Тестируем модель"""

''' Готовим данные для проверки, размер n_size с шагом 1'''

eval_array = Eval_df.to_numpy()
eval_samples = [eval_array[i - n_size:i] for i in range(len(eval_array)) if i - n_size >= 0]
eval_normlzd = [normalize(i, axis=0, norm='max') for i in eval_samples]
eval_normlzd = np.array(eval_normlzd).reshape(-1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1)
print(f'\n268: eval_normlzd.shape\t|\tЧисло пар для теста: {eval_normlzd.shape}')

negative_anchor = []
positve_anchor = []
for index, i in enumerate(Ytrain):
    if i == 0:
        negative_anchor.append(X_norm[index])
    else:
        positve_anchor.append(X_norm[index])

negative_anchor = np.array(negative_anchor).reshape((-1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1))
positve_anchor = np.array(positve_anchor).reshape((-1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1))

Min_prediction_pattern_name = []
date = []
open = []
high = []
low = []
close = []
volume = []
distance = []
Eval_str_dates = [str(i) for i in Eval_dates]
signal = []  # лэйбл
k = 0
treshhold = treshhold

for indexI, eval in enumerate(tqdm(eval_normlzd)):

    neg_predictions = []
    for neg in negative_anchor:
        neg_pred = model.predict([neg.reshape(1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1),
                                  eval.reshape(1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1)])
        neg_predictions.append(neg_pred)

    date.append(Eval_str_dates[indexI + (n_size - 1)])
    open.append(float(eval_array[indexI + (n_size - 1), [0]]))
    high.append(float(eval_array[indexI + (n_size - 1), [1]]))
    low.append(float(eval_array[indexI + (n_size - 1), [2]]))
    close.append(float(eval_array[indexI + (n_size - 1), [3]]))
    volume.append(float(eval_array[indexI + (n_size - 1), [4]]))
    Min_prediction_pattern_name.append(neg_predictions.index(min(neg_predictions)))

    min_ex = min(neg_predictions)
    distance.append(float(min_ex))

    if min_ex <= treshhold:

        signal.append(1)
    else:
        signal.append(0)

Predictions = pd.DataFrame(
    list(zip(date, open, high, low, close, volume, signal, Min_prediction_pattern_name, distance)),
    columns=['date', 'open', 'high', 'low', 'close', 'volume', 'signal', 'pattern No.', 'distance'])

Predictions.to_csv(f'{destination_root}/test_results_{n_size}bar_latentdim{latent_dim}.csv')
=======
from __future__ import absolute_import
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

from functions_for_train_nn import get_locals, get_patterns, create_pairs, get_train_samples
from losses import euclid_dis, eucl_dist_output_shape, contrastive_loss, accuracy
from models import create_base_net

pd.pandas.set_option("display.max_columns", None)
pd.set_option("expand_frame_repr", False)
pd.set_option("precision", 2)


source_root = "source_root"
destination_root = "outputs"
model_name='Best_model'
filename = "VZ_15_Minutes_(with_indicators)_2018_18012022.txt"
out_filename='test_results.csv'
eval_dates_save='Eval_dates.csv'
eval_data_df='Eval_df.csv'

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
Eval_dates = Eval_df.index.to_list()
Train_df=Train_df.reset_index(drop=True)
Eval_df=Eval_df.reset_index(drop=True)


Eval_dates.to_csv(f'{destination_root}/{eval_dates_save}')
Eval_df.to_csv(f'{destination_root}/{eval_data_df}')



"""Основыен параметры"""
num_classes = 2
extr_window = 40
n_size = 20  # размер мемори

"""Параметры обучения"""
batch_size=10
epochs=500
treshhold=0.05 #  граница уверености



Min_train_locals, Max_train__locals = get_locals(Train_df, extr_window)

buy_patern, sell_patern = get_patterns(
    Train_df.to_numpy(),
    Min_train_locals["index"].values.tolist(),
    Max_train__locals["index"].values.tolist(),
    n_size,
)




print(f"buy_patern.shape: {buy_patern.shape}\t|\sell_patern.shape: {sell_patern.shape}")




"""Получаем Xtrain и Ytrain для обучения сети"""
Xtrain, Ytrain = get_train_samples(buy_patern, sell_patern)
"""Нормализуем Xtrain"""
X_norm = [normalize(i, axis=0, norm="max") for i in Xtrain]
"""решейпим для подачи в сеть"""
X_norm = np.array(X_norm).reshape(
    -1, buy_patern[0].shape[0], buy_patern[0][0].shape[0], 1
)
Ytrain = Ytrain.reshape(-1, 1)

"""Получаем пары"""
digit_indices = [np.where(Ytrain == i)[0] for i in range(num_classes)]
tr_pairs, tr_y = create_pairs(X_norm, digit_indices)

"""Создаем сеть"""
input_shape = (buy_patern[0].shape[0], buy_patern[0][0].shape[0], 1)
base_network = create_base_net(input_shape)

"""Запускаем обучение"""
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclid_dis, output_shape=eucl_dist_output_shape)(
    [processed_a, processed_b]
)

model = Model([input_a, input_b], distance)

model.compile(loss=contrastive_loss, optimizer="adam", metrics=[accuracy])
history=model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, batch_size=batch_size, epochs=epochs)

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.show()


model.save(f'{destination_root}/{model_name}')
>>>>>>> SUPL-23
