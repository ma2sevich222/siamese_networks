from __future__ import absolute_import
from __future__ import print_function

import os
import random

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from other_codes.old_project.old_utilits.functions_for_train_nn import (
    get_locals,
    get_patterns,
    create_pairs,
    get_train_samples,
)
from other_codes.old_project.old_utilits.losses import (
    euclid_dis,
    eucl_dist_output_shape,
    contrastive_loss,
    accuracy,
)
from keras.models.base_net import create_base_net
from utilits.data_load import data_load

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from tqdm import tqdm
import GPUtil as GPU
import psutil

# сделаем так, чтобы tf не резервировал под себя сразу всю память
# https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
gpus = tf.config.experimental.list_physical_devices("GPU")
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
    print(
        "GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util: {2:3.0f}% | Total: {3:.0f}MB".format(
            gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil * 100, gpu.memoryTotal
        )
    )


seed = 347
# https://coderoad.ru/51249811/Воспроизводимые-результаты-в-Tensorflow-с-tf-set_random_seed
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

pd.pandas.set_option("display.max_columns", None)
pd.set_option("expand_frame_repr", False)
pd.set_option("precision", 2)


"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
"""""" """""" """""" """""" """"" Parameters Block """ """""" """""" """""" """"""
from other_codes.not_used.constants import *


"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
"""""" """""" """""" """""" """"" Main Block """ """""" """""" """""" """""" """"""
indices = [
    i for i, x in enumerate(FILENAME) if x == "_"
]  # находим индексы вхождения '_'
ticker = FILENAME[: indices[0]]


"""Загрузка и подготовка данных"""
Train_df, Eval_df, Eval_dates_str = data_load(SOURCE_ROOT, FILENAME)

Min_train_locals, Max_train__locals = get_locals(Train_df, EXTR_WINDOW)

buy_patern, sell_patern = get_patterns(
    Train_df.to_numpy(),
    Min_train_locals["index"].values.tolist(),
    Max_train__locals["index"].values.tolist(),
    PATTERN_SIZE,
)


buy_reshaped = buy_patern.reshape(buy_patern.shape[0], -1)
np.savetxt(
    f"{DESTINATION_ROOT}/buy_patterns_extr_window{EXTR_WINDOW}"
    f"_latent_dim{latent_dim}"
    f"_pattern_size{PATTERN_SIZE}.csv",
    buy_reshaped,
)

print(
    f"Найдено уникальных:\n"
    f"buy_patern.shape: {buy_patern.shape}\t|\tsell_patern.shape: {sell_patern.shape}"
)


"""Получаем Xtrain и Ytrain для обучения сети"""
Xtrain, Ytrain = get_train_samples(buy_patern, sell_patern)
"""Нормализуем Xtrain"""
X_norm = [normalize(i, norm=norm, axis=0) for i in Xtrain]  # norm="max"
"""решейпим для подачи в сеть"""
X_norm = np.array(X_norm).reshape(
    -1, buy_patern[0].shape[0], buy_patern[0][0].shape[0], 1
)
Ytrain = Ytrain.reshape(-1, 1)
"""Получаем пары"""
digit_indices = [np.where(Ytrain == i)[0] for i in range(num_classes)]
tr_pairs, tr_y = create_pairs(X_norm, digit_indices, num_classes)
"""Создаем сеть"""
input_shape = (buy_patern[0].shape[0], buy_patern[0][0].shape[0], 1)
base_network = create_base_net(input_shape, latent_dim)
""" Запускаем обучение """
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclid_dis, output_shape=eucl_dist_output_shape)(
    [processed_a, processed_b]
)


early_stopping = EarlyStopping(
    monitor="loss", patience=100, verbose=0, restore_best_weights=True, mode="min"
)
# reduce_lr = ReduceLROnPlateau(monitor='accuracy', factor=0.9, patience=12, min_lr=1e-07, verbose=1, mode='max')
checkpoint = ModelCheckpoint(
    f"{MODELS_ROOT}/{TICKER}_model.h5",
    monitor="accuracy",
    verbose=0,
    save_best_only=True,
)

model = Model([input_a, input_b], distance)
optimizer = tf.keras.optimizers.Adam(learning_rate)
model.compile(loss=contrastive_loss, optimizer=optimizer, metrics=[accuracy])
history = model.fit(
    [tr_pairs[:, 0], tr_pairs[:, 1]],
    tr_y,
    batch_size=BATCH_SIZE,
    epochs=epochs,
    verbose=1,
    callbacks=[early_stopping, checkpoint],
)

hist = pd.DataFrame(history.history)
# del hist['lr']
pd.DataFrame(hist).plot(figsize=(8, 5))
plt.show()


""" Поиск выученных паттернов в тестовом датасете """
# Загрузим модель с лучшими весами
model.load_weights(f"{MODELS_ROOT}/{TICKER}_model.h5")

eval_array = Eval_df.to_numpy()
eval_samples = [
    eval_array[i - PATTERN_SIZE : i]
    for i in range(len(eval_array))
    if i - PATTERN_SIZE >= 0
]
eval_normlzd = [normalize(i, axis=0, norm=norm) for i in eval_samples]
eval_normlzd = np.array(eval_normlzd).reshape(
    -1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1
)

min_prediction_pattern_name = []
date, open = [], []
high = []
low = []
close = []
volume = []
distance = []
signal = []
k = 0
for indexI, eval in enumerate(tqdm(eval_normlzd)):

    buy_predictions = []
    for buy in buy_patern:
        buy_pred = model.predict(
            [
                buy.reshape(
                    1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1
                ),
                eval.reshape(
                    1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1
                ),
            ]
        )
        buy_predictions.append(buy_pred)

    date.append(Eval_dates_str[indexI + (PATTERN_SIZE - 1)])
    open.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [0]]))
    high.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [1]]))
    low.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [2]]))
    close.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [3]]))
    volume.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [4]]))
    min_prediction_pattern_name.append(buy_predictions.index(min(buy_predictions)))

    min_ex = min(buy_predictions)
    distance.append(float(min_ex))
    print(f"min(buy_predictions):\t{min_ex}")

    if min_ex <= TRESHHOLD_DISTANCE:
        signal.append(1)
    else:
        signal.append(0)

Predictions = pd.DataFrame(
    list(
        zip(
            date,
            open,
            high,
            low,
            close,
            volume,
            signal,
            min_prediction_pattern_name,
            distance,
        )
    ),
    columns=[
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "signal",
        "pattern",
        "distance",
    ],
)

Predictions.to_csv(
    f"{DESTINATION_ROOT}/test_results_extr_window{EXTR_WINDOW}"
    f"_latent_dim{latent_dim}"
    f"_pattern_size{PATTERN_SIZE}.csv",
    index=True,
)
