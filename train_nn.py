from __future__ import absolute_import
from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
from keras.models import save_model
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from functions_for_train_nn import get_locals, get_patterns,create_pairs,get_train_samples
from models import create_base_net
from losses import euclid_dis,eucl_dist_output_shape,contrastive_loss,compute_accuracy,accuracy


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
