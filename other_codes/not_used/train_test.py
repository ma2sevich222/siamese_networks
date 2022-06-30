##################################################################################
# Copyright © 2021-2099 Ekosphere. All rights reserved
# Author: Evgeny Matusevich
# Contacts: <ma2sevich222@gmail.com>
##################################################################################


import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from constants import (
    SOURCE_ROOT,
    DESTINATION_ROOT,
    FILENAME,
    START_TEST,
    END_TEST,
    EXTR_WINDOW,
    PATTERN_SIZE,
    OVERLAP,
    profit_value,
    TRAIN_WINDOW,
)
from models.torch_models import shotSiameseNetwork
from other_codes.not_used.data_load import data_load_OHLCV
from utilits.project_functions import (
    get_train_data,
    get_triplet_random,
    train_triplet_net,
)

torch.cuda.empty_cache()
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
"""""" """""" """""" """""" """"" Parameters Block """ """""" """""" """""" """"""
# 'lr': 0.0009470240447408595, 'batch_size': 185, 'embedding_dim': 235}

epochs = 7  # количество эпох
lr = 0.000009470240447408595  # learnig rate
embedding_dim = 160  # размер скрытого пространства
margin = 20  # маржа для лосс функции
batch_size = 150  # размер батчсайз
# n_samples_to_train = 1000  # Количество триплетов для тренировки
# tresh_hold = 15  # граница предсказаний ниже которой предсказания будут отображаться на графике

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
"""""" """""" """""" """""" """"" Distance functions """ """""" """""" """""" """"""
distance_function = lambda x, y: 1.0 - F.cosine_similarity(
    x, y
)  # функция расчета расстояния для триплет лосс
# distance_function = PairwiseDistance(p=2, eps=1e-06,)
# distance_function = l_infinity
# distance_function = euclid_dist
# distance_function = manhatten_dist

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
"""""" """""" """""" """""" """"" Data load,clean and prepare  """ """""" """""" """""" """"""

"""indices = [
    i for i, x in enumerate(FILENAME) if x == "_"
]  # находим индексы вхождения '_'
ticker = FILENAME[: indices[0]]"""

Train_df, Eval_df, train_dates, test_dates = data_load_OHLCV(
    SOURCE_ROOT, FILENAME, START_TEST, END_TEST, PATTERN_SIZE, TRAIN_WINDOW
)  # загрузка данных
train_x, n_samples_to_train = get_train_data(
    Train_df, profit_value, EXTR_WINDOW, PATTERN_SIZE, OVERLAP, train_dates
)  # получаем данные для создания триплетов
n_classes = len(train_x)
# train_x = np.array(train_x, dtype=object)
# print(test_dates.values[PATTERN_SIZE])

train_triplets = get_triplet_random(n_samples_to_train, n_classes, train_x)

print(" Размер данных для обучения:", np.array(train_triplets).shape)

tensor_A = torch.Tensor(train_triplets[0])  # transform to torch tensor
tensor_P = torch.Tensor(train_triplets[1])
tensor_N = torch.Tensor(train_triplets[2])

my_dataset = TensorDataset(tensor_A, tensor_P, tensor_N)  # create your datset
my_dataloader = DataLoader(my_dataset, batch_size=batch_size)

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
"""""" """""" """""" """""" """"" Train net  """ """""" """""" """""" """"""

net = shotSiameseNetwork(embedding_dim=embedding_dim).cuda()
train_triplet_net(lr, epochs, my_dataloader, net, distance_function)

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
"""""" """""" """""" """""" """"" Test data prepare  """ """""" """""" """""" """"""
scaler = StandardScaler()
eval_array = Eval_df.to_numpy()
eval_samples = [
    eval_array[i - PATTERN_SIZE : i]
    for i in range(len(eval_array))
    if i - PATTERN_SIZE >= 0
]
eval_normlzd = [scaler.fit_transform(i) for i in eval_samples]
eval_normlzd = np.array(eval_normlzd).reshape(
    -1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1
)

"""eval_array = Eval_df[
                ['DiffEMA', 'SmoothDiffEMA', 'VolatilityTunnel', 'BuyIntense', 'SellIntense']].to_numpy()
eval_samples = [eval_array[i - PATTERN_SIZE:i] for i in range(len(eval_array)) if i - PATTERN_SIZE >= 0]
eval_ohlcv = Eval_df[['Open', 'High', 'Low', 'Close', 'Volume']].to_numpy()
ohlcv_samples = [eval_ohlcv[i - PATTERN_SIZE:i] for i in range(len(eval_ohlcv)) if i - PATTERN_SIZE >= 0]"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
"""""" """""" """""" """""" """"" Test model  """ """""" """""" """""" """"""

date = []
open = []
high = []
low = []
close = []
volume = []
buy_pred = []
train_data_shape = []

net.eval()
with torch.no_grad():
    for indexI, eval_arr in enumerate(tqdm(eval_normlzd)):
        anchor = train_x[0][0].reshape(
            1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1
        )
        eval_arr_r = eval_arr.reshape(
            1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1
        )
        # anchor = cv_train_x[0][0].reshape(1, eval_samples[0].shape[0], 2, 1)
        # eval_arr_r = eval_arr.reshape(1, eval_samples[0].shape[0], 2, 1)

        anchor = torch.Tensor(anchor)
        eval_arr_r = torch.Tensor(eval_arr_r)
        output1, output2, output3 = net(
            anchor.cuda().permute(0, 3, 1, 2),
            eval_arr_r.cuda().permute(0, 3, 1, 2),
            eval_arr_r.cuda().permute(0, 3, 1, 2),
        )
        net_pred = distance_function(output1, output3)
        buy_pred.append(float(net_pred.to("cpu").numpy()))

        date.append(test_dates.Datetime[indexI + (PATTERN_SIZE - 1)])
        open.append(float(eval_samples[indexI][-1, [0]]))
        high.append(float(eval_samples[indexI][-1, [1]]))
        low.append(float(eval_samples[indexI][-1, [2]]))
        close.append(float(eval_samples[indexI][-1, [3]]))
        volume.append(float(eval_samples[indexI][-1, [4]]))
        train_data_shape.append(float(Train_df.shape[0]))

Predictions = pd.DataFrame(
    {
        "Datetime": date,
        "Open": open,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
        "Distance": buy_pred,
        "Train_shape": train_data_shape,
    }
)
Predictions.to_csv(
    f"{DESTINATION_ROOT}/test_results{FILENAME[:-4]}_extrw{EXTR_WINDOW}_patsize{PATTERN_SIZE}_ov{OVERLAP}.csv",
    index=False,
)
