import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from all_pytorch.torch_functions_for_train import train_net, cos_em_create_pairs
from all_pytorch.torch_models import SiameseNetwork
from utilits.data_load import data_load
from utilits.functions_for_train_nn import get_locals, get_patterns, get_train_samples

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""" Parameters Block """""""""""""""""""""""""""
from constants import *

epochs = 5
lr = 0.0005
embeddig_dim = 10
margin = 0.5
batch_size = 10

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""" Main Block """""""""""""""""""""""""""""""""
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

Min_train_locals, Max_train__locals = get_locals(Train_df, EXTR_WINDOW)

buy_patern, sell_patern = get_patterns(
    Train_df.to_numpy(),
    Min_train_locals["index"].values.tolist(),
    Max_train__locals["index"].values.tolist(),
    PATTERN_SIZE,
)

buy_reshaped = buy_patern.reshape(buy_patern.shape[0], -1)
np.savetxt(f"{DESTINATION_ROOT}/buy_patterns_extr_window{EXTR_WINDOW}"
           f"_pattern_size{PATTERN_SIZE}.csv", buy_reshaped)

print(f"Найдено уникальных:\n"
      f"buy_patern.shape: {buy_patern.shape}\t|\tsell_patern.shape: {sell_patern.shape}")

"""Получаем Xtrain и Ytrain для обучения сети"""
Xtrain, Ytrain = get_train_samples(buy_patern, sell_patern)
"""Нормализуем Xtrain"""
X_norm = [normalize(i, axis=0, norm="max") for i in Xtrain]
"""решейпим для подачи в сеть"""
X_norm = np.array(X_norm).reshape(
    -1, buy_patern[0].shape[0], buy_patern[0][0].shape[0], 1
)
Ytrain = Ytrain.reshape(-1, 1)

""" Получаем пары """
digit_indices = [np.where(Ytrain == i)[0] for i in range(num_classes)]
tr_pairs, tr_y = cos_em_create_pairs(X_norm, digit_indices, num_classes)

tensor_x1 = torch.Tensor(tr_pairs[:, 0])  # transform to torch tensor
tensor_x2 = torch.Tensor(tr_pairs[:, 1])
tensor_y = torch.Tensor(tr_y)

my_dataset = TensorDataset(tensor_x1, tensor_x2, tensor_y)  # create your datset
my_dataloader = DataLoader(my_dataset, batch_size=batch_size)

net = SiameseNetwork(embeddig_dim=embeddig_dim).cuda()
cos_crit = torch.nn.CosineEmbeddingLoss(margin=margin)

train_net(cos_crit, lr, epochs, my_dataloader, net, labels_1d=False)  # crit, lr, epochs, my_dataloader,net,

"""Тест модели"""

eval_array = Eval_df.to_numpy()
eval_samples = [eval_array[i - PATTERN_SIZE:i] for i in range(len(eval_array)) if i - PATTERN_SIZE >= 0]
eval_normlzd = [normalize(i, axis=0, norm='max') for i in eval_samples]
eval_normlzd = np.array(eval_normlzd).reshape(-1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1)

Min_prediction_pattern_name = []
date = []
open = []
high = []
low = []
close = []
volume = []
distance = []
signal = []  # лэйбл
k = 0

with torch.no_grad():
    for indexI, eval_arr in enumerate(tqdm(eval_normlzd)):

        buy_predictions = []
        for buy in buy_patern:
            buy = buy.reshape(1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1)
            eval_arr = eval_arr.reshape(1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1)
            buy = torch.Tensor(buy)
            eval_arr = torch.Tensor(eval_arr)
            output1, output2 = net(buy.cuda().permute(0, 3, 1, 2), eval_arr.cuda().permute(0, 3, 1, 2))
            cos = torch.nn.CosineSimilarity()
            buy_pred = cos(output1, output2)
            buy_predictions.append(buy_pred)

        date.append(Eval_dates_str[indexI + (PATTERN_SIZE - 1)])
        open.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [0]]))
        high.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [1]]))
        low.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [2]]))
        close.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [3]]))
        volume.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [4]]))
        Min_prediction_pattern_name.append(buy_predictions.index(min(buy_predictions)))

        min_ex = min(buy_predictions)
        distance.append(float(min_ex))

        if min_ex <= TRESHHOLD_DISTANCE:
            signal.append(1)
        else:
            signal.append(0)

Predictions = pd.DataFrame(
    list(zip(date, open, high, low, close, volume, signal, Min_prediction_pattern_name, distance)),
    columns=['date', 'open', 'high', 'low', 'close', 'volume', 'signal', 'pattern', 'distance'])

Predictions.to_csv(f'{DESTINATION_ROOT}/test_results_extr_window{EXTR_WINDOW}'
                   f'_pattern_size{PATTERN_SIZE}.csv')
