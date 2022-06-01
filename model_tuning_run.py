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
from constants import *
from models.torch_models import SiameseNetwork
from utilits.project_functions import (
    unstandart_data_load,
    get_train_data,
    get_triplet_random,
    train_triplet_net,
    euclid_dist,
)
import optuna
from utilits.data_load import data_load_OHLCV

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
"""""" """""" """""" """""" """"" Parameters Block """ """""" """""" """""" """"""

profit_value = 0.0015
epochs = 100  # количество эпох
lr = 0.0009470240447408595  # learnig rate
embedding_dim = 165  # размер скрытого пространства
margin = 2  # маржа для лосс функции
batch_size = 100  # размер батчсайз
# = 1000  # Количество триплетов для тренировки
tresh_hold = (
    15  # граница предсказаний ниже которой предсказания будут отображаться на графике
)

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
"""""" """""" """""" """""" """"" Distance functions """ """""" """""" """""" """"""
# distance_function = lambda x, y: 1.0 - F.cosine_similarity(x, y)  # функция расчета расстояния для триплет лосс
# distance_function = PairwiseDistance(p=2, eps=1e-06,)
# distance_function = l_infinity
distance_function = euclid_dist
# distance_function = manhatten_dist

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
"""""" """""" """""" """""" """"" Data load,clean and prepare  """ """""" """""" """""" """"""

"""indices = [
    i for i, x in enumerate(FILENAME) if x == "_"
]  # находим индексы вхождения '_'
ticker = FILENAME[: indices[0]]"""

Train_df, Eval_df, train_dates, test_dates = data_load_OHLCV(
    SOURCE_ROOT, FILENAME
)  # загрузка данных
train_x, n_samples_to_train = get_train_data(
    Train_df, profit_value, EXTR_WINDOW, PATTERN_SIZE, OVERLAP, train_dates
)  # получаем данные для создания триплетов
n_classes = len(train_x)
train_x = np.array(train_x, dtype=object)

train_triplets = get_triplet_random(n_samples_to_train, n_classes, train_x)

print(" Размер данных для обучения:", np.array(train_triplets).shape)

tensor_A = torch.Tensor(train_triplets[0])  # transform to torch tensor
tensor_P = torch.Tensor(train_triplets[1])
tensor_N = torch.Tensor(train_triplets[2])

my_dataset = TensorDataset(tensor_A, tensor_P, tensor_N)  # create your datset
my_dataloader = DataLoader(my_dataset, batch_size=batch_size)


def objective(trial):
    # boundaries for the optimizer's
    lr = trial.suggest_loguniform("lr", 1e-8, 1e-2)
    tbatch_size = trial.suggest_int("batch_size", 5, 245, step=5)
    embedding_dim = trial.suggest_int("embedding_dim", 5, 400, step=5)
    my_dataset = TensorDataset(tensor_A, tensor_P, tensor_N)
    my_dataloader = DataLoader(my_dataset, batch_size=tbatch_size)
    ##### If you need more parameters for optimization, it is done like this:
    # new_parameter =  trial.suggest_loguniform("new_parameter", lower_bound, upper_bound)
    tdistance_function = distance_function
    # create new model(and all parameters) every iteration
    model = SiameseNetwork(embedding_dim=embedding_dim).cuda()

    _, last_epoch_loss = train_triplet_net(
        lr, epochs, my_dataloader, model, tdistance_function
    )
    return last_epoch_loss


# Create "exploration"
study = optuna.create_study(direction="minimize", study_name="Optimal lr")


study.optimize(objective, n_trials=10)

print(study.best_params)
