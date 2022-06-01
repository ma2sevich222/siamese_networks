#######################################################
# Copyright © 2021-2099 Ekosphere. All rights reserved
# Author: Evgeny Matusevich
# Contacts: <ma2sevich222@gmail.com>
# File: model_tuning_run.py
#######################################################


import numpy as np
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from models.torch_models import shotSiameseNetwork
from utilits.project_functions import get_train_data, get_triplet_random, train_triplet_net, forward_trade, \
    find_best_dist, get_signals

# from utilits.strategies_Chekh import Long_n_Short_Strategy as LnS


""""""""""""""""""""""""""""" Parameters Block """""""""""""""""""""""""""
source = 'source_root'
out_root = 'outputs'
source_file_name = 'GC_2020_2022_30min.csv'
pattern_size = 143
extr_window = 200
overlap = 61
profit_value = 0.003
step = 0.01
train_window = 7000
select_dist_window = 7000
forward_window = 1000

""""""""""""""""""""""""""""" Net Parameters Block """""""""""""""""""""""""""
epochs = 7  # количество эпох
lr = 0.000009470240447408595  # learnig rate
embedding_dim = 160  # размер скрытого пространства
margin = 20  # маржа для лосс функции
batch_size = 150  # размер батчсайз
distance_function = lambda x, y: 1.0 - F.cosine_similarity(x, y)

df = pd.read_csv(f"{source}/{source_file_name}")  # Загрузка данных

n_iters = (len(df) - sum([train_window, select_dist_window, forward_window])) // forward_window
print(f'********************** Количество итераций = {n_iters}*********************************')

df_for_split = df.copy()
'''splited_dfs = []

for i in range(n_splits):
    if i > (n_splits + 1):
        df_sliced = df_for_split.iloc[:(len(df_for_split) // n_splits)]
        splited_dfs.append(df_sliced)
        df_for_split = df_for_split.iloc[(len(df_for_split) // n_splits):]
    else:
        splited_dfs.append(df_for_split)'''
iteration = 0
signals = []
for n in range(n_iters):
    # print(f'Размер среза {len(dat)}')
    train_df = df_for_split.iloc[:train_window]
    test_df = df_for_split.iloc[train_window:sum([train_window, select_dist_window])]
    forward_df = df_for_split.iloc[
                 sum([train_window, select_dist_window]):sum([train_window, select_dist_window, forward_window])]
    df_for_split = df_for_split.iloc[forward_window:]
    df_for_split = df_for_split.reset_index(drop=True)

    print('Новый срез')
    print(f'Размер обучающей выборки = {len(train_df)}')
    print(
        f"Период обучения с {train_df.loc[train_df.index[0], 'Datetime']} по {train_df.loc[train_df.index[-1], 'Datetime']}")
    print(f'Размер выборки для подбора расстояния = {len(test_df)}')
    print(
        f"Подбор расстояния с {test_df.loc[test_df.index[0], 'Datetime']} по {test_df.loc[test_df.index[-1], 'Datetime']}")
    print(f'Размер выборки для форвардного анализа расстояния = {len(forward_df)}')
    print(
        f"Форвардный анализ с {forward_df.loc[forward_df.index[0], 'Datetime']} по {forward_df.loc[forward_df.index[-1], 'Datetime']}")
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    forward_df = forward_df.reset_index(drop=True)
    train_dates = pd.DataFrame({'Datetime': train_df.Datetime.values})
    test_dates = pd.DataFrame({'Datetime': test_df.Datetime.values})
    forward_dates = pd.DataFrame({'Datetime': forward_df.Datetime.values})
    del train_df["Datetime"], test_df["Datetime"], forward_df["Datetime"]

    train_x, n_samples_to_train = get_train_data(train_df, profit_value, extr_window, pattern_size, overlap,
                                                 train_dates)  # получаем данные для создания триплетов

    n_classes = len(train_x)
    train_triplets = get_triplet_random(n_samples_to_train, n_classes, train_x)
    print(" Размер данных для обучения:", np.array(train_triplets).shape)

    tensor_A = torch.Tensor(train_triplets[0])  # transform to torch tensor
    tensor_P = torch.Tensor(train_triplets[1])
    tensor_N = torch.Tensor(train_triplets[2])

    my_dataset = TensorDataset(tensor_A, tensor_P, tensor_N)  # create your datset
    my_dataloader = DataLoader(my_dataset, batch_size=batch_size)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""" Train net  """""""""""""""""""""""""""

    net = shotSiameseNetwork(embedding_dim=embedding_dim).cuda()
    train_triplet_net(lr, epochs, my_dataloader, net, distance_function)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""" Test data prepare  """""""""""""""""""""""""""
    scaler = StandardScaler()
    eval_array = test_df.to_numpy()
    eval_samples = [eval_array[i - pattern_size:i] for i in range(len(eval_array)) if i - pattern_size >= 0]
    eval_normlzd = [scaler.fit_transform(i) for i in eval_samples]
    eval_normlzd = np.array(eval_normlzd).reshape(-1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""" Test model  """""""""""""""""""""""""""

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
        for indexI, eval_arr in enumerate(eval_normlzd):
            anchor = train_x[0][0].reshape(1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1)
            eval_arr_r = eval_arr.reshape(1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1)
            anchor = torch.Tensor(anchor)
            eval_arr_r = torch.Tensor(eval_arr_r)
            output1, output2, output3 = net(anchor.cuda().permute(0, 3, 1, 2),
                                            eval_arr_r.cuda().permute(0, 3, 1, 2),
                                            eval_arr_r.cuda().permute(0, 3, 1, 2))
            net_pred = distance_function(output1, output3)
            buy_pred.append(float(net_pred.to('cpu').numpy()))

            date.append(test_dates.Datetime[indexI + (pattern_size - 1)])
            open.append(float(eval_samples[indexI][-1, [0]]))
            high.append(float(eval_samples[indexI][-1, [1]]))
            low.append(float(eval_samples[indexI][-1, [2]]))
            close.append(float(eval_samples[indexI][-1, [3]]))
            volume.append(float(eval_samples[indexI][-1, [4]]))
            train_data_shape.append(float(train_df.shape[0]))

    test_result = pd.DataFrame(
        {'Datetime': date, 'Open': open, 'High': high, 'Low': low, 'Close': close, 'Volume': volume,
         'Distance': buy_pred,
         'Train_shape': train_data_shape})

    buy_before, sell_after = find_best_dist(test_result, step)

    print(f'BUY BEFORE = {buy_before}')
    print(f'SELL AFTER = {sell_after}')

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""" Forward data prepare  """""""""""""""""""""""""""
    scaler = StandardScaler()
    eval_array = forward_df.to_numpy()
    eval_samples = [eval_array[i - pattern_size:i] for i in range(len(eval_array)) if i - pattern_size >= 0]
    eval_normlzd = [scaler.fit_transform(i) for i in eval_samples]
    eval_normlzd = np.array(eval_normlzd).reshape(-1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """"""""""""""""""""""""""""" Forward model  """""""""""""""""""""""""""

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
        for indexI, eval_arr in enumerate(eval_normlzd):
            anchor = train_x[0][0].reshape(1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1)
            eval_arr_r = eval_arr.reshape(1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1)
            anchor = torch.Tensor(anchor)
            eval_arr_r = torch.Tensor(eval_arr_r)
            output1, output2, output3 = net(anchor.cuda().permute(0, 3, 1, 2),
                                            eval_arr_r.cuda().permute(0, 3, 1, 2),
                                            eval_arr_r.cuda().permute(0, 3, 1, 2))
            net_pred = distance_function(output1, output3)
            buy_pred.append(float(net_pred.to('cpu').numpy()))

            date.append(forward_dates.Datetime[indexI + (pattern_size - 1)])
            open.append(float(eval_samples[indexI][-1, [0]]))
            high.append(float(eval_samples[indexI][-1, [1]]))
            low.append(float(eval_samples[indexI][-1, [2]]))
            close.append(float(eval_samples[indexI][-1, [3]]))
            volume.append(float(eval_samples[indexI][-1, [4]]))
            train_data_shape.append(float(train_df.shape[0]))

    forward_result = pd.DataFrame(
        {'Datetime': date, 'Open': open, 'High': high, 'Low': low, 'Close': close, 'Volume': volume,
         'Distance': buy_pred,
         'Train_shape': train_data_shape})

    signal = get_signals(forward_result, buy_before, sell_after)
    signals.append(signal)
    iteration += 1
    print(f'*************** Итерация No. = {iteration}  завершена')

signals_combained = pd.concat(signals, ignore_index=True, sort=False)

forward_trade(signals_combained, out_root, source_file_name, pattern_size, extr_window, overlap)
