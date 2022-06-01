#######################################################
# Copyright © 2021-2099 Ekosphere. All rights reserved
# Author: Evgeny Matusevich
# Contacts: <ma2sevich222@gmail.com>
# File: bt_run.py
#######################################################


import backtesting._plotting as plt_backtesting
import numpy as np
import pandas as pd
import plotly.express as px
import torch
from backtesting import Backtest
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from constants import SOURCE_ROOT, DESTINATION_ROOT, FILENAME, TRAIN_WINDOW, profit_value, START_TEST, END_TEST
from models.torch_models import shotSiameseNetwork
from utilits.data_load import data_load_OHLCV, data_load_CL
from utilits.project_functions import get_triplet_random, train_triplet_net, get_train_data, get_CLtrain_data
# from utilits.strategies_Chekh import Long_n_Short_Strategy as LnS
from utilits.strategies_AT import Long_n_Short_Strategy_Float as LnSF


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""" Net Parameters Block """""""""""""""""""""""""""
epochs = 7  # количество эпох
lr = 0.000009470240447408595  # learnig rate
embedding_dim = 160  # размер скрытого пространства
margin = 20  # маржа для лосс функции
batch_size = 150  # размер батчсайз
distance_function = lambda x, y: 1.0 - F.cosine_similarity(x, y)

result_filename = f'{DESTINATION_ROOT}/bt_run_outputs/hyp_parameters_select_{FILENAME[:-4]}_step'
df_stats_list = []
runs = 0
for PATTERN_SIZE in tqdm(range(50, 60, 10), desc=" Прогресс подбора "):
    runs += 0
    for EXTR_WINDOW in range(100, 110, 10):
        for OVERLAP in range(0, 10, 10):
            print(
                '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print(
                f'Начало тестирования для параметров PATTERN_SIZE = {PATTERN_SIZE}, EXTR_WINDOW = {EXTR_WINDOW}, OVERLAP = {OVERLAP} ')
            print(
                '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            Train_df, Eval_df, train_dates, test_dates = data_load_OHLCV(SOURCE_ROOT, FILENAME, START_TEST, END_TEST,
                                                                      PATTERN_SIZE, TRAIN_WINDOW)  # загрузка данных
            train_x, n_samples_to_train = get_train_data(Train_df, profit_value, EXTR_WINDOW, PATTERN_SIZE, OVERLAP,
                                                           train_dates)  # получаем данные для создания триплетов
            n_classes = len(train_x)

            print(f'Дата начала тестирования : {test_dates.values[PATTERN_SIZE - 1]}')

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
            eval_array = Eval_df.to_numpy()
            eval_samples = [eval_array[i - PATTERN_SIZE:i] for i in range(len(eval_array)) if i - PATTERN_SIZE >= 0]
            eval_normlzd = [scaler.fit_transform(i) for i in eval_samples]
            eval_normlzd = np.array(eval_normlzd).reshape(-1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1)
            '''eval_array = Eval_df[
                ['DiffEMA', 'SmoothDiffEMA', 'VolatilityTunnel', 'BuyIntense', 'SellIntense']].to_numpy()
            eval_samples = [eval_array[i - PATTERN_SIZE:i] for i in range(len(eval_array)) if i - PATTERN_SIZE >= 0]
            eval_ohlcv = Eval_df[['Open', 'High', 'Low', 'Close', 'Volume']].to_numpy()
            ohlcv_samples = [eval_ohlcv[i - PATTERN_SIZE:i] for i in range(len(eval_ohlcv)) if i - PATTERN_SIZE >= 0]'''
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

                    date.append(test_dates.Datetime[indexI + (PATTERN_SIZE - 1)])
                    open.append(float(eval_samples[indexI][-1, [0]]))
                    high.append(float(eval_samples[indexI][-1, [1]]))
                    low.append(float(eval_samples[indexI][-1, [2]]))
                    close.append(float(eval_samples[indexI][-1, [3]]))
                    volume.append(float(eval_samples[indexI][-1, [4]]))
                    train_data_shape.append(float(Train_df.shape[0]))

            df = pd.DataFrame(
                {'Datetime': date, 'Open': open, 'High': high, 'Low': low, 'Close': close, 'Volume': volume,
                 'Distance': buy_pred,
                 'Train_shape': train_data_shape})

            """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
            """"""""""""""""""""""""""""" Backtest model  """""""""""""""""""""""""""

            plt_backtesting._MAX_CANDLES = 100_000
            pd.pandas.set_option('display.max_columns', None)
            pd.set_option("expand_frame_repr", False)
            pd.options.display.expand_frame_repr = False
            pd.set_option("precision", 2)

            df.set_index('Datetime', inplace=True)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df['Signal'] = 0
            print('******* Результы предсказания сети *******')
            print(df)
            print()

            """ Параметры тестирования """
            i = 0
            deposit = 40000000000000000000000000  # сумма одного контракта GC & CL
            comm = 4.6  # GC - комиссия для золота
            # comm = 4.52  # CL - комиссия для нейти
            sell_after = 1.6
            buy_before = 0.6
            step = 0.1  # с каким шагом проводим тест разметки
            # result_filename = f'{DESTINATION_ROOT}/selection_distances_{FILENAME[:-4]}_step{step}'

            """ Тестирвоание """

            df_stats = pd.DataFrame()
            for sell_after in range(int(1 / step), int(round(df.Distance.max(), 1) / step)):
                for buy_before in range(int(round(df.Distance.min(), 1) / step), int(1 / step)):
                    # print(f'Диапазон Distance from {sell_trash/10} to {buy_trash/10}')
                    df['Signal'].where(~(df.Distance >= sell_after * step), -1, inplace=True)
                    df['Signal'].where(~(df.Distance <= buy_before * step), 1, inplace=True)
                    # df['Signal'] = np.roll(df.Signal, 2)

                    # сделаем так, чтобы 0 расценивался как "держать прежнюю позицию"
                    df.loc[df['Signal'] == 0, 'Signal'] = np.nan  # заменим 0 на nan
                    df['Signal'] = df['Signal'].ffill()  # заменим nan на предыдущие значения
                    df.dropna(axis=0, inplace=True)  # Удаляем наниты
                    df = df.loc[df['Signal'] != 0]
                    df_duplicated = df[df.index.duplicated(keep=False)].sort_index()  # проверка дубликатов
                    assert df_duplicated.shape[0] == 0, "В коде существуют дубликаты!"
                    # оставим только не нулевые строки
                    bt = Backtest(df, LnSF, cash=deposit, commission=0.00, trade_on_close=True)
                    stats = bt.run(deal_amount='fix', fix_sum=200000000000000000000)[:27]
                    '''if stats['Return (Ann.) [%]'] > 0:  # будем показывать и сохранять только доходные разметки
                         bt.plot(plot_volume=True, relative_equity=True,
                          filename=f'{result_filename}_{buy_before * step}_{sell_after * step}.html'
                          )'''
                    df_stats = df_stats.append(stats, ignore_index=True)
                    df_stats.loc[i, 'Net Profit [$]'] = df_stats.loc[i, 'Equity Final [$]'] - deposit - df_stats.loc[
                        i, '# Trades'] * comm
                    df_stats.loc[i, 'buy_before'] = buy_before * step
                    df_stats.loc[i, 'sell_after'] = sell_after * step
                    df_stats.loc[i, 'train_window'] = TRAIN_WINDOW
                    df_stats.loc[i, 'pattern_size'] = PATTERN_SIZE
                    df_stats.loc[i, 'extr_window'] = EXTR_WINDOW
                    df_stats.loc[i, 'profit_value'] = profit_value
                    df_stats.loc[i, 'overlap'] = OVERLAP

                    '''if df_stats.loc[i, 'Net Profit [$]'].item() > 0:  # сохраняем только данные с положительным профитом
                        bt.plot(plot_volume=True, relative_equity=True,
                                filename=f'{DESTINATION_ROOT}/bt_run_outputs/bt_plot/bt_plot_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}_step{step}_{buy_before * step}_{sell_after * step}.html'
                                )
                        df.to_csv(
                            f'{DESTINATION_ROOT}/bt_run_outputs/signals/signals_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}_step{step}_{buy_before * step}_{sell_after * step}.csv')
                        stats.to_csv(
                            f'{DESTINATION_ROOT}/bt_run_outputs/stats/stats_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}_step{step}_{buy_before * step}_{sell_after * step}.txt')'''
                    i += 1

            df_stats_list.append(df_stats)

    if runs == 0:
        df_hyp_parameters = pd.concat(df_stats_list, ignore_index=True, sort=False)
        df_hyp_parameters.sort_values(by="Net Profit [$]", ascending=False).to_excel(
            f'{result_filename}_intermedia.xlsx')
        runs = 0
        print('<<<<<<<<<<<< Промежуеточный файл сохранен >>>>>>>>>>>>>>')

df_hyp_parameters = pd.concat(df_stats_list, ignore_index=True, sort=False)
df_hyp_parameters.sort_values(by="Net Profit [$]", ascending=False).to_excel(f'{result_filename}.xlsx')
# print(df_stats)

df_plot = df_hyp_parameters[
    ['Net Profit [$]', 'pattern_size', 'extr_window', 'overlap', 'train_window']]
fig = px.parallel_coordinates(df_plot, color="Net Profit [$]",
                              labels={"Net Profit [$]": "Net Profit ($)", "buy_before": "buy_before dist",
                                      "sell_after": "sell_after dist", "pattern_size": "pattern_size (bars)",
                                      "extr_window": "extr_window (bars)", "train_window": "train_window (bars)",
                                      "overlap": "overlap (bars)"},
                              range_color=[df_plot['Net Profit [$]'].min(), df_plot['Net Profit [$]'].max()],
                              color_continuous_scale=px.colors.sequential.Viridis,
                              title=f'hyper parameters select {FILENAME[:-4]}')

fig.write_html(f"{DESTINATION_ROOT}/bt_run_outputs/hyp_parameters_select_{FILENAME[:-4]}.html")  # сохраняем в файл
fig.show()
