#######################################################
# Copyright © 2021-2099 Ekosphere. All rights reserved
# Author: Evgeny Matusevich
# Contacts: <ma2sevich222@gmail.com>
# File: backtest_for_Matusevich.py
#######################################################
import os

import backtesting._plotting as plt_backtesting
import numpy as np
import pandas as pd
from backtesting import Backtest
from tqdm import trange

from constants import DESTINATION_ROOT, PATTERN_SIZE, EXTR_WINDOW, profit_value, OVERLAP, FILENAME
# from utilits.strategies_Chekh import Long_n_Short_Strategy as LnS
from utilits.strategies_AT import Long_n_Short_Strategy_Float as LnSF

# from utilits.strategies_hold_long_short_ import Long_n_Short_Strategy as LnSS

plt_backtesting._MAX_CANDLES = 100_000
pd.pandas.set_option('display.max_columns', None)
pd.set_option("expand_frame_repr", False)
pd.options.display.expand_frame_repr = False
pd.set_option("precision", 2)

source_root = DESTINATION_ROOT
source_filename = 'test_resultsGC_2020_2022_60min_extrw140_patsize86_ov10.csv'
out_root = f'{FILENAME[:-4]}_only_best'
os.mkdir(f'{DESTINATION_ROOT}/{out_root}')

""" Откроем файл с разметкой нейронки """
df = pd.read_csv(f'{source_root}/{source_filename}')
df.set_index('Datetime', inplace=True)
df.index = pd.to_datetime(df.index)
df = df.sort_index()
df['Signal'] = 0
print(df)
print()

""" Параметры тестирования """
i = 0
deposit = 40000  # сумма одного контракта GC & CL
comm = 4.6  # GC - комиссия для золота
# comm = 4.52  # CL - комиссия для нейти
sell_after = 1.6
buy_before = 0.6
step = 0.1  # с каким шагом проводим тест разметки
result_filename = f'{DESTINATION_ROOT}/{out_root}/hyp_parameters_select_{source_filename[:-4]}_step{step}'

""" Тестирвоание """""" Тестирвоание """
df_stats = pd.DataFrame()
for sell_after in trange(int(1 / step), int(round(df.Distance.max(), 1) / step)):
    for buy_before in range(int(round(df.Distance.min(), 1) / step), int(1 / step)):
        # print(f'Диапазон Distance from {sell_trash/10} to {buy_trash/10}')
        df['Signal'].where(~(df.Distance >= sell_after * step), -1, inplace=True)
        df['Signal'].where(~(df.Distance <= buy_before * step), 1, inplace=True)
        # df['Signal'] = np.roll(df.Signal, 2)

        # сделаем так, чтобы 0 расценивался как "держать прежнюю позицию"
        df.loc[df['Signal'] == 0, 'Signal'] = np.nan  # заменим 0 на nan
        df['Signal'] = df['Signal'].ffill()  # заменим nan на предыдущие значения
        df.dropna(axis=0, inplace=True)  # Удаляем наниты
        df = df.loc[df['Signal'] != 0]  # оставим только не нулевые строки
        df.to_csv(
            f'{DESTINATION_ROOT}/{out_root}/signals_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}_step{step}_{buy_before * step}_{sell_after * step}.csv')

        bt = Backtest(df, LnSF, cash=deposit, commission=0.00, trade_on_close=True)
        stats = bt.run(deal_amount='fix', fix_sum=20000)[:27]
        if stats['Return (Ann.) [%]'] > 0:  # будем показывать и сохранять только доходные разметки
            bt.plot(plot_volume=True, relative_equity=True,
                    filename=f'{DESTINATION_ROOT}/{out_root}/bt_plot_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}_step{step}_{buy_before * step}_{sell_after * step}.html'
                    )
        stats.to_csv(
            f'{DESTINATION_ROOT}/{out_root}/stats_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}_step{step}_{buy_before * step}_{sell_after * step}.txt')

        df_stats = df_stats.append(stats, ignore_index=True)
        df_stats.loc[i, 'Net Profit [$]'] = df_stats.loc[i, 'Equity Final [$]'] - deposit - df_stats.loc[
            i, '# Trades'] * comm
        df_stats.loc[i, 'buy_before'] = buy_before * step
        df_stats.loc[i, 'sell_after'] = sell_after * step
        df_stats.loc[i, 'train_window'] = int(df['Train_shape'].iloc[0])
        df_stats.loc[i, 'pattern_size'] = PATTERN_SIZE
        df_stats.loc[i, 'extr_window'] = EXTR_WINDOW
        df_stats.loc[i, 'profit_value'] = profit_value
        df_stats.loc[i, 'overlap'] = OVERLAP
        i += 1

df_stats.sort_values(by="Net Profit [$]", ascending=False).to_excel(f'{result_filename}.xlsx')
print(df_stats)
