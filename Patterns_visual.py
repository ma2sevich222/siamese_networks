#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 22:14:41 2022

@author: ma2sevich
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics.pairwise import cosine_distances

n_size = 20  # задаем размер батча

DATA = pd.read_csv('source_root/VZ_15_Minutes_(with_indicators)_2018_18012022.txt', delimiter=",")
NEW_DATA = DATA[['<Date>', ' <Time>', ' <Open>', ' <High>', ' <Low>', ' <Close>', ' <Volume>']].copy()
names = NEW_DATA.columns.to_list()
NEW_DATA = NEW_DATA[~(NEW_DATA == 0).any(axis=1)]
Date = pd.to_datetime(NEW_DATA['<Date>'] + NEW_DATA[' <Time>'], format='%d.%m.%Y%H:%M:%S')
NEW_DATA.drop(['<Date>', ' <Time>'], axis=1, inplace=True)
NEW_DATA.insert(0, 'Date', Date)
NEW_DATA.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
NEW_DATA = NEW_DATA[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
NEW_DATA['SMA'] = NEW_DATA.iloc[:, 3].rolling(window=10).mean()
NEW_DATA['CMA30'] = NEW_DATA['Close'].expanding().mean()
NEW_DATA['SMA'] = NEW_DATA['SMA'].fillna(0)

Eval_df = NEW_DATA.loc[10000:]
Eval_dates = Eval_df[['Date']]
Eval_df.drop(['Date'], axis=1, inplace=True)
Eval_df = Eval_df.reset_index(drop=True)

loader = np.loadtxt("outputs/buy_patterns.txt")  # загружаем паттерны
patterns = loader.reshape(-1, 20, 7)  # решейпим в оригинальный размер

Results = pd.read_csv('outputs/pattern_model_test.csv', index_col=[0])
Results = Results.rename(columns={"pattern No.": "pattern"})
paterns_to_df = [pd.DataFrame(i, columns=['open', 'high', 'low', 'close', 'volume', 'feata', 'featb']) for i in
                 patterns]
Eval_ = [Eval_df[i - 20:i] for i in Eval_df.index if (i - 20) >= 0]  # переводим в df для отрисовки


def pattern_samples_plot(patterns, Eval_df, eval_results, pattern_No):  # df,list of df(cutted),df

    indexes = eval_results[(eval_results.signal == 1) & (eval_results.pattern == pattern_No)].index
    if len(indexes) == 0:
        print('Данный паттерн не был обнаружен в данных')
    else:
        print(f"Найдено совпадений: {len(indexes)}")
        for i in indexes:
            plot_sample = Eval_df[i].reset_index()
            plt.figure(figsize=[5, 5])
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 3))
            ax1.set(ylabel='PRICE',
                    title=f'Pattern: {pattern_No}')
            ax2.set(ylabel='PRICE',
                    title=f'Step: {i}/ distance:{round(eval_results.distance[i], 4)}')
            width = .4
            width2 = .05

            ax1_up = patterns[pattern_No][patterns[pattern_No].close >= patterns[pattern_No].open]
            ax1_down = patterns[pattern_No][patterns[pattern_No].close < patterns[pattern_No].open]
            ax2_up = plot_sample[plot_sample.Close >= plot_sample.Open]
            ax2_down = plot_sample[plot_sample.Close < plot_sample.Open]

            col1 = 'green'
            col2 = 'red'

            ax1.bar(ax1_up.index, ax1_up.close - ax1_up.open, width, bottom=ax1_up.open, color=col1)
            ax1.bar(ax1_up.index, ax1_up.high - ax1_up.close, width2, bottom=ax1_up.close, color=col1)
            ax1.bar(ax1_up.index, ax1_up.low - ax1_up.open, width2, bottom=ax1_up.open, color=col1)
            ax1.bar(ax1_down.index, ax1_down.close - ax1_down.open, width, bottom=ax1_down.open, color=col2)
            ax1.bar(ax1_down.index, ax1_down.high - ax1_down.open, width2, bottom=ax1_down.open, color=col2)
            ax1.bar(ax1_down.index, ax1_down.low - ax1_down.close, width2, bottom=ax1_down.close, color=col2)

            ax2.bar(ax2_up.index, ax2_up.Close - ax2_up.Open, width, bottom=ax2_up.Open, color=col1)
            ax2.bar(ax2_up.index, ax2_up.High - ax2_up.Close, width2, bottom=ax2_up.Close, color=col1)
            ax2.bar(ax2_up.index, ax2_up.Low - ax2_up.Open, width2, bottom=ax2_up.Open, color=col1)
            ax2.bar(ax2_down.index, ax2_down.Close - ax2_down.Open, width, bottom=ax2_down.Open, color=col2)
            ax2.bar(ax2_down.index, ax2_down.High - ax2_down.Open, width2, bottom=ax2_down.Open, color=col2)
            ax2.bar(ax2_down.index, ax2_down.Low - ax2_down.Close, width2, bottom=ax2_down.Close, color=col2)

    plt.show()


def calculate_cos_dist(patterns, pattern):
    pattern_array = patterns[pattern].reshape(-1, patterns[pattern].shape[0] * patterns[pattern].shape[1])
    pattern_matrix = patterns.reshape(-1, patterns.shape[1] * patterns.shape[2])
    cos_distance_array = cosine_distances(pattern_array, pattern_matrix)
    nearlest_patterns_names = np.argsort(cos_distance_array[0])
    nearlest_pattern_distances = np.sort(cos_distance_array[0])
    nearlist_neibors = dict(zip(nearlest_patterns_names[:10], nearlest_pattern_distances[:10]))

    return nearlist_neibors


def plot_nearlist_patterns(patterns, nearlist_neibors):  # передаем массив паттернов и список ближайших паттернов

    keys = list(nearlist_neibors.keys())
    values = list(nearlist_neibors.values())

    fig = make_subplots(rows=3, cols=4, subplot_titles=("Образец", f"Паттерн:{keys[1]} дистанция:{round(values[1], 4)}",
                                                        f"Паттерн: {keys[2]} дистанция: {round(values[2], 4)}",
                                                        f"Паттерн: {keys[3]} дистанция: {round(values[3], 4)}",
                                                        f"Паттерн: {keys[4]} дистанция: {round(values[4], 4)}",
                                                        f"Паттерн: {keys[5]} дистанция: {round(values[5], 4)}",
                                                        f"Паттерн: {keys[6]} дистанция: {round(values[6], 4)}",
                                                        f"Паттерн: {keys[7]} дистанция: {round(values[7], 4)}",
                                                        f"Паттерн: {keys[8]} дистанция: {round(values[8], 4)}",
                                                        f"Паттерн: {keys[9]} дистанция: {round(values[9], 4)}"))

    def plot_candy(pattern, row, col):
        fig.add_trace(go.Candlestick(x=np.array([i for i in range(len(pattern))]),
                                     open=pattern['open'].values,
                                     high=pattern['high'].values,
                                     low=pattern['low'].values,
                                     close=pattern['close'].values), secondary_y=False, row=row, col=col)

    plot_candy(patterns[keys[0]], 1, 1)
    plot_candy(patterns[keys[1]], 1, 2)
    plot_candy(patterns[keys[2]], 1, 3)
    plot_candy(patterns[keys[3]], 1, 4)
    plot_candy(patterns[keys[4]], 2, 1)
    plot_candy(patterns[keys[5]], 2, 2)
    plot_candy(patterns[keys[6]], 2, 3)
    plot_candy(patterns[keys[7]], 2, 4)
    plot_candy(patterns[keys[8]], 3, 1)
    plot_candy(patterns[keys[9]], 3, 2)

    fig.update_xaxes(rangeslider={'visible': False}, row=1, col=1)
    fig.update_xaxes(rangeslider={'visible': False}, row=1, col=2)
    fig.update_xaxes(rangeslider={'visible': False}, row=1, col=3)
    fig.update_xaxes(rangeslider={'visible': False}, row=1, col=4)
    fig.update_xaxes(rangeslider={'visible': False}, row=2, col=1)
    fig.update_xaxes(rangeslider={'visible': False}, row=2, col=2)
    fig.update_xaxes(rangeslider={'visible': False}, row=2, col=3)
    fig.update_xaxes(rangeslider={'visible': False}, row=2, col=4)
    fig.update_xaxes(rangeslider={'visible': False}, row=3, col=1)
    fig.update_xaxes(rangeslider={'visible': False}, row=3, col=2)

    fig.update_layout(title_text="Визуальное сравнение паттернов . Слева на право : образец и 9 ближайших по возрастанию дистанции от образца.")

    fig.show()  # на выходе слева на право интересующий нас паттерн и 9 ближайших


pattern_samples_plot(paterns_to_df, Eval_, Results, 48)
nearlist_neibors = calculate_cos_dist(patterns, 48)
plot_nearlist_patterns(paterns_to_df, nearlist_neibors)
