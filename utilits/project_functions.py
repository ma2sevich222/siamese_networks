import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.signal import argrelmin, argrelmax
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.nn import TripletMarginWithDistanceLoss
import backtesting._plotting as plt_backtesting
from backtesting import Backtest
from tqdm import trange
import torch.nn as nn
import torchbnn as bnn

# from utilits.strategies_AT import Long_n_Short_Strategy_Float as LnSF
import os
import pandas as pd
import plotly.express as px
from utilits.lazy_strategy import LazyStrategy


# функция извлечения паттернов / возвращает массив [ [паттерны класса buy],[паттерны класса sell] ]
def get_train_data(
    data_df, profit_value, EXTR_WINDOW, PATTERN_SIZE, OVERLAP, train_dates
):
    close_price = data_df["Close"].values.tolist()

    min_indexes = argrelmin(np.array(close_price), order=EXTR_WINDOW)[0]
    min_indexes = [
        i
        for i in min_indexes
        if (i + EXTR_WINDOW <= len(data_df) - 1) and (i + OVERLAP <= len(data_df) - 1)
    ]
    max_indexes = argrelmax(np.array(close_price), order=EXTR_WINDOW)[0]
    max_indexes = [
        i
        for i in max_indexes
        if (i + EXTR_WINDOW <= len(data_df) - 1) and (i + OVERLAP <= len(data_df) - 1)
    ]

    indexes_with_profit = []
    for i in min_indexes:
        if i + (EXTR_WINDOW) <= len(close_price):

            if (
                close_price[int(i) : int(i + (EXTR_WINDOW))][-1] - close_price[i]
                >= close_price[i] * profit_value
            ):
                indexes_with_profit.append(i)

    indexes_lost_profit = []
    for i in max_indexes:
        if i + (EXTR_WINDOW) <= len(close_price):
            if close_price[int(i) : int(i + (EXTR_WINDOW))][-1] - close_price[i] <= -(
                close_price[i] * profit_value
            ):
                indexes_lost_profit.append(i)

    patterns = []

    for ind in indexes_with_profit:
        if ind - PATTERN_SIZE >= 0:
            patt = data_df[(ind - PATTERN_SIZE + OVERLAP) : ind + OVERLAP].to_numpy()
            patterns.append(patt)

    sell_patterns = []
    for ind in indexes_lost_profit:
        if ind - PATTERN_SIZE >= 0:
            patt = data_df[(ind - PATTERN_SIZE + OVERLAP) : ind + OVERLAP].to_numpy()
            sell_patterns.append(patt)
    assert (
        len(patterns) > 0
    ), "На данном участке extr_window не обнаруженно паттернов класса 'buy'."
    assert (
        len(sell_patterns) > 0
    ), "На данном участке extr_window не обнаруженно паттернов класса 'sell'."
    if len(patterns) < 2:
        patterns.append(patterns[0])
    if len(sell_patterns) < 2:
        sell_patterns.append(sell_patterns[0])

    print(f"Найдено паттернов класса buy = {len(patterns)}")
    print(f"Найдено паттернов класса sell = {len(sell_patterns)}")
    n_samples_to_train = ((len(patterns) - 1) * len(sell_patterns)) + (
        (len(sell_patterns) - 1) * len(patterns)
    )
    print(f"Количество уникальных триплетов = {n_samples_to_train}")

    scaler = StandardScaler()
    std_patterns = np.array([scaler.fit_transform(i) for i in patterns]).reshape(
        -1, PATTERN_SIZE, len(data_df.columns.to_list()), 1
    )
    std_sell_patterns = np.array(
        [scaler.fit_transform(i) for i in sell_patterns]
    ).reshape(-1, PATTERN_SIZE, len(data_df.columns.to_list()), 1)

    train_x = [std_patterns, std_sell_patterns]

    if len(patterns) == len(sell_patterns):
        train_x = np.array(train_x)

    else:
        train_x = np.array(train_x, dtype=object)

    return train_x, n_samples_to_train


def get_triplet_random(batch_size, nb_classes, data):
    """
    Create batch of APN triplets with a complete random strategy

    Arguments:
    batch_size -- integer
    Returns:
    triplets -- list containing 3 tensors A,P,N of shape (batch_size,w,h,c)
    """
    os.environ["PYTHONHASHSEED"] = str(2020)
    random.seed(2020)
    np.random.seed(2020)
    torch.manual_seed(2020)

    X = data

    w, h, c = X[0][0].shape

    # создаем массив будущих триплетов
    triplets = [np.zeros((batch_size, w, h, c)) for i in range(int(3))]
    labels = []

    for i in range(batch_size):
        # Pick one random class for anchor
        anchor_class = np.random.randint(0, nb_classes)
        labels.append(anchor_class)
        nb_sample_available_for_class_AP = X[anchor_class].shape[0]

        # Pick two different random pics for this class => A and P
        # [idx_A, idx_P] = np.random.choice(nb_sample_available_for_class_AP, size=2, replace=False)
        idx_A = 0
        idx_P = np.random.choice(
            range(1, nb_sample_available_for_class_AP), size=1, replace=False
        )

        # Pick another class for N, different from anchor_class
        negative_class = (anchor_class + np.random.randint(1, nb_classes)) % nb_classes
        nb_sample_available_for_class_N = X[negative_class].shape[0]

        # Pick a random pic for this negative class => N
        idx_N = np.random.randint(0, nb_sample_available_for_class_N)

        triplets[0][i, :, :, :] = X[anchor_class][idx_A, :, :, :]
        triplets[1][i, :, :, :] = X[anchor_class][idx_P, :, :, :]
        triplets[2][i, :, :, :] = X[negative_class][idx_N, :, :, :]

    return np.array(triplets)  # , np.array(labels)


# функция отображения графика лосс функции во время обучения
def show_plot(iteration, loss):
    plt.plot(iteration, loss)

    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.show()


# функция обучения модели
def train_triplet_net(lr, epochs, my_dataloader, net, distance_function, margin):
    optimizer = optim.Adam(net.parameters(), lr)
    triplet_loss = TripletMarginWithDistanceLoss(
        distance_function=distance_function, margin=margin
    )
    counter = []
    loss_history = []
    iteration_number = 0
    l = []
    # Iterate throught the epochs
    for epoch in range(epochs):

        # Iterate over batches
        for i, (anchor, positive, negative) in enumerate(my_dataloader, 0):

            # Send the images and labels to CUDA
            anchor, positive, negative = (
                anchor.cuda().permute(0, 3, 1, 2),
                positive.cuda().permute(0, 3, 1, 2),
                negative.cuda().permute(0, 3, 1, 2),
            )

            # Zero the gradients

            output1, output2, output3 = net(anchor, positive, negative)

            output = triplet_loss(output1, output2, output3)

            # Calculate the backpropagation
            output.backward()

            # Optimize
            optimizer.step()
            optimizer.zero_grad()
            l.append(output.item())

            # Every 10 batches print out the loss
            if i % 10 == 0:
                # print(f"\rEpoch number {epoch}\n Current loss {output}\n", end="")
                iteration_number += 10

                counter.append(iteration_number)
                out = output.cpu()
                loss_history.append(out.detach().numpy())
        last_epoch_loss = torch.tensor(l[-len(my_dataloader) : -1]).mean()
    show_plot(counter, loss_history)
    return l, last_epoch_loss


"""def l_infinity(x1, x2):
    return torch.max(torch.abs(x1 - x2), dim=1).values


def euclid_dist(x1, x2):
    return torch.cdist(x1, x2) ** 2


def manhatten_dist(x1, x2):
    dif_tnzr = x1 - x2
    return torch.sum(torch.abs(dif_tnzr))"""


def get_CLtrain_data(
    data_df, profit_value, EXTR_WINDOW, PATTERN_SIZE, OVERLAP, train_dates,
):
    close_price = data_df["Close"].values.tolist()
    min_indexes = argrelmin(np.array(close_price), order=EXTR_WINDOW)[0]
    min_indexes = [
        i
        for i in min_indexes
        if (
            i + EXTR_WINDOW <= len(close_price) - 1
            and i - PATTERN_SIZE >= 0
            and i + OVERLAP <= len(close_price) - 1
        )
    ]
    max_indexes = argrelmax(np.array(close_price), order=EXTR_WINDOW)[0]
    max_indexes = [
        i
        for i in max_indexes
        if (
            i + EXTR_WINDOW <= len(close_price) - 1
            and i - PATTERN_SIZE >= 0
            and i + OVERLAP <= len(close_price) - 1
        )
    ]
    if len(min_indexes) < 10:
        while len(min_indexes) < 15:
            print("min_extr", EXTR_WINDOW)
            min_indexes = argrelmin(np.array(close_price), order=EXTR_WINDOW)[0]
            min_indexes = [
                i
                for i in min_indexes
                if (
                    i + EXTR_WINDOW <= len(close_price) - 1
                    and i - PATTERN_SIZE >= 0
                    and i + OVERLAP <= len(close_price) - 1
                )
            ]
            EXTR_WINDOW -= 3
    if len(max_indexes) < 10:
        while len(max_indexes) < 10:
            print("man_extr", EXTR_WINDOW)
            max_indexes = argrelmax(np.array(close_price), order=EXTR_WINDOW)[0]
            max_indexes = [
                i
                for i in max_indexes
                if (
                    i + EXTR_WINDOW <= len(close_price) - 1
                    and i - PATTERN_SIZE >= 0
                    and i + OVERLAP <= len(close_price) - 1
                )
            ]
            EXTR_WINDOW -= 3
    index_with_prof = [
        i
        for i in min_indexes
        if close_price[i : i + EXTR_WINDOW][-1] - close_price[i : i + EXTR_WINDOW][0]
        >= close_price[i : i + EXTR_WINDOW][0] * profit_value
    ]
    index_lost_prof = [
        i
        for i in max_indexes
        if close_price[i : i + EXTR_WINDOW][-1] - close_price[i : i + EXTR_WINDOW][0]
        <= -(close_price[i : i + EXTR_WINDOW][0] * profit_value)
    ]
    if len(index_with_prof) < 10:
        while len(index_with_prof) < 10:
            profit_value -= 0.001

            index_with_prof = [
                i
                for i in min_indexes
                if close_price[i : i + EXTR_WINDOW][-1]
                - close_price[i : i + EXTR_WINDOW][0]
                >= close_price[i : i + EXTR_WINDOW][0] * profit_value
            ]
    if len(index_lost_prof) < 10:
        while len(index_lost_prof) < 10:
            profit_value -= 0.001

            index_lost_prof = [
                i
                for i in max_indexes
                if close_price[i : i + EXTR_WINDOW][-1]
                - close_price[i : i + EXTR_WINDOW][0]
                <= -(close_price[i : i + EXTR_WINDOW][0] * profit_value)
            ]

    CL_patterns = []
    for ind in index_with_prof:
        if ind - PATTERN_SIZE >= 0:

            CL_patt = data_df[
                [
                    "DiffEMA",
                    "SmoothDiffEMA",
                    "VolatilityTunnel",
                    "BuyIntense",
                    "SellIntense",
                ]
            ][(ind - PATTERN_SIZE + OVERLAP) : ind + OVERLAP].to_numpy()

            CL_patterns.append(CL_patt)

    CL_sell_patterns = []
    for ind in index_lost_prof:
        if ind - PATTERN_SIZE >= 0:

            CL_sell_patt = data_df[
                [
                    "DiffEMA",
                    "SmoothDiffEMA",
                    "VolatilityTunnel",
                    "BuyIntense",
                    "SellIntense",
                ]
            ][(ind - PATTERN_SIZE + OVERLAP) : ind + OVERLAP].to_numpy()

            CL_sell_patterns.append(CL_sell_patt)

    if len(CL_patterns) < 2:
        CL_patterns.append(CL_patterns[0])
    if len(CL_sell_patterns) < 2:
        CL_sell_patterns.append(CL_sell_patterns[0])

    print(f"Найдено паттернов класса buy = {len(CL_patterns)}")
    print(f"Найдено паттернов класса sell = {len(CL_sell_patterns)}")
    n_samples_to_train = len(CL_patterns) * len(CL_sell_patterns)

    # n_samples_to_train = n_samples_to_train // 2

    print(f"Количество уникальных триплетов = {n_samples_to_train}")

    CL_patterns = np.array(CL_patterns).reshape(
        -1,
        PATTERN_SIZE,
        len(
            data_df[
                [
                    "DiffEMA",
                    "SmoothDiffEMA",
                    "VolatilityTunnel",
                    "BuyIntense",
                    "SellIntense",
                ]
            ].columns.to_list()
        ),
        1,
    )
    CL_sell_patterns = np.array(CL_sell_patterns).reshape(
        -1,
        PATTERN_SIZE,
        len(
            data_df[
                [
                    "DiffEMA",
                    "SmoothDiffEMA",
                    "VolatilityTunnel",
                    "BuyIntense",
                    "SellIntense",
                ]
            ].columns.to_list()
        ),
        1,
    )

    # train_x = [std_patterns, std_sell_patterns]
    train_x = [CL_patterns, CL_sell_patterns]
    if len(CL_patterns) == len(CL_sell_patterns):
        train_x = np.array(train_x)

    else:
        train_x = np.array(train_x, dtype=object)

    return train_x, n_samples_to_train


'''def find_best_dist(result_df, step):
    plt_backtesting._MAX_CANDLES = 100_000
    pd.pandas.set_option("display.max_columns", None)
    pd.set_option("expand_frame_repr", False)
    pd.options.display.expand_frame_repr = False
    pd.set_option("precision", 2)

    result_df.set_index("Datetime", inplace=True)
    result_df.index = pd.to_datetime(result_df.index)
    result_df = result_df.sort_index()
    result_df["Signal"] = 0
    print("******* Результы предсказания сети *******")
    print(result_df)
    print()

    """ Параметры тестирования """
    i = 0
    deposit = 400000  # сумма одного контракта GC & CL
    comm = 4.6  # GC - комиссия для золота
    # comm = 4.52  # CL - комиссия для нейти
    # sell_after = 1.6
    # buy_before = 0.6
    # step = 0.1  # с каким шагом проводим тест разметки
    # result_filename = f'{DESTINATION_ROOT}/selection_distances_{FILENAME[:-4]}_step{step}'

    """ Тестирвоание """

    df_stats = pd.DataFrame()

    treshhold = 1 / step
    if treshhold >= (round(result_df.Distance.max()) / step):
        treshhold = (round(result_df.Distance.max()) / step) - 1

    for sell_after in trange(
        int(treshhold), int(round(result_df.Distance.max(), 1) / step)
    ):
        for buy_before in range(
            int(round(result_df.Distance.min(), 1) / step), int(treshhold)
        ):
            # print(f'Диапазон Distance from {sell_trash/10} to {buy_trash/10}')
            result_df["Signal"].where(
                ~(result_df.Distance >= sell_after * step), -1, inplace=True
            )
            result_df["Signal"].where(
                ~(result_df.Distance <= buy_before * step), 1, inplace=True
            )
            # df['Signal'] = np.roll(df.Signal, 2)

            # сделаем так, чтобы 0 расценивался как "держать прежнюю позицию"
            result_df.loc[
                result_df["Signal"] == 0, "Signal"
            ] = np.nan  # заменим 0 на nan
            result_df["Signal"] = result_df[
                "Signal"
            ].ffill()  # заменим nan на предыдущие значения
            result_df.dropna(axis=0, inplace=True)  # Удаляем наниты
            result_df = result_df.loc[
                result_df["Signal"] != 0
            ]  # оставим только не нулевые строки
            """df.to_csv(
                    f'{DESTINATION_ROOT}/{out_root}/signals_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}_step{step}_{buy_before * step}_{sell_after * step}.csv')"""

            bt = Backtest(
                result_df, LnSF, cash=deposit, commission=0.00, trade_on_close=True
            )
            stats = bt.run(deal_amount="fix", fix_sum=200000)[:27]
            """if stats['Return (Ann.) [%]'] > 0:  # будем показывать и сохранять только доходные разметки
                    bt.plot(plot_volume=True, relative_equity=True,
                            filename=f'{DESTINATION_ROOT}/{out_root}/bt_plot_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}_step{step}_{buy_before * step}_{sell_after * step}.html'
                            )
                stats.to_csv(
                    f'{DESTINATION_ROOT}/{out_root}/stats_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}_step{step}_{buy_before * step}_{sell_after * step}.txt')"""

            df_stats = df_stats.append(stats, ignore_index=True)
            df_stats.loc[i, "Net Profit [$]"] = (
                df_stats.loc[i, "Equity Final [$]"]
                - deposit
                - df_stats.loc[i, "# Trades"] * comm
            )
            df_stats.loc[i, "buy_before"] = buy_before * step
            df_stats.loc[i, "sell_after"] = sell_after * step
            """df_stats.loc[i, 'train_window'] = int(df['Train_shape'].iloc[0])
                df_stats.loc[i, 'pattern_size'] = PATTERN_SIZE
                df_stats.loc[i, 'extr_window'] = EXTR_WINDOW
                df_stats.loc[i, 'profit_value'] = profit_value
                df_stats.loc[i, 'overlap'] = OVERLAP"""
            i += 1

    df_stats = df_stats.sort_values(by="Net Profit [$]", ascending=False)
    df_stats = df_stats.loc[df_stats["# Trades"] > 1].reset_index(drop=True)
    buy_before = df_stats.loc[df_stats.index[0], "buy_before"]
    sell_after = df_stats.loc[df_stats.index[0], "sell_after"]
    # df_plot = df_stats[["Net Profit [$]", "buy_before", "sell_after"]]
    """fig = px.parallel_coordinates(
            df_plot,
            color="Net Profit [$]",
            labels={
                "Net Profit [$]": "Net Profit ($)",
                "buy_before": "buy_before dist",
                "sell_after": "sell_after dist",
            },
            range_color=[
                df_plot["Net Profit [$]"].min(),
                df_plot["Net Profit [$]"].max(),
            ],
            color_continuous_scale=px.colors.sequential.Viridis,
            title="Зависимость профита от дистанций",
        )

    # fig.write_html("1min_gold_dep_analisys.html")  # сохраняем в файл
    fig.show()"""

    return buy_before, sell_after'''


def get_signals(result_df, buy_before, sell_after):
    """plt_backtesting._MAX_CANDLES = 100_000
    pd.pandas.set_option('display.max_columns', None)
    pd.set_option("expand_frame_repr", False)
    pd.options.display.expand_frame_repr = False
    pd.set_option("precision", 2)"""

    """result_df.set_index('Datetime', inplace=True)
    result_df.index = pd.to_datetime(result_df.index)
    result_df = result_df.sort_index()"""
    result_df["Signal"] = 0
    """print("******* Результы предсказания сети *******")
    print(result_df)
    print()'''

    """ """ Параметры тестирования """
    i = 0
    deposit = 400000  # сумма одного контракта GC & CL
    comm = 4.6  # GC - комиссия для золота
    # comm = 4.52  # CL - комиссия для нейти
    # sell_after = 1.6
    # buy_before = 0.6
    step = 0.1  # с каким шагом проводим тест разметки
    # result_filename = f'{DESTINATION_ROOT}/selection_distances_{FILENAME[:-4]}_step{step}"""

    """ Тестирвоание """

    """out_root = f"{FILENAME[:-4]}_forward_run_begin_{result_df.index[0]}_end_{result_df.index[-1]}"
    os.mkdir(f'{DESTINATION_ROOT}/{out_root}')"""
    result_df["Signal"].where(~(result_df.Distance >= sell_after), -1, inplace=True)
    result_df["Signal"].where(~(result_df.Distance <= buy_before), 1, inplace=True)

    """result_df.loc[result_df['Signal'] == 0, 'Signal'] = np.nan  # заменим 0 на nan
    result_df['Signal'] = result_df['Signal'].ffill()  # заменим nan на предыдущие значения
    result_df.dropna(axis=0, inplace=True)  # Удаляем наниты
    result_df = result_df.loc[result_df['Signal'] != 0]  # оставим только не нулевые строки"""
    """result_df.to_csv(
            f'{DESTINATION_ROOT}/{out_root}/forward_signals_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}_step{step}_{buy_before * step}_{sell_after * step}.csv')

    bt = Backtest(result_df, LnSF, cash=deposit, commission=0.00, trade_on_close=True)
    stats = bt.run(deal_amount='fix', fix_sum=200000)[:27]

    bt.plot(plot_volume=True, relative_equity=True,
                    filename=f'{DESTINATION_ROOT}/{out_root}/forward_bt_plot_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}_step{step}_{buy_before * step}_{sell_after * step}.html'
                    )
    stats.to_csv(
            f'{DESTINATION_ROOT}/{out_root}/forward_stats_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}_step{step}_{buy_before * step}_{sell_after * step}.txt')

    df_stats = df_stats.append(stats, ignore_index=True)
    df_stats.loc[i, 'Net Profit [$]'] = df_stats.loc[i, 'Equity Final [$]'] - deposit - df_stats.loc[
            i, '# Trades'] * comm
    df_stats.loc[i, 'buy_before'] = buy_before * step
    df_stats.loc[i, 'sell_after'] = sell_after * step
    df_stats.loc[i, 'train_window'] = int(df['Train_shape'].iloc[0])
    df_stats.loc[i, 'pattern_size'] = PATTERN_SIZE
    df_stats.loc[i, 'extr_window'] = EXTR_WINDOW
    df_stats.loc[i, 'profit_value'] = profit_value
    df_stats.loc[i, 'overlap'] = OVERLAP"""

    return result_df


'''def forward_trade(
    result_df,
    DESTINATION_ROOT,
    FILENAME,
    PATTERN_SIZE,
    EXTR_WINDOW,
    OVERLAP,
    train_window,
    select_dist_window,
    forward_window,
):
    plt_backtesting._MAX_CANDLES = 100_000
    pd.pandas.set_option("display.max_columns", None)
    pd.set_option("expand_frame_repr", False)
    pd.options.display.expand_frame_repr = False
    pd.set_option("precision", 2)

    result_df.set_index("Datetime", inplace=True)
    result_df.index = pd.to_datetime(result_df.index)
    result_df = result_df.sort_index()

    print("******* Результы предсказания сети *******")
    print(result_df)
    print()

    """ Параметры тестирования """
    i = 0
    deposit = 400000  # сумма одного контракта GC & CL
    comm = 4.6  # GC - комиссия для золота
    # comm = 4.52  # CL - комиссия для нейти
    # sell_after = 1.6
    # buy_before = 0.6
    # step = 0.1  # с каким шагом проводим тест разметки
    # result_filename = f'{DESTINATION_ROOT}/selection_distances_{FILENAME[:-4]}_step{step}

   

    out_root = f"{FILENAME[:-4]}_forward_run_begin_{result_df.index[0]}_end_{result_df.index[-1]}_({train_window / 1000}k_{select_dist_window / 1000}k_{forward_window / 1000}k_)"
    os.mkdir(f"{DESTINATION_ROOT}/{out_root}")

    # result_df['Signal'].where(~(result_df.Distance >= sell_after), -1, inplace=True)
    # result_df['Signal'].where(~(result_df.Distance <= buy_before), 1, inplace=True)

    result_df.loc[result_df["Signal"] == 0, "Signal"] = np.nan  # заменим 0 на nan
    result_df["Signal"] = result_df[
        "Signal"
    ].ffill()  # заменим nan на предыдущие значения
    result_df.dropna(axis=0, inplace=True)  # Удаляем наниты
    result_df = result_df.loc[
        result_df["Signal"] != 0
    ]  # оставим только не нулевые строки
    result_df.to_csv(
        f"{DESTINATION_ROOT}/{out_root}/forward_signals_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}.csv"
    )

    bt = Backtest(result_df, LnSF, cash=deposit, commission=0.00, trade_on_close=True)
    stats = bt.run(deal_amount="fix", fix_sum=200000)[:27]

    bt.plot(
        plot_volume=True,
        relative_equity=True,
        filename=f"{DESTINATION_ROOT}/{out_root}/forward_bt_plot_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}.html",
    )
    stats.to_csv(
        f"{DESTINATION_ROOT}/{out_root}/forward_stats_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}.txt"
    )

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
    print("ФОРВАРДНЫЙ АНАЛИЗ ОКОНЧЕН")'''


def get_stat_after_forward(
    result_df,
    PATTERN_SIZE,
    EXTR_WINDOW,
    OVERLAP,
    train_window,
    select_dist_window,
    forward_window,
    profit_value,
    source_file_name,
    out_root,
    out_data_root,
    runs,
    get_trade_info=False,
):
    plt_backtesting._MAX_CANDLES = 100_000
    pd.pandas.set_option("display.max_columns", None)
    pd.set_option("expand_frame_repr", False)
    pd.options.display.expand_frame_repr = False
    pd.set_option("precision", 2)

    result_df.set_index("Datetime", inplace=True)
    result_df.index = pd.to_datetime(result_df.index)
    result_df = result_df.sort_index()

    """print("******* Результы предсказания сети *******")
    print(result_df)
    print()"""

    """ Параметры тестирования """
    i = 0
    deposit = 100000  # сумма одного контракта GC & CL
    comm = 4.6  # GC - комиссия для золота
    # comm = 4.52  # CL - комиссия для нейти
    # sell_after = 1.6
    # buy_before = 0.6
    # step = 0.1  # с каким шагом проводим тест разметки
    # result_filename = f'{DESTINATION_ROOT}/selection_distances_{FILENAME[:-4]}_step{step}'''

    """ Тестирвоание """

    """out_root = f"{FILENAME[:-4]}_forward_run_begin_{result_df.index[0]}_end_{result_df.index[-1]}_({train_window / 1000}k_{select_dist_window / 1000}k_{forward_window / 1000}k_)"
    os.mkdir(f"{DESTINATION_ROOT}/{out_root}")"""

    # result_df['Signal'].where(~(result_df.Distance >= sell_after), -1, inplace=True)
    # result_df['Signal'].where(~(result_df.Distance <= buy_before), 1, inplace=True)
    df_stats = pd.DataFrame()
    result_df.loc[result_df["Signal"] == 0, "Signal"] = np.nan  # заменим 0 на nan
    result_df["Signal"] = result_df[
        "Signal"
    ].ffill()  # заменим nan на предыдущие значения
    result_df.dropna(axis=0, inplace=True)  # Удаляем наниты
    result_df = result_df.loc[
        result_df["Signal"] != 0
    ]  # оставим только не нулевые строки
    """result_df.to_csv(
        f"{DESTINATION_ROOT}/{out_root}/forward_signals_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}.csv"
    )"""

    bt = Backtest(
        result_df,
        strategy=LazyStrategy,
        cash=deposit,
        commission_type="absolute",
        commission=4.62,
        features_coeff=10,
        exclusive_orders=True,
    )
    stats = bt.run()[:27]

    """bt.plot(
        plot_volume=True,
        relative_equity=True,
        filename=f"{DESTINATION_ROOT}/{out_root}/forward_bt_plot_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}.html",
    )"""
    """stats.to_csv(
        f"{DESTINATION_ROOT}/{out_root}/forward_stats_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}.txt"
    )"""

    df_stats = df_stats.append(stats, ignore_index=True)
    df_stats["Net Profit [$]"] = (
        df_stats.loc[i, "Equity Final [$]"]
        - deposit
        - df_stats.loc[i, "# Trades"] * comm
    )
    # df_stats.loc[i, "buy_before"] = buy_before * step
    # df_stats.loc[i, "sell_after"] = sell_after * step
    df_stats["train_window"] = train_window
    df_stats["select_dist_window"] = select_dist_window
    df_stats["forward_window"] = forward_window
    df_stats["pattern_size"] = PATTERN_SIZE
    df_stats["extr_window"] = EXTR_WINDOW
    df_stats["profit_value"] = profit_value
    df_stats["overlap"] = OVERLAP
    if get_trade_info == True and df_stats["Net Profit [$]"].values > 0:
        bt.plot(
            plot_volume=True,
            relative_equity=False,
            filename=f"{out_root}/{out_data_root}/{runs}_bt_plot_{source_file_name[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}.html",
        )
        stats.to_csv(
            f"{out_root}/{out_data_root}/{runs}_stats_{source_file_name[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}.txt"
        )
        result_df.to_csv(
            f"{out_root}/{out_data_root}/{runs}_signals_{source_file_name[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}.csv"
        )

    return df_stats


def find_best_dist_stbl(result_df, step):
    plt_backtesting._MAX_CANDLES = 100_000
    pd.pandas.set_option("display.max_columns", None)
    pd.set_option("expand_frame_repr", False)
    pd.options.display.expand_frame_repr = False
    pd.set_option("precision", 2)

    result_df.set_index("Datetime", inplace=True)
    result_df.index = pd.to_datetime(result_df.index)
    result_df = result_df.sort_index()
    result_df["Signal"] = 0
    """print("******* Результы предсказания сети *******")
    print(result_df)
    print()"""

    """ Параметры тестирования """
    i = 0
    deposit = 100000  # сумма одного контракта GC & CL
    comm = 4.6  # GC - комиссия для золота
    # comm = 4.52  # CL - комиссия для нейти
    # sell_after = 1.6
    # buy_before = 0.6
    # step = 0.1  # с каким шагом проводим тест разметки
    # result_filename = f'{DESTINATION_ROOT}/selection_distances_{FILENAME[:-4]}_step{step}'

    """ Тестирвоание """

    df_stats = pd.DataFrame()
    min_dist = result_df.Distance.min()
    max_dist = result_df.Distance.max()

    while step > max_dist:
        step = step / 10

    forward = [
        i
        for i in range(int(step / step), int((max_dist / step) + 1))
        if i >= int(min_dist / step)
    ]
    reversed = forward[::-1]

    for buy_before in forward:
        for sell_after in reversed:
            # print(f'Диапазон Distance from {sell_trash/10} to {buy_trash/10}')
            result_df["Signal"].where(
                ~(result_df.Distance >= sell_after * step), -1, inplace=True
            )
            result_df["Signal"].where(
                ~(result_df.Distance <= buy_before * step), 1, inplace=True
            )
            # df['Signal'] = np.roll(df.Signal, 2)

            # сделаем так, чтобы 0 расценивался как "держать прежнюю позицию"
            result_df.loc[
                result_df["Signal"] == 0, "Signal"
            ] = np.nan  # заменим 0 на nan
            result_df["Signal"] = result_df[
                "Signal"
            ].ffill()  # заменим nan на предыдущие значения
            result_df.dropna(axis=0, inplace=True)  # Удаляем наниты
            result_df = result_df.loc[
                result_df["Signal"] != 0
            ]  # оставим только не нулевые строки
            """df.to_csv(
                    f'{DESTINATION_ROOT}/{out_root}/signals_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}_step{step}_{buy_before * step}_{sell_after * step}.csv')"""

            bt = Backtest(
                result_df,
                strategy=LazyStrategy,
                cash=deposit,
                commission_type="absolute",
                commission=4.62,
                features_coeff=10,
                exclusive_orders=True,
            )
            stats = bt.run()[:27]
            """if stats['Return (Ann.) [%]'] > 0:  # будем показывать и сохранять только доходные разметки
                    bt.plot(plot_volume=True, relative_equity=True,
                            filename=f'{DESTINATION_ROOT}/{out_root}/bt_plot_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}_step{step}_{buy_before * step}_{sell_after * step}.html'
                            )
                stats.to_csv(
                    f'{DESTINATION_ROOT}/{out_root}/stats_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}_step{step}_{buy_before * step}_{sell_after * step}.txt')"""

            df_stats = df_stats.append(stats, ignore_index=True)
            df_stats.loc[i, "Net Profit [$]"] = (
                df_stats.loc[i, "Equity Final [$]"]
                - deposit
                - df_stats.loc[i, "# Trades"] * comm
            )
            df_stats.loc[i, "buy_before"] = buy_before * step
            df_stats.loc[i, "sell_after"] = sell_after * step
            """df_stats.loc[i, 'train_window'] = int(df['Train_shape'].iloc[0])
                df_stats.loc[i, 'pattern_size'] = PATTERN_SIZE
                df_stats.loc[i, 'extr_window'] = EXTR_WINDOW
                df_stats.loc[i, 'profit_value'] = profit_value
                df_stats.loc[i, 'overlap'] = OVERLAP"""
            i += 1
        reversed = reversed[:-1]

    df_stats = df_stats.sort_values(by="Net Profit [$]", ascending=False)
    df_stats = df_stats.loc[df_stats["# Trades"] > 1].reset_index(drop=True)
    buy_before = df_stats.loc[df_stats.index[0], "buy_before"]
    sell_after = df_stats.loc[df_stats.index[0], "sell_after"]
    # df_plot = df_stats[["Net Profit [$]", "buy_before", "sell_after"]]
    """fig = px.parallel_coordinates(
            df_plot,
            color="Net Profit [$]",
            labels={
                "Net Profit [$]": "Net Profit ($)",
                "buy_before": "buy_before dist",
                "sell_after": "sell_after dist",
            },
            range_color=[
                df_plot["Net Profit [$]"].min(),
                df_plot["Net Profit [$]"].max(),
            ],
            color_continuous_scale=px.colors.sequential.Viridis,
            title="Зависимость профита от дистанций",
        )

        # fig.write_html("1min_gold_dep_analisys.html")  # сохраняем в файл
        fig.show()"""

    return buy_before, sell_after


'''def fliped_get_signals(result_df, sell_before, buy_after):

    result_df["Signal"] = 0
    """print("******* Результы предсказания сети *******")
    print(result_df)
    print()"""

    result_df["Signal"].where(~(result_df.Distance >= buy_after), 1, inplace=True)
    result_df["Signal"].where(~(result_df.Distance <= sell_before), -1, inplace=True)

    return result_df


def fliped_find_best_dist_stbl(result_df, step):
    plt_backtesting._MAX_CANDLES = 100_000
    pd.pandas.set_option("display.max_columns", None)
    pd.set_option("expand_frame_repr", False)
    pd.options.display.expand_frame_repr = False
    pd.set_option("precision", 2)

    result_df.set_index("Datetime", inplace=True)
    result_df.index = pd.to_datetime(result_df.index)
    result_df = result_df.sort_index()
    result_df["Signal"] = 0
    print("******* Результы предсказания сети *******")
    print(result_df)
    print()

    """ Параметры тестирования """
    i = 0
    deposit = 100000  # сумма одного контракта GC & CL
    comm = 4.6  # GC - комиссия для золота
    # comm = 4.52  # CL - комиссия для нейти
    # sell_after = 1.6
    # buy_before = 0.6
    # step = 0.1  # с каким шагом проводим тест разметки
    # result_filename = f'{DESTINATION_ROOT}/selection_distances_{FILENAME[:-4]}_step{step}'

    """ Тестирвоание """

    df_stats = pd.DataFrame()
    min_dist = result_df.Distance.min()
    max_dist = result_df.Distance.max()

    while step > max_dist:
        step = step / 10

    forward = [
        i
        for i in range(int(step / step), int((max_dist / step) + 1))
        if i >= int(min_dist / step)
    ]
    reversed = forward[::-1]

    for sell_before in forward:
        for buy_after in reversed:
            # print(f'Диапазон Distance from {sell_trash/10} to {buy_trash/10}')
            result_df["Signal"].where(
                ~(result_df.Distance <= sell_before * step), -1, inplace=True
            )
            result_df["Signal"].where(
                ~(result_df.Distance >= buy_after * step), 1, inplace=True
            )
            # df['Signal'] = np.roll(df.Signal, 2)

            # сделаем так, чтобы 0 расценивался как "держать прежнюю позицию"
            result_df.loc[
                result_df["Signal"] == 0, "Signal"
            ] = np.nan  # заменим 0 на nan
            result_df["Signal"] = result_df[
                "Signal"
            ].ffill()  # заменим nan на предыдущие значения
            result_df.dropna(axis=0, inplace=True)  # Удаляем наниты
            result_df = result_df.loc[
                result_df["Signal"] != 0
            ]  # оставим только не нулевые строки
            """df.to_csv(
                    f'{DESTINATION_ROOT}/{out_root}/signals_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}_step{step}_{buy_before * step}_{sell_after * step}.csv')"""

            bt = Backtest(
                result_df,
                strategy=LazyStrategy,
                cash=deposit,
                commission_type="absolute",
                commission=4.62,
                features_coeff=10,
                exclusive_orders=True,
            )
            stats = bt.run()[:27]
            """if stats['Return (Ann.) [%]'] > 0:  # будем показывать и сохранять только доходные разметки
                    bt.plot(plot_volume=True, relative_equity=True,
                            filename=f'{DESTINATION_ROOT}/{out_root}/bt_plot_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}_step{step}_{buy_before * step}_{sell_after * step}.html'
                            )
                stats.to_csv(
                    f'{DESTINATION_ROOT}/{out_root}/stats_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}_step{step}_{buy_before * step}_{sell_after * step}.txt')"""

            df_stats = df_stats.append(stats, ignore_index=True)
            df_stats.loc[i, "Net Profit [$]"] = (
                df_stats.loc[i, "Equity Final [$]"]
                - deposit
                - df_stats.loc[i, "# Trades"] * comm
            )
            df_stats.loc[i, "sell_before"] = sell_before * step
            df_stats.loc[i, "buy_after"] = buy_after * step
            """df_stats.loc[i, 'train_window'] = int(df['Train_shape'].iloc[0])
                df_stats.loc[i, 'pattern_size'] = PATTERN_SIZE
                df_stats.loc[i, 'extr_window'] = EXTR_WINDOW
                df_stats.loc[i, 'profit_value'] = profit_value
                df_stats.loc[i, 'overlap'] = OVERLAP"""
            i += 1
        reversed = reversed[:-1]

    df_stats = df_stats.sort_values(by="Net Profit [$]", ascending=False)
    df_stats = df_stats.loc[df_stats["# Trades"] > 1].reset_index(drop=True)
    sell_before = df_stats.loc[df_stats.index[0], "sell_before"]
    buy_after = df_stats.loc[df_stats.index[0], "buy_after"]

    return sell_before, buy_after'''


def uptune_get_stat_after_forward(
    result_df,
    PATTERN_SIZE,
    EXTR_WINDOW,
    OVERLAP,
    train_window,
    select_dist_window,
    forward_window,
    profit_value,
    source_file_name,
    out_root,
    out_data_root,
    trial_namber,
    get_trade_info=False,
):
    plt_backtesting._MAX_CANDLES = 100_000
    pd.pandas.set_option("display.max_columns", None)
    pd.set_option("expand_frame_repr", False)
    pd.options.display.expand_frame_repr = False
    pd.set_option("precision", 2)

    result_df.set_index("Datetime", inplace=True)
    result_df.index = pd.to_datetime(result_df.index)
    result_df = result_df.sort_index()

    """print("******* Результы предсказания сети *******")
    print(result_df)
    print()"""

    """ Параметры тестирования """
    i = 0
    deposit = 100000  # сумма одного контракта GC & CL
    comm = 4.6  # GC - комиссия для золота
    # comm = 4.52  # CL - комиссия для нейти
    # sell_after = 1.6
    # buy_before = 0.6
    # step = 0.1  # с каким шагом проводим тест разметки
    # result_filename = f'{DESTINATION_ROOT}/selection_distances_{FILENAME[:-4]}_step{step}'''

    """ Тестирвоание """

    """out_root = f"{FILENAME[:-4]}_forward_run_begin_{result_df.index[0]}_end_{result_df.index[-1]}_({train_window / 1000}k_{select_dist_window / 1000}k_{forward_window / 1000}k_)"
    os.mkdir(f"{DESTINATION_ROOT}/{out_root}")"""

    # result_df['Signal'].where(~(result_df.Distance >= sell_after), -1, inplace=True)
    # result_df['Signal'].where(~(result_df.Distance <= buy_before), 1, inplace=True)
    df_stats = pd.DataFrame()
    result_df.loc[result_df["Signal"] == 0, "Signal"] = np.nan  # заменим 0 на nan
    result_df["Signal"] = result_df[
        "Signal"
    ].ffill()  # заменим nan на предыдущие значения
    result_df.dropna(axis=0, inplace=True)  # Удаляем наниты
    result_df = result_df.loc[
        result_df["Signal"] != 0
    ]  # оставим только не нулевые строки
    """result_df.to_csv(
        f"{DESTINATION_ROOT}/{out_root}/forward_signals_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}.csv"
    )"""

    bt = Backtest(
        result_df,
        strategy=LazyStrategy,
        cash=deposit,
        commission_type="absolute",
        commission=4.62,
        features_coeff=10,
        exclusive_orders=True,
    )
    stats = bt.run()[:27]

    """bt.plot(
        plot_volume=True,
        relative_equity=True,
        filename=f"{DESTINATION_ROOT}/{out_root}/forward_bt_plot_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}.html",
    )"""
    """stats.to_csv(
        f"{DESTINATION_ROOT}/{out_root}/forward_stats_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}.txt"
    )"""

    df_stats = df_stats.append(stats, ignore_index=True)
    df_stats["Net Profit [$]"] = (
        df_stats.loc[i, "Equity Final [$]"]
        - deposit
        - df_stats.loc[i, "# Trades"] * comm
    )
    # df_stats.loc[i, "buy_before"] = buy_before * step
    # df_stats.loc[i, "sell_after"] = sell_after * step
    df_stats["train_window"] = train_window
    df_stats["select_dist_window"] = select_dist_window
    df_stats["forward_window"] = forward_window
    df_stats["pattern_size"] = PATTERN_SIZE
    df_stats["extr_window"] = EXTR_WINDOW
    df_stats["profit_value"] = profit_value
    df_stats["overlap"] = OVERLAP
    if get_trade_info == True and df_stats["Net Profit [$]"].values > 0:
        bt.plot(
            plot_volume=True,
            relative_equity=False,
            filename=f"{out_root}/{out_data_root}/{trial_namber}_bt_plot_{source_file_name[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}.html",
        )
        stats.to_csv(
            f"{out_root}/{out_data_root}/{trial_namber}_stats_{source_file_name[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}.txt"
        )
        result_df["Signal"] = result_df["Signal"].astype(int)
        # result_df["Datetime"] = result_df.index
        result_df.insert(0, "Datetime", result_df.index)
        result_df = result_df.reset_index(drop=True)
        result_df[
            ["Datetime", "Open", "High", "Low", "Close", "Volume", "Signal"]
        ].to_csv(
            f"{out_root}/{out_data_root}/{trial_namber}_signals_{source_file_name[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}.csv"
        )

    return df_stats


def tensor_size_calc(pattern_size, features_num, kernel, strd, conv_chs):
    layer = nn.Conv2d(1, conv_chs, kernel_size=kernel, stride=strd)
    input = torch.randn(100, 1, pattern_size, features_num)
    output = layer(input).detach().numpy()

    after_conv = np.prod(output.shape[1:])
    return after_conv


def bayes_train_triplet_net(lr, epochs, my_dataloader, net, distance_function, margin):
    optimizer = optim.Adam(net.parameters(), lr)
    triplet_loss = TripletMarginWithDistanceLoss(
        distance_function=distance_function, margin=margin
    )
    klloss = bnn.BKLLoss(reduction="mean", last_layer_only=False)
    counter = []
    loss_history = []
    iteration_number = 0
    l = []
    # Iterate throught the epochs
    for epoch in range(epochs):

        # Iterate over batches
        for i, (anchor, positive, negative) in enumerate(my_dataloader, 0):

            # Send the images and labels to CUDA
            anchor, positive, negative = (
                anchor.cuda().permute(0, 3, 1, 2),
                positive.cuda().permute(0, 3, 1, 2),
                negative.cuda().permute(0, 3, 1, 2),
            )

            # Zero the gradients

            output1, output2, output3 = net(anchor, positive, negative)

            output = triplet_loss(output1, output2, output3)
            kl = klloss(net)
            total_cost = output + 0.01 * kl
            # Calculate the backpropagation
            total_cost.backward()

            # Optimize
            optimizer.step()
            optimizer.zero_grad()
            l.append(output.item())

            # Every 10 batches print out the loss
            if i % 10 == 0:
                # print(f"\rEpoch number {epoch}\n Current loss {output}\n", end="")
                iteration_number += 10

                counter.append(iteration_number)
                out = output.cpu()
                loss_history.append(out.detach().numpy())
        last_epoch_loss = torch.tensor(l[-len(my_dataloader) : -1]).mean()
    show_plot(counter, loss_history)
    return l, last_epoch_loss


def bayes_tune_get_stat_after_forward(
    result_df,
    lookback_size,
    epochs,
    n_hiden,
    n_hiden_two,
    train_window,
    forward_window,
    source_file_name,
    out_root,
    out_data_root,
    trial_namber,
    get_trade_info=False,
):
    plt_backtesting._MAX_CANDLES = 100_000
    pd.pandas.set_option("display.max_columns", None)
    pd.set_option("expand_frame_repr", False)
    pd.options.display.expand_frame_repr = False
    pd.set_option("precision", 2)

    result_df.set_index("Datetime", inplace=True)
    result_df.index = pd.to_datetime(result_df.index)
    result_df = result_df.sort_index()

    """print("******* Результы предсказания сети *******")
    print(result_df)
    print()"""

    """ Параметры тестирования """
    i = 0
    deposit = 100000  # сумма одного контракта GC & CL
    comm = 4.6  # GC - комиссия для золота
    # comm = 4.52  # CL - комиссия для нейти
    # sell_after = 1.6
    # buy_before = 0.6
    # step = 0.1  # с каким шагом проводим тест разметки
    # result_filename = f'{DESTINATION_ROOT}/selection_distances_{FILENAME[:-4]}_step{step}'''

    """ Тестирвоание """

    """out_root = f"{FILENAME[:-4]}_forward_run_begin_{result_df.index[0]}_end_{result_df.index[-1]}_({train_window / 1000}k_{select_dist_window / 1000}k_{forward_window / 1000}k_)"
    os.mkdir(f"{DESTINATION_ROOT}/{out_root}")"""

    # result_df['Signal'].where(~(result_df.Distance >= sell_after), -1, inplace=True)
    # result_df['Signal'].where(~(result_df.Distance <= buy_before), 1, inplace=True)
    df_stats = pd.DataFrame()
    result_df.loc[result_df["Signal"] == 0, "Signal"] = np.nan  # заменим 0 на nan
    result_df["Signal"] = result_df[
        "Signal"
    ].ffill()  # заменим nan на предыдущие значения
    result_df.dropna(axis=0, inplace=True)  # Удаляем наниты
    result_df = result_df.loc[
        result_df["Signal"] != 0
    ]  # оставим только не нулевые строки
    """result_df.to_csv(
        f"{DESTINATION_ROOT}/{out_root}/forward_signals_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}.csv"
    )"""

    bt = Backtest(
        result_df,
        strategy=LazyStrategy,
        cash=deposit,
        commission_type="absolute",
        commission=4.62,
        features_coeff=10,
        exclusive_orders=True,
    )
    stats = bt.run()[:27]

    """bt.plot(
        plot_volume=True,
        relative_equity=True,
        filename=f"{DESTINATION_ROOT}/{out_root}/forward_bt_plot_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}.html",
    )"""
    """stats.to_csv(
        f"{DESTINATION_ROOT}/{out_root}/forward_stats_{FILENAME[:-4]}_patern{PATTERN_SIZE}_extrw{EXTR_WINDOW}_overlap{OVERLAP}.txt"
    )"""

    df_stats = df_stats.append(stats, ignore_index=True)
    df_stats["Net Profit [$]"] = (
        df_stats.loc[i, "Equity Final [$]"]
        - deposit
        - df_stats.loc[i, "# Trades"] * comm
    )
    # df_stats.loc[i, "buy_before"] = buy_before * step
    # df_stats.loc[i, "sell_after"] = sell_after * step
    df_stats["train_window"] = train_window
    df_stats["forward_window"] = forward_window
    df_stats["lookback_size"] = lookback_size
    df_stats["epochs"] = epochs
    df_stats["n_hiden"] = n_hiden
    df_stats["n_hiden_two"] = n_hiden_two
    if get_trade_info == True and df_stats["Net Profit [$]"].values > 0:
        bt.plot(
            plot_volume=True,
            relative_equity=False,
            filename=f"{out_root}/{out_data_root}/{trial_namber}_bt_plot_{source_file_name[:-4]}train_window{train_window}forward_window{forward_window}_lookback_size{lookback_size}.html",
        )
        stats.to_csv(
            f"{out_root}/{out_data_root}/{trial_namber}_stats_{source_file_name[:-4]}_train_window{train_window}forward_window{forward_window}_lookback_size{lookback_size}.txt"
        )
        result_df["Signal"] = result_df["Signal"].astype(int)
        # result_df["Datetime"] = result_df.index
        result_df.insert(0, "Datetime", result_df.index)
        result_df = result_df.reset_index(drop=True)
        result_df[
            ["Datetime", "Open", "High", "Low", "Close", "Volume", "Signal"]
        ].to_csv(
            f"{out_root}/{out_data_root}/{trial_namber}_signals_{source_file_name[:-4]}_train_window{train_window}forward_window{forward_window}_lookback_size{lookback_size}.csv",
            index=False,
        )

    return df_stats
