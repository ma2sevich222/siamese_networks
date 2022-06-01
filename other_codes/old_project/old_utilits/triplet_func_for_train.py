import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from scipy.signal import argrelmin, argrelmax
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
from torch import optim

# from torch.nn import PairwiseDistance
from torch.nn import TripletMarginWithDistanceLoss
from sklearn.preprocessing import StandardScaler


def get_triplet_random(batch_size, nb_classes, data):
    """
    Create batch of APN triplets with a complete random strategy

    Arguments:
    batch_size -- integer
    Returns:
    triplets -- list containing 3 tensors A,P,N of shape (batch_size,w,h,c)
    """

    X = data

    w, h, c = X[0][0].shape

    # initialize result
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

    return np.array(triplets), np.array(labels)


def show_plot(iteration, loss):
    plt.plot(iteration, loss)

    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.show()


def train_triplet_net(lr, epochs, my_dataloader, net, distance_function):
    optimizer = optim.Adam(net.parameters(), lr)
    triplet_loss = TripletMarginWithDistanceLoss(distance_function=distance_function)
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
                print(f"\rEpoch number {epoch}\n Current loss {output}\n", end="")
                iteration_number += 10

                counter.append(iteration_number)
                out = output.cpu()
                loss_history.append(out.detach().numpy())
        last_epoch_loss = torch.tensor(l[-len(my_dataloader) : -1]).mean()
    show_plot(counter, loss_history)
    return l, last_epoch_loss

    def train(num_epochs, model, criterion, optimizer, train_loader, batch_size):
        loss_history = []
        l = []
        model.train()
        for epoch in range(0, num_epochs):

            for i, batch in enumerate(train_loader, 0):
                anc, pos, neg = batch
                output_anc, output_pos, output_neg = model(
                    anc.to(device), pos.to(device), neg.to(device)
                )
                loss = criterion(output_anc, output_pos, output_neg)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                l.append(loss.item())
            last_epoch_loss = torch.tensor(l[-len(train_loader) : -1]).mean()
            print("Epoch {} with {:.4f} loss".format(epoch, last_epoch_loss))

        return l, last_epoch_loss


def get_patterns_index_classes(patterns, n_of_classes_return):
    patterns_reshape = patterns.reshape(
        -1, patterns.shape[1] * patterns.shape[2] * patterns.shape[3]
    )
    cos_distance_matrix = cosine_distances(patterns_reshape)
    to_find_best = np.array(
        [np.sum(i) / cos_distance_matrix.shape[0] for i in cos_distance_matrix]
    )
    sorted_best = list(np.argsort(to_find_best))
    return sorted_best[:n_of_classes_return]


def get_class_and_neighbours(patterns, pattern, n_patterns_in_class):
    pattern_array = patterns[pattern].reshape(
        -1, patterns[pattern].shape[0] * patterns[pattern].shape[1]
    )
    pattern_matrix = patterns.reshape(
        -1, patterns.shape[1] * patterns.shape[2] * patterns.shape[3]
    )
    cos_distance_array = cosine_distances(pattern_array, pattern_matrix)
    nearlest_patterns_names = np.argsort(cos_distance_array[0])
    nearlest_pattern_distances = np.sort(cos_distance_array[0])

    nearlist_neibors = pd.DataFrame(
        {
            "pattern": nearlest_patterns_names[:n_patterns_in_class],
            "distance": nearlest_pattern_distances[:n_patterns_in_class],
        }
    )

    return nearlist_neibors


def get_patterns_with_profit2(data_df, profit_value, EXTR_WINDOW, PATTERN_SIZE):
    close_price = data_df["Close"].values.tolist()
    patterns_indexes = []
    for i in range(len(close_price) - 1):
        if i - EXTR_WINDOW >= 0 and i + EXTR_WINDOW <= (len(close_price) - 1):
            if (
                close_price[i] <= min(close_price[i - EXTR_WINDOW : i + EXTR_WINDOW])
                and (close_price[i] + (close_price[i] * profit_value))
                - max(close_price[i : i + EXTR_WINDOW])
                <= 0
            ):
                patterns_indexes.append(i)

    patterns = []
    for ind in patterns_indexes:
        if ind - PATTERN_SIZE >= 0:
            patt = data_df[(ind - PATTERN_SIZE) : ind].to_numpy()
            patterns.append(patt)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data_df.index.tolist(), y=data_df["Close"], mode="lines", name="CLOSE"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=patterns_indexes,
            y=data_df["Close"].iloc[patterns_indexes],
            mode="markers",
            marker=dict(symbol="triangle-up", size=15),
        )
    )

    fig.update_layout(
        title="Разметка данных",
        xaxis_title="DATE",
        yaxis_title="CLOSE",
        legend_title="Legend",
    )
    fig.show()

    return np.array(patterns)


def get_patterns_with_profit(
    data_df,
    profit_value,
    EXTR_WINDOW,
    PATTERN_SIZE,
    OVERLAP,
    train_dates,
    save_to_dir=False,
):
    close_price = data_df["Close"].values.tolist()
    min_indexes = argrelmin(np.array(close_price), order=EXTR_WINDOW)[0]
    min_indexes = [i for i in min_indexes if i + EXTR_WINDOW <= len(data_df)]
    prices = []
    steps = [step for step in range(1, EXTR_WINDOW + 1)]
    for i in min_indexes:
        step_prices_for_ecach_idex = []
        for step in steps:
            growth_price = (close_price[(i + step)] - close_price[i]) / close_price[i]
            step_prices_for_ecach_idex.append(growth_price)
        prices.append(step_prices_for_ecach_idex)
    squares_df = pd.DataFrame()
    for colname, vlue_col in zip(min_indexes, prices):
        squares_df[f"{str(colname)}"] = vlue_col
    axey = squares_df.mean(axis=1).values.tolist()
    # axey = [[x if x >= profit_value else 0 for x in axey]]
    print(squares_df.shape)

    conf_intervals = []
    # lconf_intervals = []
    for r in range(len(squares_df)):
        values_arr = squares_df.iloc[r, :].to_numpy()
        # mean_ = np.mean(values_arr)
        std_ = np.std(values_arr) / 2
        # dof = len(values_arr) - 1
        # confidence = 0.95
        # t_crit = np.abs(t.ppf((1 - confidence) / 2, dof))
        # low = mean_ - std_ * t_crit / np.sqrt(len(values_arr))
        # high = mean_ + std_ * t_crit / np.sqrt(len(values_arr))
        # lconf_intervals.append(low)
        conf_intervals.append(std_)

    fig = go.Figure()
    # fig.add_trace(go.Bar(x=[f' Бар : {i} ' for i in steps], y=axey, error_y=dict(type='data',symmetric=False, array=hconf_intervals, arrayminus=lconf_intervals)))
    fig = px.bar(x=[f" Бар : {i} " for i in steps], y=axey, error_y=conf_intervals)
    fig.update_layout(
        title=dict(
            text=f" Прирост цены в  взависимости от удаления от паттерна с доверительными интервалами для 95% данных. Параметры : profit_value = {profit_value}, EXTR_WINDOW = {EXTR_WINDOW}, PATTERN_SIZE,OVERLAP = {OVERLAP} ",
            font=dict(size=15),
        ),
        xaxis_title="Бары",
        yaxis_title="Величина прироста в долях",
    )
    fig.show()

    indexes_with_profit = []
    for i in min_indexes:
        if i + (EXTR_WINDOW) <= len(close_price):

            if (
                close_price[i]
                + (close_price[i] * profit_value)
                - sum(close_price[int(i) : int(i + (EXTR_WINDOW))])
                / len(close_price[int(i) : int(i + (EXTR_WINDOW))])
                <= 0
            ):
                indexes_with_profit.append(i)

    patterns = []
    after_patt = []
    for ind in indexes_with_profit:
        if ind - PATTERN_SIZE >= 0:
            patt = data_df[(ind - PATTERN_SIZE + OVERLAP) : ind + OVERLAP].to_numpy()
            patterns.append(patt)  # df.loc[:, ['foo','bar','dat']]

            if ind + EXTR_WINDOW <= len(data_df):
                aft_pat = data_df["Close"][ind : ind + EXTR_WINDOW].values.tolist()
                after_patt.append(aft_pat)

            else:
                after_patt.append([0 for i in range(EXTR_WINDOW)])

    fig = go.Figure()  # x=data_df.index.tolist()
    fig.add_trace(
        go.Scatter(x=train_dates, y=data_df["Close"], mode="lines", name="CLOSE")
    )
    fig.add_trace(
        go.Scatter(
            x=train_dates.iloc[indexes_with_profit],
            y=data_df["Close"].iloc[indexes_with_profit],
            mode="markers",
            marker=dict(symbol="triangle-up", size=15),
        )
    )

    fig.update_layout(
        title=f"Разметка данных на основе параметров profit_value = {profit_value}, EXTR_WINDOW = {EXTR_WINDOW}, PATTERN_SIZE = {PATTERN_SIZE}  ",
        xaxis_title="DATE",
        yaxis_title="CLOSE",
        legend_title="Legend",
    )
    fig.show()
    if save_to_dir == True:
        parent_dir = "saved_patterns"
        dir = f"patt_size<{PATTERN_SIZE}>_Ex_window<{EXTR_WINDOW}>_prof_vle<{profit_value}>"
        path = os.path.join(parent_dir, dir)
        try:
            os.makedirs(path, exist_ok=False)
            print("Директория  '%s' создана" % dir)
            patterns_arr = np.array(patterns)
            patterns_reshaped = patterns_arr.reshape(patterns_arr.shape[0], -1)
            np.savetxt(
                f"{path}/buy_patterns_extr_window<{EXTR_WINDOW}>"
                f"_pattern_size{PATTERN_SIZE}.csv",
                patterns_reshaped,
            )
            print("Файл с паттернами сохранен")
        except OSError as error:
            print("Директория '%s' уже существует" % dir)

    return np.array(patterns), np.array(after_patt), indexes_with_profit


def l_infinity(x1, x2):
    return torch.max(torch.abs(x1 - x2), dim=1).values


def euclid_dist(x1, x2):
    return torch.cdist(x1, x2) ** 2


def manhatten_dist(x1, x2):
    dif_tnzr = x1 - x2
    return torch.sum(torch.abs(dif_tnzr))


def clusterized_pattern_save(train_x, PATTERN_SIZE, EXTR_WINDOW, profit_value, OVERLAP):
    out_dir = "outputs/saved_patterns"
    np.save(
        f"{out_dir}/buypat_extrw{EXTR_WINDOW}_patsize{PATTERN_SIZE}_profit{profit_value}_overlap{OVERLAP}",
        train_x,
    )
    print("Файл с паттернами сохранен")


def clusterted_patterns_load(out_dir, file_name):
    train_x = np.load(f"{out_dir}/{file_name}", allow_pickle=True)
    return np.array(train_x)


def get_patterns_and_other_classes_with_profit(
    data_df,
    profit_value,
    EXTR_WINDOW,
    PATTERN_SIZE,
    OVERLAP,
    train_dates,
    save_to_dir=False,
):
    close_price = data_df["Close"].values.tolist()
    min_indexes = argrelmin(np.array(close_price), order=EXTR_WINDOW)[0]
    min_indexes = [i for i in min_indexes if i + EXTR_WINDOW <= len(data_df)]
    prices = []
    steps = [step for step in range(1, EXTR_WINDOW + 1)]
    for i in min_indexes:
        step_prices_for_ecach_idex = []
        for step in steps:
            growth_price = (close_price[(i + step)] - close_price[i]) / close_price[i]
            step_prices_for_ecach_idex.append(growth_price)
        prices.append(step_prices_for_ecach_idex)
    squares_df = pd.DataFrame()
    for colname, vlue_col in zip(min_indexes, prices):
        squares_df[f"{str(colname)}"] = vlue_col
    axey = squares_df.mean(axis=1).values.tolist()
    # axey = [[x if x >= profit_value else 0 for x in axey]]
    print(squares_df.shape)

    conf_intervals = []
    # lconf_intervals = []
    for r in range(len(squares_df)):
        values_arr = squares_df.iloc[r, :].to_numpy()
        # mean_ = np.mean(values_arr)
        std_ = np.std(values_arr) / 2
        # dof = len(values_arr) - 1
        # confidence = 0.95
        # t_crit = np.abs(t.ppf((1 - confidence) / 2, dof))
        # low = mean_ - std_ * t_crit / np.sqrt(len(values_arr))
        # high = mean_ + std_ * t_crit / np.sqrt(len(values_arr))
        # lconf_intervals.append(low)
        conf_intervals.append(std_)

    fig = go.Figure()
    # fig.add_trace(go.Bar(x=[f' Бар : {i} ' for i in steps], y=axey, error_y=dict(type='data',symmetric=False, array=hconf_intervals, arrayminus=lconf_intervals)))
    fig = px.bar(x=[f" Бар : {i} " for i in steps], y=axey, error_y=conf_intervals)
    fig.update_layout(
        title=dict(
            text=f" Прирост цены в  взависимости от удаления от паттерна с доверительными интервалами для 95% данных. Параметры : profit_value = {profit_value}, EXTR_WINDOW = {EXTR_WINDOW}, PATTERN_SIZE,OVERLAP = {OVERLAP} ",
            font=dict(size=15),
        ),
        xaxis_title="Бары",
        yaxis_title="Величина прироста в долях",
    )
    fig.show()

    indexes_with_profit = []

    for i in min_indexes:
        if i + (EXTR_WINDOW) <= len(close_price):

            if (
                close_price[i]
                + (close_price[i] * profit_value)
                - sum(close_price[int(i) : int(i + (EXTR_WINDOW))])
                / len(close_price[int(i) : int(i + (EXTR_WINDOW))])
                <= 0
            ):
                indexes_with_profit.append(i)
    before_patterns = []
    patterns = []
    after_patterns = []
    after_patt = []
    for ind in indexes_with_profit:
        if ind - PATTERN_SIZE >= 0:
            patt = data_df[(ind - PATTERN_SIZE + OVERLAP) : ind + OVERLAP].to_numpy()
            patterns.append(patt)  # df.loc[:, ['foo','bar','dat']]

            if ind + EXTR_WINDOW <= len(data_df):
                aft_pat = data_df["Close"][ind : ind + EXTR_WINDOW].values.tolist()
                after_patt.append(aft_pat)

            else:
                after_patt.append([0 for i in range(EXTR_WINDOW)])
        if ind - 2 * PATTERN_SIZE >= 0:
            b_patt = data_df[
                (ind - 2 * PATTERN_SIZE + OVERLAP) : ind - PATTERN_SIZE + OVERLAP
            ].to_numpy()
            before_patterns.append(b_patt)
        if ind + PATTERN_SIZE <= len(data_df):
            a_patt = data_df[(ind + OVERLAP) : ind + PATTERN_SIZE + OVERLAP].to_numpy()
            after_patterns.append(a_patt)

    fig = go.Figure()  # x=data_df.index.tolist()
    fig.add_trace(
        go.Scatter(x=train_dates, y=data_df["Close"], mode="lines", name="CLOSE")
    )
    fig.add_trace(
        go.Scatter(
            x=train_dates.iloc[indexes_with_profit],
            y=data_df["Close"].iloc[indexes_with_profit],
            mode="markers",
            marker=dict(symbol="triangle-up", size=15),
        )
    )

    fig.update_layout(
        title=f"Разметка данных на основе параметров profit_value = {profit_value}, EXTR_WINDOW = {EXTR_WINDOW}, PATTERN_SIZE = {PATTERN_SIZE}  ",
        xaxis_title="DATE",
        yaxis_title="CLOSE",
        legend_title="Legend",
    )
    fig.show()
    if save_to_dir == True:
        parent_dir = "saved_patterns"
        dir = f"patt_size<{PATTERN_SIZE}>_Ex_window<{EXTR_WINDOW}>_prof_vle<{profit_value}>"
        path = os.path.join(parent_dir, dir)
        try:
            os.makedirs(path, exist_ok=False)
            print("Директория  '%s' создана" % dir)
            patterns_arr = np.array(patterns)
            patterns_reshaped = patterns_arr.reshape(patterns_arr.shape[0], -1)
            np.savetxt(
                f"{path}/buy_patterns_extr_window<{EXTR_WINDOW}>"
                f"_pattern_size{PATTERN_SIZE}.csv",
                patterns_reshaped,
            )
            print("Файл с паттернами сохранен")
        except OSError as error:
            print("Директория '%s' уже существует" % dir)

    before_patterns = np.array(before_patterns).reshape(
        -1, PATTERN_SIZE, len(data_df.columns.to_list()), 1
    )
    patterns = np.array(patterns).reshape(
        -1, PATTERN_SIZE, len(data_df.columns.to_list()), 1
    )
    after_patterns = np.array(after_patterns).reshape(
        -1, PATTERN_SIZE, len(data_df.columns.to_list()), 1
    )

    train_x = [before_patterns, patterns, after_patterns]

    return np.array(patterns), np.array(after_patt), indexes_with_profit, train_x


def test_get_patterns_and_other_classes_with_profit(
    data_df,
    profit_value,
    EXTR_WINDOW,
    PATTERN_SIZE,
    OVERLAP,
    train_dates,
    save_to_dir=False,
):
    close_price = data_df["Close"].values.tolist()

    """buy_indexes = []
    sel_indexes = []
    small_profit_indexes = []
    small_lost_indexes = []

    for index_value in range(len(close_price)):
        if index_value + EXTR_WINDOW <= len(close_price) - 1:
            growth_price = np.mean(np.array(
                [(close_price[(index_value + step)] - close_price[index_value]) / close_price[index_value] for step in
                 range(EXTR_WINDOW)]))
            if growth_price > profit_value:
                buy_indexes.append(index_value)
            elif growth_price > 0 and growth_price < profit_value:
                small_profit_indexes.append(index_value)
            elif growth_price < (-1 * profit_value):
                sel_indexes.append(index_value)
            elif growth_price < 0 and growth_price >= (-1 * profit_value):
                small_lost_indexes

    buy_patterns = []
    small_profit_patterns = []
    sell_patterns = []
    small_lost_patterns = []

    for ind in buy_indexes:
        if ind - PATTERN_SIZE >= 0:
            patt = data_df[(ind - PATTERN_SIZE + OVERLAP): ind + OVERLAP].to_numpy()
            buy_patterns.append(patt)

    for ind in small_profit_indexes:
        if ind - PATTERN_SIZE >= 0:
            patt = data_df[(ind - PATTERN_SIZE + OVERLAP): ind + OVERLAP].to_numpy()
            small_profit_patterns.append(patt)

    for ind in sel_indexes:
        if ind - PATTERN_SIZE >= 0:
            patt = data_df[(ind - PATTERN_SIZE + OVERLAP): ind + OVERLAP].to_numpy()
            sell_patterns.append(patt)

    for ind in small_lost_indexes:
        if ind - PATTERN_SIZE >= 0:
            patt = data_df[(ind - PATTERN_SIZE + OVERLAP): ind + OVERLAP].to_numpy()
            small_lost_patterns.append(patt)

    buy_patterns = np.array(buy_patterns).reshape(-1, PATTERN_SIZE, len(data_df.columns.to_list()), 1)
    small_profit_patterns = np.array(small_profit_patterns).reshape(-1, PATTERN_SIZE, len(data_df.columns.to_list()), 1)
    sell_patterns = np.array(sell_patterns).reshape(-1, PATTERN_SIZE, len(data_df.columns.to_list()), 1)
    small_lost_patterns = np.array(small_lost_patterns).reshape(-1, PATTERN_SIZE, len(data_df.columns.to_list()), 1)
    fig = go.Figure()  # x=data_df.index.tolist()
    fig.add_trace(go.Scatter(x=train_dates, y=data_df['Close'], mode='lines', name='CLOSE'))
    fig.add_trace(go.Scatter(x=train_dates.iloc[buy_indexes], y=data_df['Close'].iloc[buy_indexes],
                             mode='markers',
                             marker=dict(symbol='triangle-up', size=15)))
    fig.add_trace(go.Scatter(x=train_dates.iloc[small_profit_indexes], y=data_df['Close'].iloc[small_profit_indexes],
                             mode='markers',
                             marker=dict(symbol='triangle-up', size=15)))
    fig.add_trace(go.Scatter(x=train_dates.iloc[sel_indexes], y=data_df['Close'].iloc[sel_indexes],
                             mode='markers',
                             marker=dict(symbol='triangle-down', size=15)))
    fig.add_trace(go.Scatter(x=train_dates.iloc[small_lost_indexes], y=data_df['Close'].iloc[small_lost_indexes],
                             mode='markers',
                             marker=dict(symbol='triangle-down', size=15)))

    fig.update_layout(
        title=f'Разметка данных на основе параметров profit_value = {profit_value}, EXTR_WINDOW = {EXTR_WINDOW}, PATTERN_SIZE = {PATTERN_SIZE}  ',
        xaxis_title='DATE', yaxis_title='CLOSE', legend_title='Legend')
    fig.show()"""

    min_indexes = argrelmin(np.array(close_price), order=EXTR_WINDOW)[0]
    min_indexes = [i for i in min_indexes if i + EXTR_WINDOW <= len(data_df)]
    max_indexes = argrelmax(np.array(close_price), order=EXTR_WINDOW)[0]
    max_indexes = [i for i in max_indexes if i + EXTR_WINDOW <= len(data_df)]

    prices = []
    steps = [step for step in range(1, EXTR_WINDOW + 1)]
    for i in min_indexes:
        step_prices_for_ecach_idex = []
        for step in steps:
            growth_price = (close_price[(i + step)] - close_price[i]) / close_price[i]
            step_prices_for_ecach_idex.append(growth_price)
        prices.append(step_prices_for_ecach_idex)
    squares_df = pd.DataFrame()
    for colname, vlue_col in zip(min_indexes, prices):
        squares_df[f"{str(colname)}"] = vlue_col
    axey = squares_df.mean(axis=1).values.tolist()
    # axey = [[x if x >= profit_value else 0 for x in axey]]
    print(squares_df.shape)

    conf_intervals = []
    # lconf_intervals = []
    for r in range(len(squares_df)):
        values_arr = squares_df.iloc[r, :].to_numpy()
        # mean_ = np.mean(values_arr)
        std_ = np.std(values_arr) / 2
        # dof = len(values_arr) - 1
        # confidence = 0.95
        # t_crit = np.abs(t.ppf((1 - confidence) / 2, dof))
        # low = mean_ - std_ * t_crit / np.sqrt(len(values_arr))
        # high = mean_ + std_ * t_crit / np.sqrt(len(values_arr))
        # lconf_intervals.append(low)
        conf_intervals.append(std_)

    fig = go.Figure()
    # fig.add_trace(go.Bar(x=[f' Бар : {i} ' for i in steps], y=axey, error_y=dict(type='data',symmetric=False, array=hconf_intervals, arrayminus=lconf_intervals)))
    fig = px.bar(x=[f" Бар : {i} " for i in steps], y=axey, error_y=conf_intervals)
    fig.update_layout(
        title=dict(
            text=f" Прирост цены в  взависимости от удаления от паттерна с доверительными интервалами для 95% данных. Параметры : profit_value = {profit_value}, EXTR_WINDOW = {EXTR_WINDOW}, PATTERN_SIZE,OVERLAP = {OVERLAP} ",
            font=dict(size=15),
        ),
        xaxis_title="Бары",
        yaxis_title="Величина прироста в долях",
    )
    fig.show()

    indexes_with_profit = []
    for i in min_indexes:
        if i + (EXTR_WINDOW) <= len(close_price):

            if (
                close_price[i]
                + (close_price[i] * profit_value)
                - sum(close_price[int(i) : int(i + (EXTR_WINDOW))])
                / len(close_price[int(i) : int(i + (EXTR_WINDOW))])
                <= 0
            ):
                indexes_with_profit.append(i)

    indexes_lost_profit = []
    for i in max_indexes:
        if i + (EXTR_WINDOW) <= len(close_price):
            if (
                close_price[i]
                + (close_price[i] * profit_value)
                - sum(close_price[int(i) : int(i + (EXTR_WINDOW))])
                / len(close_price[int(i) : int(i + (EXTR_WINDOW))])
                > 0
            ):
                indexes_lost_profit.append(i)

    before_patterns = []
    patterns = []
    after_patterns = []
    after_patt = []
    for ind in indexes_with_profit:
        if ind - PATTERN_SIZE >= 0:
            patt = data_df[(ind - PATTERN_SIZE + OVERLAP) : ind + OVERLAP].to_numpy()
            patterns.append(patt)  # df.loc[:, ['foo','bar','dat']]

            if ind + EXTR_WINDOW <= len(data_df):
                aft_pat = data_df["Close"][ind : ind + EXTR_WINDOW].values.tolist()
                after_patt.append(aft_pat)

            else:
                after_patt.append([0 for i in range(EXTR_WINDOW)])
        if ind - 2 * PATTERN_SIZE >= 0:
            b_patt = data_df[
                (ind - 2 * PATTERN_SIZE + OVERLAP) : ind - PATTERN_SIZE + OVERLAP
            ].to_numpy()
            before_patterns.append(b_patt)
        if ind + PATTERN_SIZE <= len(data_df):
            a_patt = data_df[(ind + OVERLAP) : ind + PATTERN_SIZE + OVERLAP].to_numpy()
            after_patterns.append(a_patt)
    before_sell_pattern = []
    after_sell_patterns = []
    sell_patterns = []
    for ind in indexes_lost_profit:
        if ind - PATTERN_SIZE >= 0:
            patt = data_df[(ind - PATTERN_SIZE + OVERLAP) : ind + OVERLAP].to_numpy()
            sell_patterns.append(patt)
        if ind - 2 * PATTERN_SIZE >= 0:
            b_patt = data_df[
                (ind - 2 * PATTERN_SIZE + OVERLAP) : ind - PATTERN_SIZE + OVERLAP
            ].to_numpy()
            before_sell_pattern.append(b_patt)

        if ind + PATTERN_SIZE <= len(data_df):
            a_patt = data_df[(ind + OVERLAP) : ind + PATTERN_SIZE + OVERLAP].to_numpy()
            after_sell_patterns.append(a_patt)

    fig = go.Figure()  # x=data_df.index.tolist()
    fig.add_trace(
        go.Scatter(x=train_dates, y=data_df["Close"], mode="lines", name="CLOSE")
    )
    fig.add_trace(
        go.Scatter(
            x=train_dates.iloc[indexes_with_profit],
            y=data_df["Close"].iloc[indexes_with_profit],
            mode="markers",
            marker=dict(symbol="triangle-up", size=15, color="green"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=train_dates.iloc[indexes_lost_profit],
            y=data_df["Close"].iloc[indexes_lost_profit],
            mode="markers",
            marker=dict(symbol="triangle-down", size=15, color="red"),
        )
    )

    fig.update_layout(
        title=f"Разметка данных на основе параметров profit_value = {profit_value}, EXTR_WINDOW = {EXTR_WINDOW}, PATTERN_SIZE = {PATTERN_SIZE}  ",
        xaxis_title="DATE",
        yaxis_title="CLOSE",
        legend_title="Legend",
    )
    fig.show()
    if save_to_dir == True:
        parent_dir = "saved_patterns"
        dir = f"patt_size<{PATTERN_SIZE}>_Ex_window<{EXTR_WINDOW}>_prof_vle<{profit_value}>"
        path = os.path.join(parent_dir, dir)
        try:
            os.makedirs(path, exist_ok=False)
            print("Директория  '%s' создана" % dir)
            patterns_arr = np.array(patterns)
            patterns_reshaped = patterns_arr.reshape(patterns_arr.shape[0], -1)
            np.savetxt(
                f"{path}/buy_patterns_extr_window<{EXTR_WINDOW}>"
                f"_pattern_size{PATTERN_SIZE}.csv",
                patterns_reshaped,
            )
            print("Файл с паттернами сохранен")
        except OSError as error:
            print("Директория '%s' уже существует" % dir)

    # norm_patterns = np.array([normalize(i, axis=0, norm='max') for i in patterns]).reshape(-1, PATTERN_SIZE, len(data_df.columns.to_list()), 1)
    # norm_sell_patterns = np.array([normalize(i, axis=0, norm='max') for i in sell_patterns]).reshape(-1, PATTERN_SIZE, len(data_df.columns.to_list()), 1)
    scaler = StandardScaler()
    std_patterns = np.array([scaler.fit_transform(i) for i in patterns]).reshape(
        -1, PATTERN_SIZE, len(data_df.columns.to_list()), 1
    )
    std_sell_patterns = np.array(
        [scaler.fit_transform(i) for i in sell_patterns]
    ).reshape(-1, PATTERN_SIZE, len(data_df.columns.to_list()), 1)

    before_patterns = np.array(before_patterns).reshape(
        -1, PATTERN_SIZE, len(data_df.columns.to_list()), 1
    )
    patterns = np.array(patterns).reshape(
        -1, PATTERN_SIZE, len(data_df.columns.to_list()), 1
    )
    after_patterns = np.array(after_patterns).reshape(
        -1, PATTERN_SIZE, len(data_df.columns.to_list()), 1
    )
    sell_patterns = np.array(sell_patterns).reshape(
        -1, PATTERN_SIZE, len(data_df.columns.to_list()), 1
    )
    before_sell_pattern = np.array(before_sell_pattern).reshape(
        -1, PATTERN_SIZE, len(data_df.columns.to_list()), 1
    )
    after_sell_patterns = np.array(after_sell_patterns).reshape(
        -1, PATTERN_SIZE, len(data_df.columns.to_list()), 1
    )

    train_x = [std_patterns, std_sell_patterns]

    return np.array(patterns), np.array(after_patt), indexes_with_profit, train_x
