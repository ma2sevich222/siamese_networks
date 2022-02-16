import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
from sklearn.metrics.pairwise import cosine_distances

pd.options.mode.chained_assignment = None

"""get_locals принимает на вход df['column_name] с данными  и размер окна внутри которого выделять локальные экстремумы.
Выводит график c размеченными точками локальных минимумов и максимумов"""


def get_locals(data_df, n):  # данные подаются в формате df
    data_df["index1"] = data_df.index
    data_df["min"] = data_df.iloc[
        argrelextrema(data_df.Close.values, np.less_equal, order=n)[0]
    ]["Close"]
    data_df["max"] = data_df.iloc[
        argrelextrema(data_df.Close.values, np.greater_equal, order=n)[0]
    ]["Close"]
    f = plt.figure()
    f.set_figwidth(80)
    f.set_figheight(65)
    plt.scatter(data_df.index1, data_df["min"], c="r", label='MIN')
    plt.scatter(data_df.index1, data_df["max"], c="g", label='MAX')
    plt.plot(data_df.index1, data_df["Close"])
    plt.ylabel('CLOSE')
    plt.title('График локальный минимумов и максимумов')
    plt.legend()
    plt.show()


"""predictions_plotting-  на вход подается:  data - датафрэйм с результатами теста сети
tresh_list- список границы предсказания, на графике будут отображены предсказания ниже или равные значениям границ в списке
pattern_list - список паттернов которые нужны отобразить"""


def predictions_plotting(data, tresh_list, pattern_list):
    filtered_df = []
    for i, j in zip(tresh_list, pattern_list):
        sample_df = data[
            (data.distance <= i) & (data.signal == 1) & (data.pattern == j)
            ]
        filtered_df.append(sample_df)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=data["date"], y=data["close"], mode="lines", name="CLOSE")
    )
    for i, j, k in zip(filtered_df, tresh_list, pattern_list):
        fig.add_trace(
            go.Scatter(
                x=i["date"],
                y=i["close"],
                mode="markers",
                name=f"distance <= {j}/patern:{k}",
                marker=dict(symbol="triangle-up", size=15),
            )
        )
    fig.update_layout(
        title="BUY signals predictions",
        xaxis_title="DATE",
        yaxis_title="CLOSE",
        legend_title="Legend",
    )

    fig.show()


""" функция pattern_samples_plot
patterns-массив с размеченными паттернами, Eval_df-данные, которые использовались для проверки сети,
eval_results-результат прдедсказаний сети,pattern_No - интересующий нас паттерн.
Результат работы функции: Серия графиков в формате ohlc ,
где слева график введенного паттерна, справа  график образца
входных данных для которого сеть предсказала этот паттерн """


def pattern_samples_plot(patterns, Eval_df, eval_results, pattern_No):
    indexes = eval_results[
        (eval_results.signal == 1) & (eval_results.pattern == pattern_No)
        ].index
    if len(indexes) == 0:
        print("Данный паттерн не был обнаружен в данных")
    else:
        print(f"Найдено совпадений: {len(indexes)}")
        for i in indexes:
            plot_sample = Eval_df[i].reset_index()
            plt.figure(figsize=[5, 5])
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 3))
            ax1.set(ylabel="PRICE", title=f"Размеченный паттерн: {pattern_No}")
            ax2.set(
                ylabel="PRICE",
                title=f"Участок тренда, определенный как данный паттерн и дистанция: {i}/ distance:{round(eval_results.distance[i], 4)}",
            )
            width = 0.4
            width2 = 0.05

            ax1_up = patterns[pattern_No][
                patterns[pattern_No].Close >= patterns[pattern_No].Open
                ]
            ax1_down = patterns[pattern_No][
                patterns[pattern_No].Close < patterns[pattern_No].Open
                ]
            ax2_up = plot_sample[plot_sample.Close >= plot_sample.Open]
            ax2_down = plot_sample[plot_sample.Close < plot_sample.Open]

            col1 = "green"
            col2 = "red"

            ax1.bar(
                ax1_up.index,
                ax1_up.Close - ax1_up.Open,
                width,
                bottom=ax1_up.Open,
                color=col1,
            )
            ax1.bar(
                ax1_up.index,
                ax1_up.High - ax1_up.Close,
                width2,
                bottom=ax1_up.Close,
                color=col1,
            )
            ax1.bar(
                ax1_up.index,
                ax1_up.Low - ax1_up.Open,
                width2,
                bottom=ax1_up.Open,
                color=col1,
            )
            ax1.bar(
                ax1_down.index,
                ax1_down.Close - ax1_down.Open,
                width,
                bottom=ax1_down.Open,
                color=col2,
            )
            ax1.bar(
                ax1_down.index,
                ax1_down.High - ax1_down.Open,
                width2,
                bottom=ax1_down.Open,
                color=col2,
            )
            ax1.bar(
                ax1_down.index,
                ax1_down.Low - ax1_down.Close,
                width2,
                bottom=ax1_down.Close,
                color=col2,
            )

            ax2.bar(
                ax2_up.index,
                ax2_up.Close - ax2_up.Open,
                width,
                bottom=ax2_up.Open,
                color=col1,
            )
            ax2.bar(
                ax2_up.index,
                ax2_up.High - ax2_up.Close,
                width2,
                bottom=ax2_up.Close,
                color=col1,
            )
            ax2.bar(
                ax2_up.index,
                ax2_up.Low - ax2_up.Open,
                width2,
                bottom=ax2_up.Open,
                color=col1,
            )
            ax2.bar(
                ax2_down.index,
                ax2_down.Close - ax2_down.Open,
                width,
                bottom=ax2_down.Open,
                color=col2,
            )
            ax2.bar(
                ax2_down.index,
                ax2_down.High - ax2_down.Open,
                width2,
                bottom=ax2_down.Open,
                color=col2,
            )
            ax2.bar(
                ax2_down.index,
                ax2_down.Low - ax2_down.Close,
                width2,
                bottom=ax2_down.Close,
                color=col2,
            )

    plt.show()


"""calculate_cos_dist - принимает массив паттернов и номер паттерна который интересует, возвращает словарь номер паттерна- дистанция.
данный словарь является аргументом функции ниже, для отрисовки паттернов из этого словаря"""


def calculate_cos_dist(patterns, pattern):
    pattern_array = patterns[pattern].reshape(
        -1, patterns[pattern].shape[0] * patterns[pattern].shape[1]
    )
    pattern_matrix = patterns.reshape(-1, patterns.shape[1] * patterns.shape[2])
    cos_distance_array = cosine_distances(pattern_array, pattern_matrix)
    nearlest_patterns_names = np.argsort(cos_distance_array[0])
    nearlest_pattern_distances = np.sort(cos_distance_array[0])
    nearlist_neibors = dict(
        zip(nearlest_patterns_names[:10], nearlest_pattern_distances[:10])
    )

    return nearlist_neibors


"""Принимает масив паттернов и список из функции выше, отрисовывает интересующий паттерн и ближайшие к нему"""


def plot_nearlist_patterns(
        patterns, nearlist_neibors
):  # передаем массив паттернов и список ближайших паттернов

    keys = list(nearlist_neibors.keys())
    values = list(nearlist_neibors.values())

    fig = make_subplots(
        rows=3,
        cols=4,
        subplot_titles=(
            f"Представленный образец:{keys[0]}",
            f"Паттерн:{keys[1]} дистанция:{round(values[1], 4)}",
            f"Паттерн: {keys[2]} дистанция: {round(values[2], 4)}",
            f"Паттерн: {keys[3]} дистанция: {round(values[3], 4)}",
            f"Паттерн: {keys[4]} дистанция: {round(values[4], 4)}",
            f"Паттерн: {keys[5]} дистанция: {round(values[5], 4)}",
            f"Паттерн: {keys[6]} дистанция: {round(values[6], 4)}",
            f"Паттерн: {keys[7]} дистанция: {round(values[7], 4)}",
            f"Паттерн: {keys[8]} дистанция: {round(values[8], 4)}",
            f"Паттерн: {keys[9]} дистанция: {round(values[9], 4)}",
        ),
    )

    def plot_candy(pattern, row, col):
        fig.add_trace(
            go.Candlestick(
                x=np.array([i for i in range(len(pattern))]),
                open=pattern["Open"].values,
                high=pattern["High"].values,
                low=pattern["Low"].values,
                close=pattern["Close"].values,
            ),
            secondary_y=False,
            row=row,
            col=col,
        )

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

    fig.update_xaxes(rangeslider={"visible": False}, row=1, col=1)
    fig.update_xaxes(rangeslider={"visible": False}, row=1, col=2)
    fig.update_xaxes(rangeslider={"visible": False}, row=1, col=3)
    fig.update_xaxes(rangeslider={"visible": False}, row=1, col=4)
    fig.update_xaxes(rangeslider={"visible": False}, row=2, col=1)
    fig.update_xaxes(rangeslider={"visible": False}, row=2, col=2)
    fig.update_xaxes(rangeslider={"visible": False}, row=2, col=3)
    fig.update_xaxes(rangeslider={"visible": False}, row=2, col=4)
    fig.update_xaxes(rangeslider={"visible": False}, row=3, col=1)
    fig.update_xaxes(rangeslider={"visible": False}, row=3, col=2)

    fig.update_layout(
        title_text="Визуальное сравнение размеченных паттернов гарантирующих рост. Слева на право : образец и 9 ближайших по возрастанию дистанции от образца."
    )

    fig.show()  # на выходе слева на право интересующий нас паттерн и 9 ближайших


"""Хит-мэп паттернов по косинусному расстоянию"""


def patterns_heatmap(patterns):
    reshaped = patterns.reshape(-1, patterns.shape[1] * patterns.shape[2])
    cos_distance_matrix = cosine_distances(reshaped)
    fig = px.imshow(
        cos_distance_matrix,
        aspect="auto",
        labels=dict(x="Pattern", y="Pattern", color="Distance"),
        title="Patters heatmap",
    )
    fig.show()
