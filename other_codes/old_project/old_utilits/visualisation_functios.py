import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
from sklearn.decomposition import PCA
from sklearn.metrics import auc
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import StandardScaler

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
    # f = plt.figure()
    # f.set_figwidth(80)
    # f.set_figheight(65)
    # plt.scatter(data_df.index1, data_df["min"], c="r", label='MIN')
    # plt.scatter(data_df.index1, data_df["max"], c="g", label='MAX')
    # plt.plot(data_df.index1, data_df["Close"])
    # plt.ylabel('CLOSE')
    # plt.title('График локальный минимумов и максимумов')
    # plt.legend()
    # plt.show()


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


def pattern_samples_plot(patterns, Eval_df, eval_results, pattern_num):
    sorted = eval_results.sort_values(by=["distance"], ascending=False)

    indexes = sorted[(sorted.pattern == pattern_num)].index
    if len(indexes) == 0:
        print("Данный паттерн не был обнаружен в данных")

    else:

        print(f"Найдено совпадений: {len(indexes)}")

        if len(indexes) > 24:
            indexes = indexes[:24]

        list_of_plots = [Eval_df[i].reset_index() for i in indexes]
        list_of_plots.insert(0, patterns[pattern_num])

        number_rows = int(math.ceil(len(list_of_plots) / 5))
        number_cols = 5

        names = [eval_results.distance[i] for i in indexes]
        plots_annotations = [f"Паттерн: {pattern_num}"]

        for i, j in zip(indexes, names):
            add = f"Шаг {str(i)}, Дистанция {str(round(j, 4))}"
            plots_annotations.append(add)

        fig = make_subplots(
            rows=number_rows, cols=number_cols, subplot_titles=plots_annotations
        )

        k = 0
        for row in range(1, number_rows + 1):
            for col in range(1, 6):

                fig.add_trace(
                    go.Candlestick(
                        x=np.array([i for i in range(len(list_of_plots[k]))]),
                        open=list_of_plots[k]["Open"].values,
                        high=list_of_plots[k]["High"].values,
                        low=list_of_plots[k]["Low"].values,
                        close=list_of_plots[k]["Close"].values,
                    ),
                    secondary_y=False,
                    row=row,
                    col=col,
                )
                fig.update_xaxes(rangeslider={"visible": False}, row=row, col=col)
                k += 1
                if k == len(list_of_plots):
                    break

        fig.update_layout(
            width=2400,
            height=1400,
            title_text="Визуальное сравннение паттерна и предсказаний на проверочных данных",
        )
        fig.update_yaxes(automargin=True)
        fig.show()


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
        title="Patterns distances heatmap",
    )
    fig.show()


def triplet_pred_plotting(data, tresh_list, pattern_list, plot_name):
    check_len = len([i for i in data["distance"].values.tolist() if i <= tresh_list[0]])
    if check_len == 0:
        print(
            "Отсутствуют предсказнаия с заданной границей, попробуйте взять другую величину tresh_hold"
        )
    else:

        filtered_df = []
        for i, j in zip(tresh_list, pattern_list):
            sample_df = data[(data.distance <= i) & (data.label == j)]
            filtered_df.append(sample_df)

            df_pattern = data[(data.label == j)]
            sns.displot(df_pattern["distance"])
            plt.title(
                f"pattern = {j}:\n"
                f'min distance = {np.round(df_pattern["distance"].min(), 4)},   '
                f'max distance = {np.round(df_pattern["distance"].max(), 4)}'
            )
            plt.show()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=data["date"], y=data["close"], mode="lines", name="CLOSE")
        )
        for i, j, k in zip(filtered_df, tresh_list, pattern_list):

            if int(k) % 2 == 0:
                mark = "triangle-up"
            else:
                mark = "triangle-down"

            fig.add_trace(
                go.Scatter(
                    x=i["date"],
                    y=i["close"],
                    mode="markers",
                    text=i["distance"],
                    name=f"distance <= {j} / patern:{k}",
                    marker=dict(symbol=mark, size=15),
                )
            )

        fig.update_layout(
            title=f"signals predictions for file {plot_name}",
            xaxis_title="DATE",
            yaxis_title="CLOSE",
            legend_title="Legend",
        )
        fig.show()


def triplet_pattern_class_plot(
    list_of_pattern_class, patterns, name, i, EXTR_WINDOW, PATTERN_SIZE
):
    list_of_plots = [patterns[i] for i in list_of_pattern_class["pattern"].values]
    """areas = []
    for patt in list_of_plots:
        x = np.array([a for a in range(len(patt))])
        y = np.array(patt[:, [3]])
        S =  trapz(y, dx= 1)
        areas.append(S)
    sorted_list_plots = [patterns[i] for i in sorted(areas.index,reverse=True)]"""

    number_rows = int(math.ceil(len(list_of_plots) / 5))
    number_cols = 5

    # names = list_of_pattern_class['pattern'].values.tolist()
    plots_annotations = [
        f"Паттерн: {i} , Дистанция: {str(round(j, 6))}"
        for i, j in zip(
            list_of_pattern_class["pattern"], list_of_pattern_class["distance"]
        )
    ]

    fig = make_subplots(
        rows=number_rows, cols=number_cols, subplot_titles=plots_annotations
    )

    k = 0
    for row in range(1, number_rows + 1):
        for col in range(1, 6):

            fig.add_trace(
                go.Candlestick(
                    x=np.array([i for i in range(len(list_of_plots[k]))]),
                    open=list_of_plots[k]["Open"].values,
                    high=list_of_plots[k]["High"].values,
                    low=list_of_plots[k]["Low"].values,
                    close=list_of_plots[k]["Close"].values,
                ),
                secondary_y=False,
                row=row,
                col=col,
            )
            fig.update_xaxes(rangeslider={"visible": False}, row=row, col=col)
            k += 1
            if k == len(list_of_plots):
                break

    fig.update_layout(
        width=1800,
        height=400,
        title_text=f"Визуальное сравннение {name} паттернов  класса {i} c параметрами: EXTR_WINDOW = {EXTR_WINDOW}, PATTERN_SIZE = {PATTERN_SIZE}  ",
    )
    fig.update_yaxes(automargin=True)
    fig.show()


def plot_clasters(
    patterns, pattern_labels, EXTR_WINDOW, PATTERN_SIZE, profit_value, sil_score_max
):
    scaler = StandardScaler()
    patterns_std = scaler.fit_transform(
        patterns.reshape(-1, patterns.shape[1] * patterns.shape[2])
    )
    pca = PCA(n_components=2)
    patterns_PCA = pd.DataFrame(
        pca.fit_transform(patterns_std)
    )  # df.rename(columns={'oldName1': 'newName1','oldName2': 'newName2'},inplace=True, errors='raise')
    patterns_PCA["label"] = pattern_labels
    patterns_PCA.rename(
        columns={0: "PCA feature 1", 1: "PCA feature 2"}, inplace=True, errors="raise"
    )
    fig = px.scatter(
        patterns_PCA,
        x="PCA feature 1",
        y="PCA feature 2",
        color="label",
        symbol="label",
        title=f" Наиболее подходящий результат кластеризации соглано silhouette_score = {round(sil_score_max, 2)}. Параметры паттернов: EXTR_WINDOW = {EXTR_WINDOW}, PATTERN_SIZE = {PATTERN_SIZE}, profit_value = {profit_value} ",
    )
    fig.update_traces(
        marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )
    fig.show()


"""def extend_plotting(data, tresh_list, pattern_list):
    filtered_df = []
    for i, j in zip(tresh_list, pattern_list):
        sample_df = data[(data.distance <= i) & (data.pattern == j)]
        filtered_df.append(sample_df)

        df_pattern = data[(data.pattern == j)]
        sns.displot(df_pattern["distance"])
        plt.title(f'pattern = {j}:\n'
                  f'min distance = {np.round(df_pattern["distance"].min(), 4)},   '
                  f'max distance = {np.round(df_pattern["distance"].max(), 4)}')
        plt.show()


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data['close'], mode='lines', name='CLOSE'))
    for i, j, k in zip(filtered_df, tresh_list, pattern_list):
        fig.add_trace(go.Scatter(x=i['date'], y=i['close'], mode='markers', text=i['distance'],
                                 name=f"distance >= {j} / patern:{k}",
                                 marker=dict(symbol='triangle-up', size=15)))

    fig.update_layout(title=f'BUY signals predictions for file ',
                      xaxis_title='DATE', yaxis_title='CLOSE', legend_title='Legend')
    fig.show() """


def cluster_triplet_pattern_class_plot(
    list_of_pattern_class,
    patterns,
    after_patt,
    name,
    clas_no,
    profit_value,
    EXTR_WINDOW,
    PATTERN_SIZE,
):
    areas = []
    for ind in list_of_pattern_class:
        x = np.array([a for a in range(len(after_patt[ind]))])

        y = np.array(after_patt[ind])
        S = auc(x, y) - auc(x, [y[0] for i in range(EXTR_WINDOW)])

        areas.append(S)

    sorted_list_of_pattern_class = [
        i for _, i in sorted(zip(areas, list_of_pattern_class), reverse=True)
    ]
    sorted_area = sorted(areas, reverse=True)
    list_of_plots = [patterns[i] for i in np.array(sorted_list_of_pattern_class)]

    if len(list_of_plots) > 20:
        list_of_plots = list_of_plots[:20]

    number_rows = int(math.ceil(len(list_of_plots) / 5))
    number_cols = 5

    # names = list_of_pattern_class['pattern'].values.tolist()
    plots_annotations = [
        f"Паттерн: {i}" for i, k in zip(sorted_list_of_pattern_class, sorted_area)
    ]

    fig = make_subplots(
        rows=number_rows, cols=number_cols, subplot_titles=plots_annotations
    )

    k = 0
    for row in range(1, number_rows + 1):
        for col in range(1, 6):

            fig.add_trace(
                go.Candlestick(
                    x=np.array([i for i in range(len(list_of_plots[k]))]),
                    open=list_of_plots[k]["Open"].values,
                    high=list_of_plots[k]["High"].values,
                    low=list_of_plots[k]["Low"].values,
                    close=list_of_plots[k]["Close"].values,
                ),
                secondary_y=False,
                row=row,
                col=col,
            )
            fig.update_xaxes(rangeslider={"visible": False}, row=row, col=col)
            k += 1
            if k == len(list_of_plots):
                break

    fig.update_layout(
        width=1400,
        height=800,
        title_text=f"Визуальное сравннение {name} паттернов  клаcтера {clas_no}. Параметры : profit_value = {profit_value}, EXTR_WINDOW = {EXTR_WINDOW}, PATTERN_SIZE = {PATTERN_SIZE} ",
    )
    fig.update_yaxes(automargin=True)
    fig.show()


def named_patterns_heatmap(patterns, name, EXTR_WINDOW, PATTERN_SIZE):
    reshaped = patterns.reshape(-1, patterns.shape[1] * patterns.shape[2])
    cos_distance_matrix = cosine_distances(reshaped)
    fig = px.imshow(
        cos_distance_matrix,
        aspect="auto",
        labels=dict(x="Pattern", y="Pattern", color="Distance"),
        title=f"{name} Patterns distances heatmap with parameters : EXTR_WINDOW = {EXTR_WINDOW},PATTERN_SIZE = {PATTERN_SIZE} ",
    )
    fig.show()


"""def extend_plotting(data, tresh_list, pattern_list, file_name):
    filtered_df = []
    for i, j in zip(tresh_list, pattern_list):
        sample_df = data[(data.distance <= i) & (data.pattern == j)]
        filtered_df.append(sample_df)

        df_pattern = data[(data.pattern == j)]
        sns.displot(df_pattern["distance"])
        plt.title(f'pattern = {j}:\n'
                  f'min distance = {np.round(df_pattern["distance"].min(), 4)},   '
                  f'max distance = {np.round(df_pattern["distance"].max(), 4)}')
        plt.show()
        # print(filtered_df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data['close'], mode='lines', name='CLOSE'))
    for i, j, k in zip(filtered_df, tresh_list, pattern_list):
        fig.add_trace(go.Scatter(x=i['date'], y=i['close'], mode='markers', name=f"distance >= {j} / patern:{k}",
                                 marker=dict(symbol='triangle-up', size=15)))

    fig.update_layout(title=f'BUY signals predictions for file {file_name}',
                      xaxis_title='DATE', yaxis_title='CLOSE', legend_title='Legend')
    fig.show()"""


def mid_dist_add_prediction_plots(root_as_string, TRESHHOLD_DISTANCE):
    # df = pd.read_csv(f'{source_root}/{profit_test_with_clustering_file_name}')
    df = pd.read_csv(root_as_string)
    # df = df.rename(columns={"pattern No.": "pattern"})
    # del df['Unnamed: 0']
    print(df)
    print(f'\nВсего распознано уникальных паттернов:\t{len(pd.unique(df["label"]))}')
    num_patterns = pd.value_counts(df["label"]).to_frame()
    print(f"Распределение числа распознанных паттернов:\n{num_patterns.T}\n")

    fig, ax = plt.subplots()
    ax.bar(num_patterns.index, num_patterns["label"])
    ax.set_xlabel("Номер паттерна")
    ax.set_ylabel("Число распознаваний")
    plt.title(
        f"Число определенных паттернов по видам\n при treshhold_distance = {TRESHHOLD_DISTANCE}"
    )
    plt.show()

    return df


def extend_plotting(data, treshhold, name):
    patterns_each_class = [col for col in data.columns if "labels_clas" in col]

    for clas, column in enumerate(patterns_each_class):
        pattern_list = pd.value_counts(data[column]).to_frame()
        pattern_list = pattern_list.index.to_list()
        tresh_list = [treshhold for i in range(len(pattern_list))]
        filtered_df = []
        for i, j in zip(tresh_list, pattern_list):
            sample_df = data[
                (data[f"distance_class_{clas}"] >= i[0])
                & (data[f"distance_class_{clas}"] <= i[1])
                & (data[f"labels_clas_{clas}"] == j)
            ]
            filtered_df.append(sample_df)

            df_pattern = data[(data[f"labels_clas_{clas}"] == j)]
            sns.displot(df_pattern[f"distance_class_{clas}"])
            plt.title(
                f"pattern = {j}:\n"
                f'min distance = {np.round(df_pattern[f"distance_class_{clas}"].min(), 4)},'
                f'max distance = {np.round(df_pattern[f"distance_class_{clas}"].max(), 4)}'
            )
            plt.show()
        # print(filtered_df)
        sell_periods = []
        buy_periods = []
        hold_periods = []
        sell_trash = 1.8
        hold_trash_per = [0.1, 1.8]
        buy_trash = 0.1
        for ind, i in enumerate(data[f"distance_class_{clas}"].values.tolist()):
            if i >= sell_trash:
                sell_periods.append(ind)
            elif i > hold_trash_per[0] and i < hold_trash_per[1]:
                hold_periods.append(ind)
            elif i <= buy_trash:
                buy_periods.append(ind)

        # fig=go.Figure()#make_subplots(rows=2, cols=1,shared_xaxes=True)
        # colors=[]
        # for i in periods:
        # if i == 'hold':
        # colors.append('yellow')
        # elif i== 'sell':
        # colors.append('firebrick')
        # elif i =='buy':
        # colors.append('green')

        # colors = {'hold': 'steelblue',
        # 'buy': 'green',
        # 'sell': 'firebrick'}
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)  # go.Figure()
        fig.add_trace(
            go.Scatter(x=data.date, y=data["close"], mode="lines", name="CLOSE")
        )

        fig.add_trace(
            go.Bar(
                x=data["date"][sell_periods],
                y=[86 for i in range(len(sell_periods))],
                marker_color=["red" for clr in range(len(sell_periods))],
                name=f"Ожидается падение цены, нижняя граница дистанции  = {sell_trash}",
                opacity=0.4,
            )
        )
        fig.add_trace(
            go.Bar(
                x=data["date"][hold_periods],
                y=[86 for i in range(len(hold_periods))],
                marker_color=["yellow" for clr in range(len(hold_periods))],
                name=f"Незначительное движение цены, период  = {hold_trash_per}",
                opacity=0.4,
            )
        )
        fig.add_trace(
            go.Bar(
                x=data["date"][buy_periods],
                y=[86 for i in range(len(buy_periods))],
                marker_color=["green" for clr in range(len(buy_periods))],
                name=f"Ожидается рост цены, верхняя граница дистанции  = {buy_trash}",
                opacity=0.4,
            )
        )
        # for i, j, k in zip(filtered_df, tresh_list, pattern_list):
        # fig.add_trace(go.Scatter(x=i.index, y=i['close'], text=i[f"distance_class_{clas}"], mode='markers',
        # name=f"assure_period = {j} / patern:{k}",
        # marker=dict(symbol='triangle-up', size=15)),1, 1)'''
        fig.add_trace(
            go.Bar(
                x=data.date,
                y=data[f"distance_class_{clas}"],
                marker_color="crimson",
                name="Динамика изменения дистанции предсказания",
            ),
            2,
            1,
        )
        fig.update_yaxes(title_text="Дистанция предсказания", row=2, col=1)
        fig.update_layout(
            title=f"Движение цены предсказанное сетью для файла {name} ",
            xaxis_title="DATE",
            yaxis_title="CLOSE",
            legend_title="Legend",
        )

        fig.show()


def cos_similaryrty_extend_plotting(data, tresh_list, pattern_list, name):
    filtered_df = []
    for i, j in zip(tresh_list, pattern_list):
        sample_df = data[(data.distance >= i) & (data.pattern == j)]
        filtered_df.append(sample_df)

        df_pattern = data[(data.pattern == j)]
        sns.displot(df_pattern["distance"])
        plt.title(
            f"pattern = {j}:\n"
            f'min distance = {np.round(df_pattern["distance"].min(), 4)},   '
            f'max distance = {np.round(df_pattern["distance"].max(), 4)}'
        )
        plt.show()
        # print(filtered_df)

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
                name=f"distance >= {j} / patern:{k}",
                marker=dict(symbol="triangle-up", size=15),
            )
        )

    fig.update_layout(
        title=f"BUY signals predictions for file {name}",
        xaxis_title="DATE",
        yaxis_title="CLOSE",
        legend_title="Legend",
    )
    fig.show()


def add_prediction_plots(root_as_string, TRESHHOLD_DISTANCE):
    # df = pd.read_csv(f'{source_root}/{profit_test_with_clustering_file_name}')
    df = pd.read_csv(root_as_string)
    # df = df.rename(columns={"pattern No.": "pattern"})
    # del df['Unnamed: 0']
    print(df)
    print(f'\nВсего распознано уникальных паттернов:\t{len(pd.unique(df["pattern"]))}')
    num_patterns = pd.value_counts(df["pattern"]).to_frame()
    print(f"Распределение числа распознанных паттернов:\n{num_patterns.T}\n")

    fig, ax = plt.subplots()
    ax.bar(num_patterns.index, num_patterns["pattern"])
    ax.set_xlabel("Номер паттерна")
    ax.set_ylabel("Число распознаваний")
    plt.title(
        f"Число определенных паттернов по видам\n при treshhold_distance = {TRESHHOLD_DISTANCE}"
    )
    plt.show()

    return df


def class_full_analysys(
    test_results,
    eval_df,
    patterns,
    PATTERN_SIZE,
    EXTR_WINDOW,
    profit_value,
    OVERLAP,
    classNo,
    save_best=False,
):
    close_price = test_results["close"].values.tolist()
    eval_samp_df = [
        eval_df[i - PATTERN_SIZE : i]
        for i in range(len(eval_df))
        if i - PATTERN_SIZE >= 0
    ]
    profit_columns = []
    areas = []
    for ind in range(len(test_results)):
        if ind + EXTR_WINDOW <= len(test_results) - 1:
            x = np.array([a for a in range(EXTR_WINDOW)])
            y = test_results[ind : ind + EXTR_WINDOW]["close"].to_numpy()
            S = auc(x, y) - auc(x, [y[0] for i in range(EXTR_WINDOW)])
            areas.append(S)
        else:
            areas.append(0)

        """else:
            size_window= len(test_results) - EXTR_WINDOW
            if size_window <2:
                areas.append(0)
            else:
                x = np.array([a for a in range(size_window)])
                y = test_results[ind:ind + size_window]['close'].to_numpy()
                S = auc(x, y) - auc(x, [y[0] for i in range(size_window)])
                areas.append(S)"""
    future_grouth_pricecolumn = []
    for i in range(len(test_results)):

        get_profit_list = []
        future_grouth_price = []

        for j in range(1, EXTR_WINDOW + 1):

            if i + j <= len(test_results) - 1:

                value = test_results.iloc[i][test_results.columns.get_loc("close")]
                profit_from_value = (
                    test_results.iloc[i][test_results.columns.get_loc("close")]
                    * profit_value
                )
                growth_price = (
                    (close_price[(i + j)] - close_price[i]) / close_price[i]
                ) * 100
                growt_with_prof = (
                    (close_price[i] + (close_price[i] * profit_value)) - close_price[i]
                ) / close_price[i]
                if growth_price < growt_with_prof:
                    future_grouth_price.append(0)
                else:
                    future_grouth_price.append(growth_price)

                if (
                    value
                    + profit_from_value
                    - (test_results.iloc[i + j][test_results.columns.get_loc("close")])
                    <= 0
                ):
                    get_profit_list.append(1)
                else:
                    get_profit_list.append(0)
            else:
                get_profit_list.append(0)
                future_grouth_price.append(0)

        future_grouth_pricecolumn.append(future_grouth_price)
        profit_columns.append(get_profit_list)

    for gr_price in range(len(future_grouth_pricecolumn[0])):
        test_results[f"step_g_{gr_price + 1}"] = [
            b[gr_price] for b in future_grouth_pricecolumn
        ]

    test_results["future_areas"] = areas

    for i in range(len(profit_columns[0])):
        test_results[f"profit_after_EXTR_WINDOW = {i + 1}"] = [
            b[i] for b in profit_columns
        ]

    dict_unig = test_results[f"labels_clas_{classNo}"].value_counts()
    dict_unig = dict_unig.to_dict()

    best_patterns_in_class = []

    for a in list(dict_unig.keys()):

        label_df = test_results.loc[(test_results[f"labels_clas_{classNo}"] == a)]
        n_act = len(label_df)
        label_pred_window = label_df.iloc[:, -EXTR_WINDOW:]
        label_summary = np.sum(label_pred_window.to_numpy(), axis=0)
        label_summary = np.array([round(i / len(label_df), 3) for i in label_summary])

        names = list(eval_df.columns.values)
        pattern = pd.DataFrame(
            patterns[classNo][a].reshape(PATTERN_SIZE, len(names)), columns=names
        )
        if len(label_df) >= 14:
            label_df = label_df.iloc[:: int(len(label_df) / 14)]
        # строим график
        fig = make_subplots(
            rows=len(label_df) + 1,
            cols=2,
            subplot_titles=(
                f"Доля срабатываний паттерна от общего числа, через n баров от точки предсказания",
                f'Образец паттерна {a}, ср.дистанция = {round(label_df[f"distance_class_{classNo}"].mean(), 4)}, число срабатываний = {n_act}',
                f"Участок определенный как паттерн",
                f"Следующий участок за паттерном  равный {EXTR_WINDOW} барам",
            ),
        )
        # добавляем график срабатываний
        fig.add_trace(
            go.Bar(x=[i for i in range(1, EXTR_WINDOW)], y=label_summary), 1, 1
        )
        # fig.update_xaxes(title_text="n шагов от предсказания", row=1, col=1)
        fig.update_yaxes(
            title_text=f"доля срабатываний c учетом profit_value  = {profit_value}",
            row=1,
            col=1,
        )
        # добавляем паттерн
        fig.add_trace(
            go.Candlestick(
                x=pattern.index.values,
                open=pattern["Open"],
                high=pattern["High"],
                low=pattern["Low"],
                close=pattern["Close"],
                name="Дистанция в 1 примере",
            ),
            1,
            2,
        )

        fig.update_xaxes(rangeslider={"visible": False}, row=1, col=2)

        for i, j in enumerate(list(label_df.index.values)):
            col = 1

            if j + (PATTERN_SIZE) + EXTR_WINDOW <= len(eval_df):
                fig.add_trace(
                    go.Candlestick(
                        x=eval_samp_df[j].index.values,
                        open=eval_samp_df[j]["Open"],
                        high=eval_samp_df[j]["High"],
                        low=eval_samp_df[j]["Low"],
                        close=eval_samp_df[j]["Close"],
                        name=f"{round(label_df[f'distance_class_{classNo}'][j], 4)}",
                    ),
                    i + 2,
                    col,
                )

                fig.update_xaxes(rangeslider={"visible": False}, row=i + 2, col=col)

                col += 1

                fig.add_trace(
                    go.Candlestick(
                        x=eval_df[
                            j + PATTERN_SIZE : j + PATTERN_SIZE + EXTR_WINDOW
                        ].index.values,
                        # .append(eval_samp_df[j + PATTERN_SIZE + EXTR_WINDOW]).index.values,
                        open=eval_df[j + PATTERN_SIZE : j + PATTERN_SIZE + EXTR_WINDOW][
                            "Open"
                        ],
                        high=eval_df[j + PATTERN_SIZE : j + PATTERN_SIZE + EXTR_WINDOW][
                            "High"
                        ],
                        low=eval_df[j + PATTERN_SIZE : j + PATTERN_SIZE + EXTR_WINDOW][
                            "Low"
                        ],
                        close=eval_df[
                            j + PATTERN_SIZE : j + PATTERN_SIZE + EXTR_WINDOW
                        ]["Close"],
                        name=f"Дистация в {i + 2} примере",
                    ),
                    i + 2,
                    col,
                )

                fig.update_xaxes(rangeslider={"visible": False}, row=i + 2, col=col)
                # fig.layout.annotations[i+1+2].update(text=f"Участок близкий к паттерну. Дистанция : {round(label_df[f'distance_class_{classNo}'][j]),4}")

        w = 1700
        h = 700 + (350 * len(label_df))

        fig.update_layout(
            title=f"График анализа  паттерна {a}, кластера {classNo}, c параметрами : PATTERN_SIZE = {PATTERN_SIZE}, EXTR_WINDOW = {EXTR_WINDOW}, profit_value = {profit_value}, OVERLAP = {OVERLAP}",
            autosize=False,
            width=w,
            height=h,
            showlegend=False,
        )

        fig.show()

        """df = label_df.sort_values(by=[f"distance_class_{classNo}"])
        fig1 = px.line(df, x=f"distance_class_{classNo}", y='future_areas', title=f'Зависимость площади от дистанции, паттерна:  {a}, класса : {classNo}, c параметрами : PATTERN_SIZE = {PATTERN_SIZE}, EXTR_WINDOW = {EXTR_WINDOW}, profit_value = {profit_value}, OVERLAP = {OVERLAP}')
        fig1.show()"""

        df = label_df.sort_values(by=[f"distance_class_{classNo}"])
        df = df.reset_index(drop=True)
        means_grouth = []
        hconf_intervals = []
        lconf_intervals = []
        just_std = []
        for step_name in range(1, EXTR_WINDOW + 1):
            # mean_gr = label_df[f'step_g_{step_name}'].mean()     #if mean_gr >= profit_value:means_grouth.append(mean_gr)
            # means_grouth.append(mean_gr)
            arr_value = label_df[f"step_g_{step_name}"].to_numpy()
            # gen_samol = test_results[f'step_g_{step_name}'].to_numpy()
            mean_ = np.mean(arr_value)
            means_grouth.append(mean_)
            std_ = np.std(arr_value) / 2
            just_std.append(std_)
            # dof = len(arr_value) - 1
            # confidence = 0.95
            # t_crit = np.abs(t.ppf((1 - confidence) / 2, dof))
            # low = mean_ - std_ * t_crit / np.sqrt(len(arr_value))
            # high = mean_ + std_ * t_crit / np.sqrt(len(arr_value))
            # interval=np.percentile(arr_value, [100 * (1 - confidence) / 2, 100 * (1 - (1 - confidence) / 2)])
            # lconf_intervals.append(mean_-low)
            # hconf_intervals.append(mean_+high)
            # lconf_intervals.append(interval[0])
            # hconf_intervals.append(interval[1])

        """stat_df = pd.DataFrame()
        cols = [col for col in df.columns if 'step_g_' in col ]
        step_means = [df[name].mean() for name in cols]
        best_col_index = step_means.index(max(step_means))
        print(cols)
        stat_df=pd.DataFrame(label_df[cols])
        print(stat_df)
        stat_df = stat_df.join(label_df[f"distance_class_{classNo}"])
        for col_name in cols:
            stat_df[col_name] = label_df[col_name].values.tolist()
        stat_df[f"distance_class_{classNo}"] = label_df[f"distance_class_{classNo}"].values.tolist()
        print(stat_df)"""
        cols = [col for col in df.columns if "step_g_" in col]
        best_step = []
        best_grourh = []
        distance = []

        for ind in range(len(df)):
            distance.append(df.loc[ind, f"distance_class_{classNo}"])
            best_gr_prof = np.max(df.loc[ind, cols].to_numpy())
            if best_gr_prof >= profit_value:
                best_grourh.append(best_gr_prof)
            else:
                best_grourh.append(None)
                # best_grourh.append(np.max(df.loc[ind, cols].to_numpy()))   df.loc[0, cols]
            best_step.append((np.argmax(df.loc[ind, cols].to_numpy())))

        heat_data = {
            "Дистанция": distance,
            "Прирост": best_grourh,
            "Количество баров": best_step,
        }
        heat_df = pd.DataFrame(heat_data)

        fig1 = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"Средняя доля прироста цены после распознанного паттерна и доверительные интервалы (всего распознано {n_act} паттернов)",
                "Тепловая карта взаимосвязей доли среденего прироста цены, количество баров от предсказания и дистанции",
            ),
        )

        fig1.add_trace(
            go.Bar(
                x=[step for step in range(1, EXTR_WINDOW + 1)],
                y=means_grouth,
                error_y=dict(
                    type="data",
                    array=just_std,
                    # arrayminus=lconf_intervals,
                    visible=True,
                ),
            ),
            1,
            1,
        )
        fig1.update_xaxes(title_text="Количество баров от предсказания", row=1, col=1)
        fig1.update_yaxes(title_text="Средний прирост в долях", row=1, col=1)
        fig1.add_trace(
            go.Heatmap(
                z=heat_df["Прирост"],
                x=heat_df["Количество баров"],
                y=heat_df["Дистанция"],
                text=heat_df["Прирост"],
                colorbar=dict(title="Доля прироста цены"),
            ),
            1,
            2,
        )
        fig1.update_xaxes(title_text="Количество баров", row=1, col=2)
        fig1.update_yaxes(title_text="Дистанция предскзания", row=1, col=2)

        fig1.update_layout(
            coloraxis_colorbar=dict(title="Доля прироста"),
            title_text=f"Анализ паттерна {a}, кластера {classNo}, c параметрами : PATTERN_SIZE = {PATTERN_SIZE}, EXTR_WINDOW = {EXTR_WINDOW}, profit_value = {profit_value}, OVERLAP = {OVERLAP}",
        )
        fig1.update_annotations(font_size=13)
        fig1.show()

    """if save_best == True:

            if len(label_df)>1:                                  # (y[-1] - y[0]) / (x[-1] - x[0])
                slope =(df['future_areas'].iloc[-1] - df['future_areas'].iloc[0]) / (df[f"distance_class_{classNo}"].iloc[-1] - df[f"distance_class_{classNo}"].iloc[0])
                if slope<0 and label_df["future_areas"].mean() >0:
                    best_patterns_in_class.append(a)

    if len(best_patterns_in_class) == 0:
            print(f'В классе {classNo} рабочих паттернов не обнаруженно')
    else :
            print(f'В классе {classNo} обнаруженно {len(best_patterns_in_class)} рабочих паттернов')

    return best_patterns_in_class"""


def clustered_pattern_train_plot(
    train_df,
    claster_indexes,
    indexes_with_profit,
    best_n_clasters,
    profit_value,
    EXTR_WINDOW,
    PATTERN_SIZE,
    train_dates,
):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=train_dates, y=train_df["Close"], mode="lines", name="CLOSE")
    )
    for n_claster in range(best_n_clasters):
        cluster_index = [
            i for i, j in zip(indexes_with_profit, claster_indexes) if j == n_claster
        ]
        fig.add_trace(
            go.Scatter(
                x=train_dates.iloc[cluster_index],
                y=train_df["Close"].iloc[cluster_index],
                name=f" кластер {n_claster}",
                mode="markers",
                marker=dict(symbol="triangle-up", size=15),
            )
        )
    fig.update_layout(
        title=f"Паттерны распределенные по кластерам на исторических данных. Параметры : profit_value = {profit_value}, EXTR_WINDOW = {EXTR_WINDOW}, PATTERN_SIZE = {PATTERN_SIZE}  ",
        xaxis_title="DATE",
        yaxis_title="CLOSE",
        legend_title="Legend",
    )
    fig.show()
