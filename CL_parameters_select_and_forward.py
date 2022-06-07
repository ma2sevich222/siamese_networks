import numpy as np
import pandas as pd
import plotly.express as px
import torch
import os
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange, tqdm
from models.torch_models import shotSiameseNetwork
from utilits.project_functions import (
    get_train_data,
    get_triplet_random,
    train_triplet_net,
    get_CLtrain_data,
    get_stat_after_forward,
    find_best_dist_stbl,
    get_signals,
)

"""""" """""" """""" """""" """"" Parameters Block """ """""" """""" """""" """"""
source = "source_root"
out_root = "outputs"
source_file_name = "GC_2020_2022_60min.csv"
start_forward_time = "2021-03-25 00:00:00"
get_trade_info = False  # True если хотим сохранить сигналы, статистику и график торгов
profit_value = 0.003
step = 0.1
pattern_size_list = [10, 20, 30, 40, 50, 60, 70]
extr_window_list = [50, 120, 160, 180]
overlap_list = [10, 15, 20]
train_window_list = [1500, 2500]
select_distance_list = [1500, 2500]
forward_window_list = [705, 1411, 2822, 5644]


"""""" """""" """""" """""" """"" Net Parameters Block """ """""" """""" """""" """"""
epochs = 12  # количество эпох
lr = 0.000009470240447408595  # learnig rate
embedding_dim = 160  # размер скрытого пространства
margin = 20  # маржа для лосс функции
batch_size = 150  # размер батчсайз
distance_function = lambda x, y: 1.0 - F.cosine_similarity(x, y)
final_stats_list = []
out_data_root = f"{source_file_name[:-4]}_select_and_forward"
os.mkdir(f"{out_root}/{out_data_root}")

df = pd.read_csv(f"{source}/{source_file_name}")
forward_index = df[df["Datetime"] == start_forward_time].index[0]
print(forward_index)


for train_window in tqdm(train_window_list):
    for select_dist_window in select_distance_list:
        for forward_window in forward_window_list:
            for pattern_size in pattern_size_list:
                for extr_window in extr_window_list:
                    for overlap in overlap_list:

                        df_for_split = df[
                            (
                                df.index
                                >= forward_index - train_window - select_dist_window
                            )
                        ].copy()
                        df_for_split = df_for_split.reset_index(drop=True)
                        n_iters = (
                            len(df_for_split)
                            - sum([train_window, select_dist_window, forward_window])
                        ) // forward_window

                        if n_iters < 1:
                            n_iters = 1
                        signals = []
                        for n in range(n_iters):
                            train_df = df_for_split.iloc[:train_window]
                            test_df = df_for_split.iloc[
                                train_window : sum([train_window, select_dist_window])
                            ]
                            forward_df = df_for_split.iloc[
                                sum([train_window, select_dist_window]) : sum(
                                    [train_window, select_dist_window, forward_window]
                                )
                            ]
                            df_for_split = df_for_split.iloc[forward_window:]
                            df_for_split = df_for_split.reset_index(drop=True)
                            train_df = train_df.reset_index(drop=True)
                            test_df = test_df.reset_index(drop=True)
                            forward_df = forward_df.reset_index(drop=True)
                            train_dates = pd.DataFrame(
                                {"Datetime": train_df.Datetime.values}
                            )
                            test_dates = pd.DataFrame(
                                {"Datetime": test_df.Datetime.values}
                            )
                            forward_dates = pd.DataFrame(
                                {"Datetime": forward_df.Datetime.values}
                            )
                            del (
                                train_df["Datetime"],
                                test_df["Datetime"],
                                forward_df["Datetime"],
                            )
                            train_x, n_samples_to_train = get_CLtrain_data(
                                train_df,
                                profit_value,
                                extr_window,
                                pattern_size,
                                overlap,
                                train_dates,
                            )  # получаем данные для создания триплетов

                            n_classes = len(train_x)
                            train_triplets = get_triplet_random(
                                n_samples_to_train, n_classes, train_x
                            )
                            print(
                                " Размер данных для обучения:",
                                np.array(train_triplets).shape,
                            )

                            tensor_A = torch.Tensor(
                                train_triplets[0]
                            )  # transform to torch tensor
                            tensor_P = torch.Tensor(train_triplets[1])
                            tensor_N = torch.Tensor(train_triplets[2])

                            my_dataset = TensorDataset(
                                tensor_A, tensor_P, tensor_N
                            )  # create your datset
                            my_dataloader = DataLoader(
                                my_dataset, batch_size=batch_size
                            )

                            """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
                            """""" """""" """""" """""" """"" Train net  """ """""" """""" """""" """"""

                            net = shotSiameseNetwork(embedding_dim=embedding_dim).cuda()
                            train_triplet_net(
                                lr, epochs, my_dataloader, net, distance_function
                            )

                            """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
                            """""" """""" """""" """""" """"" Test data prepare  """ """""" """""" """""" """"""
                            eval_array = test_df[
                                [
                                    "DiffEMA",
                                    "SmoothDiffEMA",
                                    "VolatilityTunnel",
                                    "BuyIntense",
                                    "SellIntense",
                                ]
                            ].to_numpy()
                            eval_samples = [
                                eval_array[i - pattern_size : i]
                                for i in range(len(eval_array))
                                if i - pattern_size >= 0
                            ]
                            eval_ohlcv = test_df[
                                ["Open", "High", "Low", "Close", "Volume"]
                            ].to_numpy()
                            ohlcv_samples = [
                                eval_ohlcv[i - pattern_size : i]
                                for i in range(len(eval_ohlcv))
                                if i - pattern_size >= 0
                            ]

                            """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
                            """""" """""" """""" """""" """"" Test model  """ """""" """""" """""" """"""

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
                                for indexI, eval_arr in enumerate(eval_samples):
                                    anchor = train_x[0][0].reshape(
                                        1,
                                        eval_samples[0].shape[0],
                                        eval_samples[0][0].shape[0],
                                        1,
                                    )
                                    eval_arr_r = eval_arr.reshape(
                                        1,
                                        eval_samples[0].shape[0],
                                        eval_samples[0][0].shape[0],
                                        1,
                                    )
                                    anchor = torch.Tensor(anchor)
                                    eval_arr_r = torch.Tensor(eval_arr_r)
                                    output1, output2, output3 = net(
                                        anchor.cuda().permute(0, 3, 1, 2),
                                        eval_arr_r.cuda().permute(0, 3, 1, 2),
                                        eval_arr_r.cuda().permute(0, 3, 1, 2),
                                    )
                                    net_pred = distance_function(output1, output3)
                                    buy_pred.append(float(net_pred.to("cpu").numpy()))

                                    date.append(
                                        test_dates.Datetime[indexI + (pattern_size - 1)]
                                    )
                                    open.append(float(ohlcv_samples[indexI][-1, [0]]))
                                    high.append(float(ohlcv_samples[indexI][-1, [1]]))
                                    low.append(float(ohlcv_samples[indexI][-1, [2]]))
                                    close.append(float(ohlcv_samples[indexI][-1, [3]]))
                                    volume.append(float(ohlcv_samples[indexI][-1, [4]]))
                                    train_data_shape.append(float(train_df.shape[0]))

                            test_result = pd.DataFrame(
                                {
                                    "Datetime": date,
                                    "Open": open,
                                    "High": high,
                                    "Low": low,
                                    "Close": close,
                                    "Volume": volume,
                                    "Distance": buy_pred,
                                    "Train_shape": train_data_shape,
                                }
                            )

                            buy_before, sell_after = find_best_dist_stbl(
                                test_result, step
                            )

                            print(f"BUY BEFORE = {buy_before}")
                            print(f"SELL AFTER = {sell_after}")

                            """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
                            """""" """""" """""" """""" """"" Forward data prepare  """ """""" """""" """""" """"""
                            forward_array = forward_df[
                                [
                                    "DiffEMA",
                                    "SmoothDiffEMA",
                                    "VolatilityTunnel",
                                    "BuyIntense",
                                    "SellIntense",
                                ]
                            ].to_numpy()
                            forward_samples = [
                                forward_array[i - pattern_size : i]
                                for i in range(len(forward_array))
                                if i - pattern_size >= 0
                            ]
                            forward_ohlcv = forward_df[
                                ["Open", "High", "Low", "Close", "Volume"]
                            ].to_numpy()
                            ohlcv_forward_samples = [
                                eval_ohlcv[i - pattern_size : i]
                                for i in range(len(forward_ohlcv))
                                if i - pattern_size >= 0
                            ]

                            """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
                            """""" """""" """""" """""" """"" Forward model  """ """""" """""" """""" """"""

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
                                for indexI, eval_arr in enumerate(forward_samples):
                                    anchor = train_x[0][0].reshape(
                                        1,
                                        forward_samples[0].shape[0],
                                        forward_samples[0][0].shape[0],
                                        1,
                                    )
                                    eval_arr_r = eval_arr.reshape(
                                        1,
                                        forward_samples[0].shape[0],
                                        forward_samples[0][0].shape[0],
                                        1,
                                    )
                                    anchor = torch.Tensor(anchor)
                                    eval_arr_r = torch.Tensor(eval_arr_r)
                                    output1, output2, output3 = net(
                                        anchor.cuda().permute(0, 3, 1, 2),
                                        eval_arr_r.cuda().permute(0, 3, 1, 2),
                                        eval_arr_r.cuda().permute(0, 3, 1, 2),
                                    )
                                    net_pred = distance_function(output1, output3)
                                    buy_pred.append(float(net_pred.to("cpu").numpy()))

                                    date.append(
                                        forward_dates.Datetime[
                                            indexI + (pattern_size - 1)
                                        ]
                                    )
                                    open.append(
                                        float(ohlcv_forward_samples[indexI][-1, [0]])
                                    )
                                    high.append(
                                        float(ohlcv_forward_samples[indexI][-1, [1]])
                                    )
                                    low.append(
                                        float(ohlcv_forward_samples[indexI][-1, [2]])
                                    )
                                    close.append(
                                        float(ohlcv_forward_samples[indexI][-1, [3]])
                                    )
                                    volume.append(
                                        float(ohlcv_forward_samples[indexI][-1, [4]])
                                    )
                                    train_data_shape.append(float(train_df.shape[0]))

                            forward_result = pd.DataFrame(
                                {
                                    "Datetime": date,
                                    "Open": open,
                                    "High": high,
                                    "Low": low,
                                    "Close": close,
                                    "Volume": volume,
                                    "Distance": buy_pred,
                                    "Train_shape": train_data_shape,
                                }
                            )

                            signal = get_signals(forward_result, buy_before, sell_after)
                            signals.append(signal)

                        signals_combained = pd.concat(
                            signals, ignore_index=True, sort=False
                        )

                        df_stata = get_stat_after_forward(
                            signals_combained,
                            pattern_size,
                            extr_window,
                            overlap,
                            train_window,
                            select_dist_window,
                            forward_window,
                            profit_value,
                            source_file_name,
                            out_root,
                            out_data_root,
                            get_trade_info=get_trade_info,
                        )
                        final_stats_list.append(df_stata)

                        intermedia = pd.concat(
                            final_stats_list, ignore_index=True, sort=False
                        )

                        intermedia.to_excel(
                            f"{out_root}/{out_data_root}/intermedia{source_file_name[:-4]}_select_and_forward.xlsx"
                        )

final_stats_df = pd.concat(final_stats_list, ignore_index=True, sort=False)
final_stats_df.sort_values(by="Net Profit [$]", ascending=False).to_excel(
    f"{out_root}/{out_data_root}/{source_file_name[:-4]}_select_and_forward.xlsx"
)

# print(df_stats)

df_plot = final_stats_df[
    [
        "Net Profit [$]",
        "pattern_size",
        "extr_window",
        "overlap",
        "train_window",
        "select_dist_window",
        "forward_window",
    ]
]
fig = px.parallel_coordinates(
    df_plot,
    color="Net Profit [$]",
    labels={
        "Net Profit [$]": "Net Profit ($)",
        "pattern_size": "pattern_size (bars)",
        "extr_window": "extr_window (bars)",
        "overlap": "overlap (bars)",
        "train_window": "train_window (bars)",
        "select_dist_window": "select_dist_window (bars)",
        "forward_window": "forward_window (bars)",
    },
    range_color=[df_plot["Net Profit [$]"].min(), df_plot["Net Profit [$]"].max()],
    color_continuous_scale=px.colors.sequential.Viridis,
    title=f"hyper parameters select {source_file_name[:-4]}",
)

fig.write_html(
    f"{out_root}/{out_data_root}/select_and_forward{source_file_name[:-4]}.html"
)  # сохраняем в файл
fig.show()
