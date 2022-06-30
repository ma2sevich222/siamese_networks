import numpy as np
import pandas as pd
import plotly.express as px
import torch
from datetime import date
import os
import random
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from models.torch_models import shotSiameseNetwork
from utilits.project_functions import (
    get_train_data,
    get_triplet_random,
    train_triplet_net,
    get_stat_after_forward,
)

torch.manual_seed(2020)
torch.cuda.manual_seed(2020)
np.random.seed(2020)
random.seed(2020)
torch.backends.cudnn.deterministic = True
"""""" """""" """""" """""" """"" Parameters Block """ """""" """""" """""" """"""

today = date.today()
date_xprmnt = today.strftime("%d_%m_%Y")
source = "source_root"
out_root = "outputs"
source_file_name = "GC_2020_2022_30min.csv"
start_forward_time = "2021-03-24 23:30:00"
get_trade_info = True  # True если хотим сохранить сигналы, статистику и график торгов
# run_mode = "best"
profit_value = 0.015
step = 0.1
pattern_size_list = [40]
extr_window_list = [240]
overlap_list = [0]
train_window_list = [10000]
forward_window_list = [5000]
select_dist_window = 0

"""""" """""" """""" """""" """"" Net Parameters Block """ """""" """""" """""" """"""
epochs = 20  # количество эпох
lr = 0.000009470240447408595  # learnig rate
embedding_dim = 160  # размер скрытого пространства
margin = 1  # маржа для лосс функции
batch_size = 20  # размер батчсайз
distance_function = lambda x, y: 1.0 - F.cosine_similarity(x, y)
final_stats_list = []

out_data_root = f"V2_{source_file_name[:-4]}_{date_xprmnt}_forward"
os.mkdir(f"{out_root}/{out_data_root}")

df = pd.read_csv(f"{source}/{source_file_name}")
forward_index = df[df["Datetime"] == start_forward_time].index[0]
print(forward_index)


for train_window in tqdm(train_window_list):

    for forward_window in forward_window_list:
        for pattern_size in pattern_size_list:
            for extr_window in extr_window_list:
                for overlap in overlap_list:

                    df_for_split = df[(df.index >= forward_index - train_window)]
                    df_for_split = df_for_split.reset_index(drop=True)
                    n_iters = (len(df_for_split) - train_window) // forward_window

                    if n_iters < 1:
                        n_iters = 1
                    signals = []
                    for n in range(n_iters):
                        train_df = df_for_split.iloc[:train_window]

                        forward_df = df_for_split.iloc[
                            train_window : sum([train_window, forward_window])
                        ]
                        df_for_split = df_for_split.iloc[forward_window:]
                        df_for_split = df_for_split.reset_index(drop=True)
                        train_df = train_df.reset_index(drop=True)
                        forward_df = forward_df.reset_index(drop=True)
                        train_dates = pd.DataFrame(
                            {"Datetime": train_df.Datetime.values}
                        )

                        forward_dates = pd.DataFrame(
                            {"Datetime": forward_df.Datetime.values}
                        )
                        del (
                            train_df["Datetime"],
                            forward_df["Datetime"],
                        )
                        train_x, n_samples_to_train = get_train_data(
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
                        my_dataloader = DataLoader(my_dataset, batch_size=batch_size)

                        """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
                        """""" """""" """""" """""" """"" Train net  """ """""" """""" """""" """"""

                        net = shotSiameseNetwork(embedding_dim=embedding_dim).cuda()
                        torch.cuda.empty_cache()
                        train_triplet_net(
                            lr, epochs, my_dataloader, net, distance_function, margin
                        )

                        """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
                        """""" """""" """""" """""" """"" Forward data prepare  """ """""" """""" """""" """"""
                        scaler = StandardScaler()
                        eval_array = forward_df.to_numpy()
                        eval_samples = [
                            eval_array[i - pattern_size : i]
                            for i in range(len(eval_array))
                            if i - pattern_size >= 0
                        ]
                        eval_normlzd = [scaler.fit_transform(i) for i in eval_samples]
                        eval_normlzd = np.array(eval_normlzd).reshape(
                            -1,
                            eval_samples[0].shape[0],
                            eval_samples[0][0].shape[0],
                            1,
                        )
                        sampled_forward_dates = [
                            forward_dates[i - pattern_size : i]
                            for i in range(len(forward_dates))
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
                        Signal = []
                        train_data_shape = []

                        net.eval()
                        with torch.no_grad():
                            for indexI, eval_arr in enumerate(eval_normlzd):
                                buy_anchor = train_x[0][0].reshape(
                                    1,
                                    eval_samples[0].shape[0],
                                    eval_samples[0][0].shape[0],
                                    1,
                                )
                                sell_anchor = train_x[1][0].reshape(
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

                                buy_anchor = torch.Tensor(buy_anchor)
                                sell_anchor = torch.Tensor(sell_anchor)
                                eval_arr_r = torch.Tensor(eval_arr_r)
                                output1, output2, output3 = net(
                                    buy_anchor.cuda().permute(0, 3, 1, 2),
                                    eval_arr_r.cuda().permute(0, 3, 1, 2),
                                    sell_anchor.cuda().permute(0, 3, 1, 2),
                                )
                                buy_pred = distance_function(output1, output2)
                                buy_pred = float(buy_pred.to("cpu").numpy())
                                output1, output2, output3 = net(
                                    sell_anchor.cuda().permute(0, 3, 1, 2),
                                    eval_arr_r.cuda().permute(0, 3, 1, 2),
                                    buy_anchor.cuda().permute(0, 3, 1, 2),
                                )

                                sell_pred = distance_function(output1, output2)
                                sell_pred = float(sell_pred.to("cpu").numpy())

                                if buy_pred < sell_pred:
                                    Signal.append(1)
                                if buy_pred > sell_pred:
                                    Signal.append(-1)

                                date.append(
                                    sampled_forward_dates[indexI]["Datetime"].iat[-1]
                                )
                                open.append(float(eval_samples[indexI][-1, [0]]))
                                high.append(float(eval_samples[indexI][-1, [1]]))
                                low.append(float(eval_samples[indexI][-1, [2]]))
                                close.append(float(eval_samples[indexI][-1, [3]]))
                                volume.append(float(eval_samples[indexI][-1, [4]]))
                                train_data_shape.append(float(train_df.shape[0]))

                        forward_result = pd.DataFrame(
                            {
                                "Datetime": date,
                                "Open": open,
                                "High": high,
                                "Low": low,
                                "Close": close,
                                "Volume": volume,
                                "Signal": Signal,
                                "Train_shape": train_data_shape,
                            }
                        )

                        signals.append(forward_result)

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
