
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import os
import optuna
from datetime import date
import random
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from models.torch_models import shotSiameseNetwork
from utilits.project_functions import (
    get_triplet_random,
    train_triplet_net,
    get_CLtrain_data,
    uptune_get_stat_after_forward,
)

today = date.today()
n_trials = 100
date_xprmnt = today.strftime("%d_%m_%Y")
source = "source_root"
out_root = "outputs"
source_file_name = "CL_2020_2022.csv"
out_data_root = (
    f"V2_{source_file_name[:-4]}_CL_data_optune_{date_xprmnt}_epoch_{n_trials}"
)
os.mkdir(f"{out_root}/{out_data_root}")
intermedia = pd.DataFrame()
intermedia.to_excel(
    f"{out_root}/{out_data_root}/intermedia_{source_file_name[:-4]}.xlsx"
)


def objective(trial):

    torch.manual_seed(2020)
    torch.cuda.manual_seed(2020)
    np.random.seed(2020)
    random.seed(2020)
    torch.backends.cudnn.deterministic = True
    """""" """""" """""" """""" """"" Общие настройки """ """""" """""" """""" """"""

    start_forward_time = "2021-01-04 00:01:25"
    df = pd.read_csv(f"{source}/{source_file_name}")

    forward_index = df[df["Datetime"] == start_forward_time].index[0]
    print("for_index", forward_index)
    step = 0.1
    profit_value = 1
    get_trade_info = True

    """""" """""" """""" """""" """"" Параметры сети """ """""" """""" """""" """"""
    epochs = 50  # количество эпох
    lr = 0.000009470240447408595  # learnig rate
    embedding_dim = 200  # размер скрытого пространства
    margin = 1  # маржа для лосс функции
    batch_size = 50  # размер батчсайз #150
    distance_function = lambda x, y: 1.0 - F.cosine_similarity(x, y)

    """""" """""" """""" """""" """"" Параметры для оптимизации   """ """ """ """ """ """ """ """ """ ""

    extr_window = trial.suggest_int("extr_window", 50, 400)
    pattern_size = trial.suggest_int("pattern_size", 90, 300)
    overlap = trial.suggest_int("overlap", 0, 45)
    train_window = trial.suggest_categorical("train_window", [10000, 40000, 120000])
    select_dist_window = trial.suggest_categorical(
        "select_dist_window", [10000, 40000, 120000]
    )
    forward_window = trial.suggest_categorical("forward_window", [10000, 40000])

    df_for_split = df[(forward_index - train_window) :]
    df_for_split = df_for_split.reset_index(drop=True)
    n_iters = (len(df_for_split) - int(train_window)) // int(forward_window)

    signals = []
    for n in range(n_iters):

        train_df = df_for_split[:train_window]

        if n == n_iters - 1:
            forward_df = df_for_split[train_window:]
        else:
            forward_df = df_for_split[
                int(train_window) : sum([int(train_window), int(forward_window)])
            ]
        df_for_split = df_for_split[int(forward_window) :]
        df_for_split = df_for_split.reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)
        forward_df = forward_df.reset_index(drop=True)
        train_dates = pd.DataFrame({"Datetime": train_df.Datetime.values})

        forward_dates = pd.DataFrame({"Datetime": forward_df.Datetime.values})
        del (
            train_df["Datetime"],
            forward_df["Datetime"],
        )

        train_x, n_samples_to_train = get_CLtrain_data(
            train_df, profit_value, extr_window, pattern_size, overlap, train_dates,
        )  # получаем данные для создания триплетов

        n_classes = len(train_x)
        train_triplets = get_triplet_random(n_samples_to_train, n_classes, train_x)
        print(
            " Размер данных для обучения:", np.array(train_triplets).shape,
        )

        tensor_A = torch.Tensor(train_triplets[0])  # transform to torch tensor
        tensor_P = torch.Tensor(train_triplets[1])
        tensor_N = torch.Tensor(train_triplets[2])

        my_dataset = TensorDataset(tensor_A, tensor_P, tensor_N)  # create your datset
        my_dataloader = DataLoader(my_dataset, batch_size=batch_size)

        """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
        """""" """""" """""" """""" """"" Train net  """ """""" """""" """""" """"""

        net = shotSiameseNetwork(embedding_dim=embedding_dim).cuda()
        torch.cuda.empty_cache()
        train_triplet_net(lr, epochs, my_dataloader, net, distance_function, margin)

        """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
        """""" """""" """""" """""" """"" Test data prepare  """ """""" """""" """""" """"""

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
            forward_ohlcv[i - pattern_size : i]
            for i in range(len(forward_ohlcv))
            if i - pattern_size >= 0
        ]
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

        net.eval()
        with torch.no_grad():
            for indexI, eval_arr in enumerate(forward_samples):
                buy_anchor = train_x[0][0].reshape(
                    1, forward_samples[0].shape[0], forward_samples[0][0].shape[0], 1,
                )
                sell_anchor = train_x[1][0].reshape(
                    1, forward_samples[0].shape[0], forward_samples[0][0].shape[0], 1,
                )
                eval_arr_r = eval_arr.reshape(
                    1, forward_samples[0].shape[0], forward_samples[0][0].shape[0], 1,
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

                date.append(sampled_forward_dates[indexI]["Datetime"].iat[-1])
                open.append(float(ohlcv_forward_samples[indexI][-1, [0]]))
                high.append(float(ohlcv_forward_samples[indexI][-1, [1]]))
                low.append(float(ohlcv_forward_samples[indexI][-1, [2]]))
                close.append(float(ohlcv_forward_samples[indexI][-1, [3]]))
                volume.append(float(ohlcv_forward_samples[indexI][-1, [4]]))

            forward_result = pd.DataFrame(
                {
                    "Datetime": date,
                    "Open": open,
                    "High": high,
                    "Low": low,
                    "Close": close,
                    "Volume": volume,
                    "Signal": Signal,
                }
            )

            signals.append(forward_result)

    signals_combained = pd.concat(signals, ignore_index=True, sort=False)

    df_stata = uptune_get_stat_after_forward(
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
        trial.number,
        get_trade_info=get_trade_info,
    )
    print(df_stata)
    net_profit = df_stata["Net Profit [$]"].values[0]
    Sharpe_Ratio = df_stata["Sharpe Ratio"].values[0]
    trades = df_stata["# Trades"].values[0]
    trial.set_user_attr("# Trades", trades)
    parameters = trial.params
    parameters.update({"trial": trial.number})
    parameters.update({"values_0": net_profit})
    parameters.update({"values_1": Sharpe_Ratio})
    parameters.update({"# Trades": trades})
    inter = pd.read_excel(
        f"{out_root}/{out_data_root}/intermedia_{source_file_name[:-4]}.xlsx"
    )
    inter = inter.append(parameters, ignore_index=True)
    inter.to_excel(
        f"{out_root}/{out_data_root}/intermedia_{source_file_name[:-4]}.xlsx",
        index=False,
    )

    torch.save(net.state_dict(), f"{out_root}/{out_data_root}/weights.pt")

    return net_profit, Sharpe_Ratio


sampler = optuna.samplers.TPESampler(seed=2020)
study = optuna.create_study(directions=["maximize", "maximize"], sampler=sampler)
study.optimize(objective, n_trials=n_trials)


tune_results = study.trials_dataframe()
"""tune_results = tune_results.rename(
    columns={"values_0": "Net_profit [$]", "values_1": "Sharpe_Ratio"}
)
print(tune_results)"""
tune_results["params_forward_window"] = tune_results["params_forward_window"].astype(
    int
)
tune_results["params_train_window"] = tune_results["params_train_window"].astype(int)
tune_results["params_select_dist_window"] = tune_results[
    "params_select_dist_window"
].astype(int)
df_plot = tune_results[
    [
        "values_0",
        "values_1",
        "user_attrs_# Trades",
        "params_pattern_size",
        "params_extr_window",
        "params_overlap",
        "params_train_window",
        "params_forward_window",
    ]
]

fig = px.parallel_coordinates(
    df_plot,
    color="values_0",
    labels={
        "values_0": "Net Profit ($)",
        "values_1": "Sharpe_Ratio",
        "user_attrs_# Trades": "Trades",
        "params_pattern_size": "pattern_size (bars)",
        "params_extr_window": "extr_window (bars)",
        "params_overlap": "overlap (bars)",
        "params_train_window": "train_window (bars)",
        "params_forward_window": "forward_window (bars)",
    },
    range_color=[df_plot["values_0"].min(), df_plot["values_0"].max()],
    color_continuous_scale=px.colors.sequential.Viridis,
    title=f"V2_hyp_parameters_select_{source_file_name[:-4]}_optune_epoch_{n_trials}",
)

fig.write_html(f"{out_root}/{out_data_root}/hyp_par_sel_{source_file_name[:-4]}.htm")
fig.show()
tune_results.to_excel(
    f"{out_root}/{out_data_root}/V2_hyp_par_sel_{source_file_name[:-4]}.xlsx"
)
