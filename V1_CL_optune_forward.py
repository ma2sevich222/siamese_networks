
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
    find_best_dist_stbl,
    get_signals,
    uptune_get_stat_after_forward,
)

today = date.today()
n_trials = 100
date_xprmnt = today.strftime("%d_%m_%Y")
source = "source_root"
out_root = "outputs"
source_file_name = "CL_2020_2022.csv"
out_data_root = (
    f"V1_{source_file_name[:-4]}_CL_data_optune_{date_xprmnt}_epoch_{n_trials}"
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
    step = 0.1
    profit_value = 1
    get_trade_info = True
    df_plot = df[forward_index:]
    import plotly.express as px

    fig = px.line(
        df,
        x="Datetime",
        y="Close",
        title="Цена закрытия на участке форвардного анализа",
    )
    fig.show()
    exit()

    """""" """""" """""" """""" """"" Параметры сети """ """""" """""" """""" """"""
    epochs = 50  # количество эпох
    lr = 0.000009470240447408595  # learnig rate
    embedding_dim = 200  # размер скрытого пространства
    margin = 1  # маржа для лосс функции
    batch_size = 50  # размер батчсайз #150
    distance_function = lambda x, y: 1.0 - F.cosine_similarity(x, y)

    """""" """""" """""" """""" """"" Параметры для оптимизации   """ """ """ """ """ """ """ """ """ ""

    extr_window = trial.suggest_int("extr_window", 250, 500)
    pattern_size = trial.suggest_int("pattern_size", 250, 450)
    overlap = trial.suggest_int("overlap", 0, 30)
    train_window = trial.suggest_categorical("train_window", [10000, 20000])
    select_dist_window = trial.suggest_categorical("select_dist_window", [10000, 40000])
    forward_window = trial.suggest_categorical("forward_window", [10000, 40000])

    df_for_split = df[forward_index - int(train_window) - int(select_dist_window) :]
    df_for_split = df_for_split.reset_index(drop=True)
    signals = []
    n_iters = (len(df_for_split) - int(train_window) - int(select_dist_window)) // int(
        forward_window
    )
    for n in range(n_iters):

        train_df = df_for_split[: int(train_window)]
        test_df = df_for_split[
            int(train_window) : sum([int(train_window), int(select_dist_window)])
        ]
        if n == n_iters - 1:
            forward_df = df_for_split[sum([train_window, select_dist_window]) :]
        else:
            forward_df = df_for_split[
                sum([train_window, select_dist_window]) : sum(
                    [train_window, select_dist_window, int(forward_window)]
                )
            ]
        df_for_split = df_for_split[int(forward_window) :]
        df_for_split = df_for_split.reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        forward_df = forward_df.reset_index(drop=True)
        train_dates = pd.DataFrame({"Datetime": train_df["Datetime"].values})
        test_dates = pd.DataFrame({"Datetime": test_df["Datetime"].values})
        forward_dates = pd.DataFrame({"Datetime": forward_df.Datetime.values})
        del (
            train_df["Datetime"],
            test_df["Datetime"],
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
        eval_ohlcv = test_df[["Open", "High", "Low", "Close", "Volume"]].to_numpy()

        ohlcv_samples = [
            eval_ohlcv[i - pattern_size : i]
            for i in range(len(eval_ohlcv))
            if i - pattern_size >= 0
        ]
        sampled_test_dates = [
            test_dates[i - pattern_size : i]
            for i in range(len(test_dates))
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
                    1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1,
                )
                eval_arr_r = eval_arr.reshape(
                    1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1,
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

                date.append(sampled_test_dates[indexI]["Datetime"].iat[-1])
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

        buy_before, sell_after = find_best_dist_stbl(test_result, step)

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
        buy_pred = []
        train_data_shape = []

        net.eval()
        with torch.no_grad():
            for indexI, eval_arr in enumerate(forward_samples):
                anchor = train_x[0][0].reshape(
                    1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1,
                )
                eval_arr_r = eval_arr.reshape(
                    1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1,
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

                date.append(sampled_forward_dates[indexI]["Datetime"].iat[-1])
                open.append(float(ohlcv_forward_samples[indexI][-1, [0]]))
                high.append(float(ohlcv_forward_samples[indexI][-1, [1]]))
                low.append(float(ohlcv_forward_samples[indexI][-1, [2]]))
                close.append(float(ohlcv_forward_samples[indexI][-1, [3]]))
                volume.append(float(ohlcv_forward_samples[indexI][-1, [4]]))
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
        "params_select_dist_window",
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
        "params_select_dist_window": "select_dist_window (bars)",
        "params_forward_window": "forward_window (bars)",
    },
    range_color=[df_plot["values_0"].min(), df_plot["values_0"].max()],
    color_continuous_scale=px.colors.sequential.Viridis,
    title=f"V1_hyp_parameters_select_{source_file_name[:-4]}_optune_epoch_{n_trials}",
)

fig.write_html(f"{out_root}/{out_data_root}/hyp_par_sel_{source_file_name[:-4]}.htm")
fig.show()
tune_results.to_excel(
    f"{out_root}/{out_data_root}/V1_hyp_par_sel_{source_file_name[:-4]}.xlsx"
)
