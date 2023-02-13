import pandas as pd
import io
import numpy as np
import pandas as pd
import torch
import optuna
import plotly.express as px
from functools import reduce
import os
from torch.autograd import Variable
import torch.optim as optim
import torchbnn as bnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datetime import date
import random
from sklearn.preprocessing import MinMaxScaler
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from functools import reduce
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchbnn.utils import freeze
from utilits.project_functions import bayes_tune_get_stat_after_forward
from sklearn.tree import DecisionTreeClassifier

today = date.today()
source = "source_root"
out_root = "outputs"
source_file_name = "GC_2020_2022_15min_nq10_extr13"
start_forward_time = "2021-01-04 00:00:00"
date_xprmnt = today.strftime("%d_%m_%Y")
out_data_root = f"deep_b_des_tree_{source_file_name[:-4]}_{date_xprmnt}"
os.mkdir(f"{out_root}/{out_data_root}")
intermedia = pd.DataFrame()
intermedia.to_excel(
    f"{out_root}/{out_data_root}/intermedia_{source_file_name[:-4]}.xlsx"
)
clf = DecisionTreeClassifier()
n_trials = 200
# df = pd.read_csv(f"{source}/{source_file_name}")
# forward_index = df[df["Datetime"] == start_forward_time].index[0]

###################################################################################################
def data_to_binary(train_df, forward_df, look_back):
    binary_train = train_df.iloc[:, 1:].diff()
    binary_train[binary_train < 0] = 0
    binary_train[binary_train > 0] = 1
    target = binary_train["Close"].values[2:]
    binary_train = binary_train[1:-1]
    binary_train["Target"] = target

    train_samples = [
        binary_train[i - look_back : i]
        for i in range(len(binary_train))
        if i - look_back >= 0
    ]
    Train_X = []
    Train_labels = []
    for sample in train_samples:
        Train_X.append(
            sample[["Open", "High", "Low", "Close", "Volume"]].to_numpy().flatten()
        )
        Train_labels.append(sample["Target"].iloc[-1])

    Train_Y = [[1, 0] if i == 0 else [0, 1] for i in Train_labels]

    binary_forward = forward_df.iloc[:, 1:].diff()
    binary_forward[binary_forward < 0] = 0
    binary_forward[binary_forward > 0] = 1
    binary_forward = binary_forward[1:]
    forward_df = forward_df[1:]
    foraward_samples = [
        forward_df[i - look_back : i]
        for i in range(len(forward_df))
        if i - look_back >= 0
    ]
    forward_binary_samples = [
        binary_forward[i - look_back : i]
        for i in range(len(binary_forward))
        if i - look_back >= 0
    ]
    Test_X = []
    Date = []
    Open = []
    High = []
    Low = []
    Close = []
    Volume = []
    for sample in forward_binary_samples:
        Test_X.append(
            sample[["Open", "High", "Low", "Close", "Volume"]].to_numpy().flatten()
        )
    for or_sample in foraward_samples:
        Date.append(or_sample["Datetime"].iloc[-1])
        Open.append(or_sample["Open"].iloc[-1])
        High.append(or_sample["High"].iloc[-1])
        Low.append(or_sample["Low"].iloc[-1])
        Close.append(or_sample["Close"].iloc[-1])
        Volume.append(or_sample["Volume"].iloc[-1])
    Signals = pd.DataFrame(
        {
            "Datetime": Date,
            "Open": Open,
            "High": High,
            "Low": Low,
            "Close": Close,
            "Volume": Volume,
        }
    )

    return np.array(Train_X), np.array(Train_Y), np.array(Test_X), Signals


class DBNataset(Dataset):
    def __init__(self, train_features, train_labels):

        self.x_train = torch.tensor(train_features, dtype=torch.float32)
        self.y_train = torch.tensor(train_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


#####################################################################################################
def objective(trial):
    start_forward_time = "2021-01-04 00:00:00"
    df = pd.read_csv(f"{source}/{source_file_name}")
    forward_index = df[df["Datetime"] == start_forward_time].index[0]
    get_trade_info = True

    """""" """""" """""" """""" """"" Параметры сети """ """""" """""" """""" """"""
    # epochs = 12
    # lr = 0.000009470240447408595
    # embedding_dim = 160
    # margin = 1
    batch_s = 300
    # distance_function = lambda x, y: 1.0 - F.cosine_similarity(x, y)

    """""" """""" """""" """""" """"" Параметры для оптимизации   """ """ """ """ """ """ """ """ """ ""

    lookback_size = trial.suggest_int("lookback_size", 2, 100)
    epochs = trial.suggest_int("epochs", 500, 1000)
    n_hiden = trial.suggest_int("n_hiden", 10, 300, step=10)
    n_hiden_two = trial.suggest_int("n_hiden_two", 10, 300, step=10)
    train_window = trial.suggest_categorical("train_window", [1000, 3000, 5000])
    forward_window = trial.suggest_categorical(
        "forward_window", [705, 1411, 2822, 5644]
    )
    ##############################################################################################
    DBNmodel = nn.Sequential(
        bnn.BayesLinear(
            prior_mu=0,
            prior_sigma=0.1,
            in_features=lookback_size * 5,
            out_features=n_hiden,
        ),
        nn.ReLU(),
        bnn.BayesLinear(
            prior_mu=0, prior_sigma=0.1, in_features=n_hiden, out_features=n_hiden_two
        ),
        nn.ReLU(),
        bnn.BayesLinear(
            prior_mu=0, prior_sigma=0.1, in_features=n_hiden_two, out_features=2
        ),
    )
    ###################################################################################################
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
        Train_X, Train_Y, Forward_X, Signals = data_to_binary(
            train_df, forward_df, lookback_size
        )

        DNB_dataset = DBNataset(
            Train_X[: len(Train_X) // 2], Train_Y[: len(Train_X) // 2]
        )
        DNB_dataloader = DataLoader(DNB_dataset, batch_size=batch_s, shuffle=False)
        cross_entropy_loss = nn.CrossEntropyLoss()
        klloss = bnn.BKLLoss(reduction="mean", last_layer_only=False)
        klweight = 0.01
        optimizer = optim.Adam(DBNmodel.parameters(), lr=0.001)

        for step in range(epochs):
            for _, (data, target) in enumerate(DNB_dataloader):

                models = DBNmodel(data)
                cross_entropy = cross_entropy_loss(models, target)

                kl = klloss(DBNmodel)
                total_cost = cross_entropy + klweight * kl

                if _ % 100 == 0:
                    print("Энтропия", cross_entropy)
                    print("Фианльная лосс", total_cost)

                optimizer.zero_grad()
                total_cost.backward()
                optimizer.step()

        DT_trainX = []
        DBNmodel.eval()
        freeze(DBNmodel)
        with torch.no_grad():
            for arr in Train_X[len(Train_X) // 2 :]:
                arr = torch.from_numpy(arr.astype(np.float32))

                pred = DBNmodel(arr)
                DT_trainX.append(
                    [
                        float(torch.argmax(pred).cpu().detach().numpy()),
                        float(pred[torch.argmax(pred)].cpu().detach().numpy()),
                    ]
                )

        des_lable = [np.argmax(i) for i in Train_Y[len(Train_X) // 2 :]]
        clf = DecisionTreeClassifier()
        clf = clf.fit(np.array(DT_trainX), np.array(des_lable))
        predictions = []
        with torch.no_grad():
            for arr in Forward_X:
                arr = torch.from_numpy(arr.astype(np.float32))
                pred = DBNmodel(arr)
                class_n = clf.predict(
                    np.array(
                        [
                            float(torch.argmax(pred).cpu().detach().numpy()),
                            float(pred[torch.argmax(pred)].cpu().detach().numpy()),
                        ]
                    ).reshape(1, -1)
                )

                predictions.append(int(class_n))
        Signals["Signal"] = predictions
        signals.append(Signals)

    signals_combained = pd.concat(signals, ignore_index=True, sort=False)
    signals_combained.loc[signals_combained["Signal"] == 0, "Signal"] = -1
    df_stata = bayes_tune_get_stat_after_forward(
        signals_combained,
        lookback_size,
        epochs,
        n_hiden,
        n_hiden_two,
        train_window,
        forward_window,
        source_file_name,
        out_root,
        out_data_root,
        trial.number,
        get_trade_info=True,
    )
    net_profit = df_stata["Net Profit [$]"].values[0]
    Sharpe_Ratio = df_stata["Sharpe Ratio"].values[0]
    trades = df_stata["# Trades"].values[0]
    trial.set_user_attr("# Trades", trades)
    parameters = trial.params
    parameters.update({"trial": trial.number})
    parameters.update({"values_0": net_profit})
    parameters.update({"values_1": Sharpe_Ratio})
    inter = pd.read_excel(
        f"{out_root}/{out_data_root}/intermedia_{source_file_name[:-4]}.xlsx"
    )
    inter = inter.append(parameters, ignore_index=True)
    inter.to_excel(
        f"{out_root}/{out_data_root}/intermedia_{source_file_name[:-4]}.xlsx",
        index=False,
    )

    # torch.save(net.state_dict(), f"{out_root}/{out_data_root}/weights.pt")

    return net_profit, Sharpe_Ratio


sampler = optuna.samplers.TPESampler(seed=2020)
study = optuna.create_study(directions=["maximize", "maximize"], sampler=sampler)
study.optimize(objective, n_trials=n_trials)


tune_results = study.trials_dataframe()

tune_results["params_forward_window"] = tune_results["params_forward_window"].astype(
    int
)
tune_results["params_train_window"] = tune_results["params_train_window"].astype(int)
df_plot = tune_results[
    [
        "values_0",
        "values_1",
        "user_attrs_# Trades",
        "params_lookback_size",
        "params_epochs",
        "params_n_hiden",
        "params_n_hiden_two",
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
        "params_lookback_size": "lookback_size (bars)",
        "params_epochs": "epochs",
        "params_n_hiden": "n_hiden",
        "params_n_hiden_two": "n_hiden_two",
        "params_train_window": "train_window (bars)",
        "params_forward_window": "forward_window (bars)",
    },
    range_color=[df_plot["values_0"].min(), df_plot["values_0"].max()],
    color_continuous_scale=px.colors.sequential.Viridis,
    title=f"bayes_parameters_select_{source_file_name[:-4]}_optune_epoch_{n_trials}",
)

fig.write_html(f"{out_root}/{out_data_root}/hyp_par_sel_{source_file_name[:-4]}.htm")
fig.show()
tune_results.to_excel(
    f"{out_root}/{out_data_root}/hyp_par_sel_{source_file_name[:-4]}.xlsx"
)
