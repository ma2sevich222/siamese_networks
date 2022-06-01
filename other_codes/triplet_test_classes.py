# import numpy as np
# import pandas as pd
import plotly.io as pio
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
from torch.nn import PairwiseDistance
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from models.torch_models import SiameseNetwork_extend_triplet
from other_codes.old_project.old_utilits.triplet_func_for_train import (
    get_triplet_random,
)
from other_codes.old_project.old_utilits.triplet_func_for_train import train_triplet_net

from utilits.data_transforms import patterns_to_df
from utilits.visualisation_functios import calculate_cos_dist
from utilits.data_load import data_load

from constants import *

epochs = 100
lr = 0.0005
embedding_dim = 20
margin = 1
batch_size = 10

base_model = models.resnet18()
model_name = base_model.__class__.__name__

pio.renderers.default = "browser"

Train_df, Eval_df, Eval_dates_str = data_load(SOURCE_ROOT, FILENAME)
buy_patterns_file_name = "buy_patterns_extr_window60_pattern_size20.csv"
sell_patterns_file_name = "sell_patterns_extr_window60_pattern_size20.csv"
column_list = Eval_df.columns.to_list()


buy_loader = np.loadtxt(f"{DESTINATION_ROOT}/{buy_patterns_file_name}")
sell_loader = np.loadtxt(f"{DESTINATION_ROOT}/{sell_patterns_file_name}")

buy_patterns = buy_loader.reshape(-1, PATTERN_SIZE, len(Eval_df.columns.to_list()))
sell_patterns = sell_loader.reshape(-1, PATTERN_SIZE, len(Eval_df.columns.to_list()))

"""patterns_heatmap(buy_patterns)
patterns_heatmap(sell_patterns)"""

buy_neighbor_patterns51 = calculate_cos_dist(buy_patterns, 51)
buy_neighbor_patterns52 = calculate_cos_dist(buy_patterns, 52)
sell_neighbor_patterns49 = calculate_cos_dist(sell_patterns, 49)
sell_neighbor_patterns40 = calculate_cos_dist(sell_patterns, 40)
buy_patern = np.array(
    [
        buy_patterns[list(buy_neighbor_patterns51.keys())[0]],
        buy_patterns[list(buy_neighbor_patterns52.keys())[0]],
    ]
)

buy_patterns_to_df = patterns_to_df(buy_patterns, column_list)
sell_patterns_to_df = patterns_to_df(sell_patterns, column_list)

"""plot_nearlist_patterns(buy_patterns_to_df, buy_neighbor_patterns51)
plot_nearlist_patterns(buy_patterns_to_df, buy_neighbor_patterns52)

plot_nearlist_patterns(sell_patterns_to_df, sell_neighbor_patterns49)
plot_nearlist_patterns(sell_patterns_to_df, sell_neighbor_patterns40)"""

buy_class51 = np.array(
    [buy_patterns[i] for i in list(buy_neighbor_patterns51.keys())]
).reshape(-1, PATTERN_SIZE, len(Eval_df.columns.to_list()), 1)
buy_class52 = np.array(
    [buy_patterns[i] for i in list(buy_neighbor_patterns52.keys())]
).reshape(-1, PATTERN_SIZE, len(Eval_df.columns.to_list()), 1)
sell_class49 = np.array(
    [sell_patterns[i] for i in list(sell_neighbor_patterns49.keys())]
).reshape(-1, PATTERN_SIZE, len(Eval_df.columns.to_list()), 1)
sell_class40 = np.array(
    [sell_patterns[i] for i in list(sell_neighbor_patterns40.keys())]
).reshape(-1, PATTERN_SIZE, len(Eval_df.columns.to_list()), 1)


train_x = np.array([buy_class51, buy_class52, sell_class49, sell_class40])

train_triplets = np.array(get_triplet_random(120, 4, train_x))

tensor_A = torch.Tensor(train_triplets[0][0])  # transform to torch tensor
tensor_P = torch.Tensor(train_triplets[0][1])
tensor_N = torch.Tensor(train_triplets[0][2])

my_dataset = TensorDataset(tensor_A, tensor_P, tensor_N)  # create your datset
my_dataloader = DataLoader(my_dataset, batch_size=batch_size)

net = SiameseNetwork_extend_triplet(base_model, embedding_dim=embedding_dim).cuda()

train_triplet_net(lr, epochs, my_dataloader, net)

"""Тест модели"""

eval_array = Eval_df.to_numpy()
eval_samples = [
    eval_array[i - PATTERN_SIZE : i]
    for i in range(len(eval_array))
    if i - PATTERN_SIZE >= 0
]
# eval_normlzd = [normalize(i, axis=0, norm='max') for i in eval_samples]
# eval_normlzd = np.array(eval_normlzd).reshape(-1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1)

Min_prediction_pattern_name = []
date = []
open = []
high = []
low = []
close = []
volume = []
distance = []
signal = []  # лэйбл
k = 0

net.eval()
with torch.no_grad():
    for indexI, eval_arr in enumerate(tqdm(eval_samples[:1000])):

        buy_predictions = []
        for buy in buy_patern:
            buy = buy.reshape(
                1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1
            )
            eval_arr = eval_arr.reshape(
                1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1
            )
            buy = torch.Tensor(buy)
            eval_arr = torch.Tensor(eval_arr)
            output1, output2, output3 = net(
                buy.cuda().permute(0, 3, 1, 2),
                buy.cuda().permute(0, 3, 1, 2),
                eval_arr.cuda().permute(0, 3, 1, 2),
            )
            dist = PairwiseDistance()
            buy_pred = dist(output1, output3)
            buy_predictions.append(buy_pred)

        date.append(Eval_dates_str[indexI + (PATTERN_SIZE - 1)])
        open.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [0]]))
        high.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [1]]))
        low.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [2]]))
        close.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [3]]))
        volume.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [4]]))
        Min_prediction_pattern_name.append(buy_predictions.index(min(buy_predictions)))

        buy_pred = min(buy_predictions)
        distance.append(float(buy_pred))

        if buy_pred <= 0.5:
            signal.append(1)
        else:
            signal.append(0)

Predictions = pd.DataFrame(
    list(
        zip(
            date,
            open,
            high,
            low,
            close,
            volume,
            signal,
            Min_prediction_pattern_name,
            distance,
        )
    ),
    columns=[
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "signal",
        "pattern",
        "distance",
    ],
)

Predictions.to_csv(
    f"{DESTINATION_ROOT}/test_results_extr_window{EXTR_WINDOW}"
    f"_pattern_size{PATTERN_SIZE}_{model_name}.csv"
)
