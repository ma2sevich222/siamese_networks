import numpy as np
import pandas as pd
import torch
import torchvision.models as models
from torch.nn import PairwiseDistance
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import normalize
from models.torch_models import SiameseNetwork_extend_triplet
from utilits.data_load import data_load
from other_codes.old_project.old_utilits.functions_for_train_nn import (
    get_locals,
    get_patterns,
)
from other_codes.old_project.old_utilits.triplet_func_for_train import (
    get_triplet_random,
)
from other_codes.old_project.old_utilits.triplet_func_for_train import train_triplet_net

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
"""""" """""" """""" """""" """"" Parameters Block """ """""" """""" """""" """"""
from constants import *

epochs = 1
lr = 0.0005
embedding_dim = 20
margin = 1
batch_size = 10
distance_function = PairwiseDistance(p=2, eps=1e-06,)

# seed = 3
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


""" Добавляем предобученную модель из модуля torchvision.models.
Переходим по ссылке https://pytorch.org/vision/stable/models.html,
подставляем нужную модель как в примере :
base_model = models.< название модели из документации >(pretrained=True)
 На данный момент доступны: resnet18 , resnet50 , shufflenet_v2_x1_0 , mnasnet1_0 """

base_model = models.resnet18()
model_name = base_model.__class__.__name__

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
"""""" """""" """""" """""" """"" Main Block """ """""" """""" """""" """""" """"""
indices = [
    i for i, x in enumerate(FILENAME) if x == "_"
]  # находим индексы вхождения '_'
ticker = FILENAME[: indices[0]]

"""Загрузка и подготовка данных"""
Train_df, Eval_df, Eval_dates_str = data_load(SOURCE_ROOT, FILENAME)

Min_train_locals, Max_train__locals = get_locals(Train_df, EXTR_WINDOW)

buy_patern, sell_patern = get_patterns(
    Train_df.to_numpy(),
    Min_train_locals["index"].values.tolist(),
    Max_train__locals["index"].values.tolist(),
    PATTERN_SIZE,
)

Min_train_locals, Max_train__locals = get_locals(Train_df, EXTR_WINDOW)

buy_patern, sell_patern = get_patterns(
    Train_df.to_numpy(),
    Min_train_locals["index"].values.tolist(),
    Max_train__locals["index"].values.tolist(),
    PATTERN_SIZE,
)

buy_reshaped = buy_patern.reshape(buy_patern.shape[0], -1)
np.savetxt(
    f"{DESTINATION_ROOT}/buy_patterns_extr_window{EXTR_WINDOW}"
    f"_pattern_size{PATTERN_SIZE}.csv",
    buy_reshaped,
)

sell_reshaped = sell_patern.reshape(sell_patern.shape[0], -1)
np.savetxt(
    f"{DESTINATION_ROOT}/sell_patterns_extr_window{EXTR_WINDOW}"
    f"_pattern_size{PATTERN_SIZE}.csv",
    sell_reshaped,
)

buy_patern = buy_patern.reshape(
    -1, buy_patern[0].shape[0], buy_patern[0][0].shape[0], 1
)
sell_patern = buy_patern.reshape(
    -1, sell_patern[0].shape[0], sell_patern[0][0].shape[0], 1
)

train_x = np.array([np.array(buy_patern), np.array(sell_patern)])

train_triplets = np.array(
    get_triplet_random(
        min(buy_patern.shape[0], sell_patern.shape[0]), num_classes, train_x
    )
)

tensor_A = torch.Tensor(train_triplets[0][0])  # transform to torch tensor
tensor_P = torch.Tensor(train_triplets[0][1])
tensor_N = torch.Tensor(train_triplets[0][2])

my_dataset = TensorDataset(tensor_A, tensor_P, tensor_N)  # create your datset
my_dataloader = DataLoader(my_dataset, batch_size=batch_size)

net = SiameseNetwork_extend_triplet(base_model, embedding_dim=embedding_dim).cuda()

train_triplet_net(lr, epochs, my_dataloader, net, distance_function)

"""Тест модели"""

eval_array = Eval_df.to_numpy()
eval_samples = [
    eval_array[i - PATTERN_SIZE : i]
    for i in range(len(eval_array))
    if i - PATTERN_SIZE >= 0
]
eval_normlzd = [normalize(i, axis=0, norm="max") for i in eval_samples]
eval_normlzd = np.array(eval_normlzd).reshape(
    -1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1
)

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
    for indexI, eval_arr in enumerate(tqdm(eval_samples)):

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
            dist = distance_function
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
