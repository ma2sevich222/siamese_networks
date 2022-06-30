import numpy as np
import pandas as pd
import torch
import torchvision.models as models
from sklearn.preprocessing import normalize
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from utilits.visualisation_functios import named_patterns_heatmap
from models.torch_models import SiameseNetwork_extend_triplet
from other_codes.not_used.data_load import data_load
from utilits.data_transforms import patterns_to_df
from other_codes.old_project.old_utilits.functions_for_train_nn import (
    get_locals,
    get_patterns,
)
from other_codes.old_project.old_utilits.triplet_func_for_train import (
    get_patterns_index_classes,
    get_class_and_neighbours,
    get_triplet_random,
)
from other_codes.old_project.old_utilits.triplet_func_for_train import train_triplet_net
from utilits.visualisation_functios import triplet_pattern_class_plot

"""""" """""" """""" """""" """""" """"""
"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
"""""" """""" """""" """""" """"" Parameters Block """ """""" """""" """""" """"""
from constants import *

epochs = 1  # количество эпох
lr = 0.0005  # learnig rate
embedding_dim = 20  # размер скрытого пространства
margin = 1  # маржа
batch_size = 100  # размер батчсайз
n_classes = 2  # Количество классов которое хотим сгенерировать для buy и sell
n_samples_in_class = 5  # Количество
n_samples_to_train = 20000  # Количество триплетов для тренировки
tresh_hold = (
    10  # граница предсказаний ниже которой предсказания будут отображаться на графике
)
distance_function = lambda x, y: 1.0 - F.cosine_similarity(x, y)
# distance_function = PairwiseDistance(p=2, eps=1e-06,)

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
normalized_buy = np.array([normalize(i, axis=0, norm="max") for i in buy_patern])
normalized_sell = np.array([normalize(i, axis=0, norm="max") for i in sell_patern])

normalized_buy = normalized_buy.reshape(
    -1, PATTERN_SIZE, len(Train_df.columns.to_list()), 1
)
normalized_sell = normalized_sell.reshape(
    -1, PATTERN_SIZE, len(Train_df.columns.to_list()), 1
)

buy_patterns_as_classes = get_patterns_index_classes(normalized_buy, n_classes)
sell_patterns_as_classes = get_patterns_index_classes(normalized_sell, n_classes)

buy_list_of_pattern_and_neibours = []
for i in buy_patterns_as_classes:
    pattern_dict = get_class_and_neighbours(normalized_buy, i, n_samples_in_class)
    buy_list_of_pattern_and_neibours.append(pattern_dict)

sell_list_of_pattern_and_neibours = []
for i in sell_patterns_as_classes:
    pattern_dict = get_class_and_neighbours(normalized_sell, i, n_samples_in_class)
    sell_list_of_pattern_and_neibours.append(pattern_dict)

column_list = Train_df.columns.to_list()
named_patterns_heatmap(buy_patern, "Buy", EXTR_WINDOW, PATTERN_SIZE)
named_patterns_heatmap(sell_patern, "Sell", EXTR_WINDOW, PATTERN_SIZE)

for i, buy_patt in enumerate(buy_list_of_pattern_and_neibours):
    triplet_pattern_class_plot(
        buy_patt,
        patterns_to_df(buy_patern, column_list),
        "buy",
        i,
        EXTR_WINDOW,
        PATTERN_SIZE,
    )

for i, sell_patt in enumerate(sell_list_of_pattern_and_neibours):
    triplet_pattern_class_plot(
        sell_patt,
        patterns_to_df(sell_patern, column_list),
        "sell",
        i,
        EXTR_WINDOW,
        PATTERN_SIZE,
    )

train_x = []
metta_label = []

for i, j in zip(buy_list_of_pattern_and_neibours, sell_list_of_pattern_and_neibours):
    buy_class_samples = []
    sell_class_semples = []
    for a in i.pattern.values:
        buy_class_samples.append(normalized_buy[a])
    for b in j.pattern.values:
        sell_class_semples.append(normalized_sell[b])

    train_x.append(buy_class_samples)
    metta_label.append("buy")
    train_x.append(sell_class_semples)
    metta_label.append("sell")

train_triplets, labels = get_triplet_random(
    n_samples_to_train, len(metta_label), np.array(train_x)
)

tensor_A = torch.Tensor(train_triplets[0])  # transform to torch tensor
tensor_P = torch.Tensor(train_triplets[1])
tensor_N = torch.Tensor(train_triplets[2])

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

pred_label = []
prediction_metta_label = []
date = []
open = []
high = []
low = []
close = []
# volume = []
distance = []

net.eval()
with torch.no_grad():
    for indexI, eval_arr in enumerate(tqdm(eval_normlzd[:1000])):

        predictions_values = []

        for class_sample in train_x:
            anchor = class_sample[0].reshape(
                1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1
            )

            eval_arr = eval_arr.reshape(
                1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1
            )
            anchor = torch.Tensor(anchor)
            eval_arr = torch.Tensor(eval_arr)
            output1, output2, output3 = net(
                anchor.cuda().permute(0, 3, 1, 2),
                anchor.cuda().permute(0, 3, 1, 2),
                eval_arr.cuda().permute(0, 3, 1, 2),
            )
            # dist = distance_function()
            net_pred = distance_function(output1, output3)
            predictions_values.append(net_pred)

        date.append(Eval_dates_str[indexI + (PATTERN_SIZE - 1)])
        open.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [0]]))
        high.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [1]]))
        low.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [2]]))
        close.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [3]]))
        # volume.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [4]]))

        pred = min(predictions_values)
        distance.append(float(net_pred))
        pred_label.append(predictions_values.index(min(predictions_values)))
        prediction_metta_label.append(
            metta_label[predictions_values.index(min(predictions_values))]
        )

Predictions = pd.DataFrame(
    list(
        zip(date, open, high, low, close, pred_label, prediction_metta_label, distance)
    ),
    columns=[
        "date",
        "open",
        "high",
        "low",
        "close",
        "label",
        "metta_label",
        "distance",
    ],
)

# num_patterns = pd.value_counts(Predictions["label"]).to_frame()
# list_of_trashholds = [tresh_hold for _ in range(num_patterns.index.shape[0])]
# list_of_patterns = num_patterns.index.to_list()
# triplet_pred_plotting(Predictions, list_of_trashholds, list_of_patterns, FILENAME)

Predictions.to_csv(
    f"{DESTINATION_ROOT}/mid_dist_test_results_extr_window{EXTR_WINDOW}"
    f"_pattern_size{PATTERN_SIZE}_{model_name}.csv"
)
