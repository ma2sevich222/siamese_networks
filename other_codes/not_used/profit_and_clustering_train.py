import random
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.cluster import SpectralClustering
from models.torch_models import SiameseNetwork
from other_codes.not_used.data_load import test_data_load
from utilits.data_transforms import patterns_to_df
from other_codes.old_project.old_utilits.triplet_func_for_train import (
    get_triplet_random,
    get_patterns_with_profit,
    l_infinity,
)
from other_codes.old_project.old_utilits.triplet_func_for_train import (
    train_triplet_net,
    clusterized_pattern_save,
)
from utilits.visualisation_functios import clustered_pattern_train_plot
from utilits.visualisation_functios import (
    cluster_triplet_pattern_class_plot,
    plot_clasters,
)
from sklearn.preprocessing import StandardScaler

# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""" """""" """""" """""" """"" Parameters Block """ """""" """""" """""" """"""
from constants import *

profit_value = 0.0015
epochs = 15  # количество эпох
lr = 0.00005  # learnig rate
embedding_dim = 30  # размер скрытого пространства
margin = 5  # маржа для лосс функции
batch_size = 25  # размер батчсайз
n_samples_to_train = 500  # Количество триплетов для тренировки
tresh_hold = (
    15  # граница предсказаний ниже которой предсказания будут отображаться на графике
)
# distance_function = lambda x, y: 1.0 - F.cosine_similarity(x, y)  # функция расчета расстояния для триплет лосс
# distance_function.__name__ = 'cosine_distance' # имя для cos_dist
# distance_function = PairwiseDistance(p=2, eps=1e-06,)
distance_function = l_infinity
# distance_function = euclid_dist


seed = 3
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

""" Добавляем предобученную модель из модуля torchvision.models.
Переходим по ссылке https://pytorch.org/vision/stable/models.html,
подставляем нужную модель как в примере :
base_model = models.< название модели из документации >()"""

base_model = models.shufflenet_v2_x1_0()
model_name = base_model.__class__.__name__

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
"""""" """""" """""" """""" """"" Main Block """ """""" """""" """""" """""" """"""
"""indices = [
    i for i, x in enumerate(FILENAME) if x == "_"
]  # находим индексы вхождения '_'
ticker = FILENAME[: indices[0]]"""

"""Загрузка и подготовка данных"""
Train_df, Eval_df, train_dates = test_data_load(SOURCE_ROOT, FILENAME)


patterns, after_pattern_Close, indexes_with_profit = get_patterns_with_profit(
    Train_df,
    profit_value,
    EXTR_WINDOW,
    PATTERN_SIZE,
    OVERLAP,
    train_dates,
    save_to_dir=False,
)  # добавляем save_to_dir = True если хотим сохранить паттерны
print(
    f"Найдено {patterns.shape[0]}   при заданных условиях profit_value: {profit_value}, EXTR_WINDOW: {EXTR_WINDOW}  "
)
print(Train_df)


sil_score_max = -1
labels_list = []
sil_scores = []
scaler = StandardScaler()
pca = PCA(n_components=2)
patterns_std = scaler.fit_transform(
    patterns.reshape(-1, patterns.shape[1] * patterns.shape[2])
)
patterns_PCA = pd.DataFrame(pca.fit_transform(patterns_std))
for n_clusters in range(2, patterns.shape[0] - 1):
    # model = TimeSeriesKMeans(n_clusters=n_clusters, n_init=15, max_iter_barycenter=10, n_jobs=25,metric='softdtw', )
    model = SpectralClustering(n_clusters=n_clusters, n_init=10, n_jobs=10)
    labels = model.fit_predict(patterns_std)
    sil_score = silhouette_score(patterns_std, labels)
    labels_list.append(labels)
    sil_scores.append(sil_score)

    if sil_score > sil_score_max:
        sil_score_max = sil_score
        best_n_clusters = n_clusters

print(f"Предпочтительно использовать {best_n_clusters} класса")
# print(labels_list[sil_scores.index(max(sil_scores))])
# print(max(sil_scores))
# print(sil_scores.index(max(sil_scores)))
# print("Идекс лэйблов",labels_list[best_n_clusters-2])
# labels_list[sil_scores.index(max(sil_scores)

plot_clasters(
    patterns,
    labels_list[sil_scores.index(max(sil_scores))],
    EXTR_WINDOW,
    PATTERN_SIZE,
    profit_value,
    sil_score_max,
)

clustered_pattern_train_plot(
    Train_df,
    labels_list[sil_scores.index(max(sil_scores))],
    indexes_with_profit,
    best_n_clusters,
    profit_value,
    EXTR_WINDOW,
    PATTERN_SIZE,
    train_dates,
)

digit_indices = list(
    [
        np.where(np.array(labels_list[sil_scores.index(max(sil_scores))]) == i)[0]
        for i in range(best_n_clusters)
    ]
)


# normalized_patterns = np.array([normalize(i, axis=0, norm='max') for i in patterns])

normalized_patterns = np.array([i for i in patterns])
normalized_patterns = normalized_patterns.reshape(
    -1, PATTERN_SIZE, len(Train_df.columns.to_list()), 1
)

# n_samples = min(Counter(labels_list[sil_scores.index(max(sil_scores))]).values())
# uniq_triplets_n = (((n_samples - 1) * n_samples) // 2) * best_n_clusters

train_x = []
column_list = Train_df.columns.to_list()

for _, buy_patt in enumerate(digit_indices):
    cluster_triplet_pattern_class_plot(
        buy_patt,
        patterns_to_df(patterns, column_list),
        after_pattern_Close,
        "buy",
        _,
        profit_value,
        EXTR_WINDOW,
        PATTERN_SIZE,
    )

for i in range(best_n_clusters):
    class_samples = [normalized_patterns[j] for j in digit_indices[i][:]]
    class_samples = np.array(class_samples)
    train_x.append(class_samples)

clusterized_pattern_save(
    np.array(train_x), PATTERN_SIZE, EXTR_WINDOW, profit_value, OVERLAP
)
train_x = [i if len(i) > 1 else np.array([i[0], i[0]]) for i in train_x]
train_x = np.array(train_x, dtype=object)


train_triplets, labels = get_triplet_random(
    n_samples_to_train, best_n_clusters, train_x
)

print(" Размер данных для обучения:", np.array(train_triplets).shape)

tensor_A = torch.Tensor(train_triplets[0])  # transform to torch tensor
tensor_P = torch.Tensor(train_triplets[1])
tensor_N = torch.Tensor(train_triplets[2])

my_dataset = TensorDataset(tensor_A, tensor_P, tensor_N)  # create your datset
my_dataloader = DataLoader(my_dataset, batch_size=batch_size)


net = SiameseNetwork(embedding_dim=embedding_dim).cuda()


"""from torchviz import make_dot #-----------> Получаем схему вычислительных графов, сохраняется в основной директории
x=torch.zeros(3,1,20,5)                               #в виде файла png
yhata,yhatp,yhatn = net(x.cuda(),x.cuda(),x.cuda())
make_dot((yhata,yhatp,yhatn), params=dict(net.named_parameters())).render("rnn_torchviz", format="png")"""


train_triplet_net(lr, epochs, my_dataloader, net, distance_function)
torch.save(net, "my_model.pth")

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
# volume = []
all_pred = []
all_pred_labels = []
# signal = []  # лэйбл
k = 0

net.eval()
with torch.no_grad():
    for indexI, eval_arr in enumerate(tqdm(eval_samples[:2000])):

        sub_predictions_values = []
        sub_labels = []

        for classes in train_x:
            class_predictions = []
            sub_class_label = []
            for i, class_sample in enumerate(classes):
                anchor = class_sample.reshape(
                    1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1
                )

                eval_arr = eval_arr.reshape(
                    1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1
                )
                anchor = torch.Tensor(anchor)
                eval_arr = torch.Tensor(eval_arr)
                output1, output2, output3 = net(
                    anchor.cuda().permute(0, 3, 1, 2),
                    eval_arr.cuda().permute(0, 3, 1, 2),
                    eval_arr.cuda().permute(0, 3, 1, 2),
                )

                net_pred = distance_function(output1, output3)

                class_predictions.append(
                    net_pred.to("cpu").numpy()
                )  # a.to('cpu').numpy()
                sub_class_label.append(i)

            sub_predictions_values.append(min(class_predictions))
            sub_labels.append(
                sub_class_label[class_predictions.index(min(class_predictions))]
            )

        # date.append(Eval_dates_str[indexI + (PATTERN_SIZE - 1)])
        open.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [0]]))
        high.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [1]]))
        low.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [2]]))
        close.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [3]]))
        all_pred.append(sub_predictions_values)
        all_pred_labels.append(sub_labels)

        # volume.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [4]]))
        # Min_prediction_pattern_name.append(predictions_values.index(min(predictions_values)))

        # buy_pred = min(predictions_values)
        # distance.append(float(buy_pred))

"""""" """   if buy_pred <= 0.5:
            signal.append(1)
        else:
            signal.append(0)"""

"""Predictions = pd.DataFrame(
    list(zip(date, open, high, low, close)),
    columns=['date', 'open', 'high', 'low', 'close'])"""

Predictions = pd.DataFrame({"open": open, "high": high, "low": low, "close": close})


for i in range(best_n_clusters):
    Predictions[f"labels_clas_{i}"] = [j[i] for j in all_pred_labels]
    Predictions[f"distance_class_{i}"] = [float(j[i]) for j in all_pred]

Predictions.to_csv(
    f"{DESTINATION_ROOT}/withProfit_test_results_extrw{EXTR_WINDOW}_patsize{PATTERN_SIZE}_{model_name}.csv"
)
# best_patterns = []
"""for cl in range(best_n_clusters):
    b_patterns = class_full_analysys(Predictions, Eval_df, np.array(train_x), PATTERN_SIZE, EXTR_WINDOW, profit_value, OVERLAP, cl, save_best=False)
    best_patterns.append(b_patterns)"""


# best_patterns =[ i for i in best_patterns if len(i)>0]

"""Min_prediction_pattern_name = []
date = []
open = []
high = []
low = []
close = []
# volume = []
all_pred = []
all_pred_labels = []
# signal = []  # лэйбл
k = 0
net.eval()
with torch.no_grad():
    for indexI, eval_arr in enumerate(tqdm(eval_normlzd[:500])):

        sub_predictions_values = []
        sub_labels = []

        for classes in range(len(best_patterns)):
            class_predictions = []
            sub_class_label = []
            for i, class_sample in enumerate(best_patterns[classes]):
                anchor = train_x[classes][class_sample].reshape(1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1)

                eval_arr = eval_arr.reshape(1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1)
                anchor = torch.Tensor(anchor)
                eval_arr = torch.Tensor(eval_arr)
                output1, output2, output3 = net(anchor.cuda().permute(0, 3, 1, 2), eval_arr.cuda().permute(0, 3, 1, 2),
                                                eval_arr.cuda().permute(0, 3, 1, 2))

                net_pred = distance_function(output1, output2)

                class_predictions.append(net_pred.to('cpu').numpy())  # a.to('cpu').numpy()
                sub_class_label.append(i)

            sub_predictions_values.append(min(class_predictions))
            sub_labels.append(sub_class_label[class_predictions.index(min(class_predictions))])

        # date.append(Eval_dates_str[indexI + (PATTERN_SIZE - 1)])
        open.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [0]]))
        high.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [1]]))
        low.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [2]]))
        close.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [3]]))
        all_pred.append(sub_predictions_values)
        all_pred_labels.append(sub_labels)

        # volume.append(float(eval_array[indexI + (PATTERN_SIZE - 1), [4]]))
        # Min_prediction_pattern_name.append(predictions_values.index(min(predictions_values)))

        # buy_pred = min(predictions_values)
        # distance.append(float(buy_pred))"""

"""""" """   if buy_pred <= 0.5:
            signal.append(1)
        else:
            signal.append(0)"""

"""Predictions = pd.DataFrame(
    list(zip(date, open, high, low, close)),
    columns=['date', 'open', 'high', 'low', 'close'])"""

"""Predictions = pd.DataFrame({'open': open, 'high': high, 'low': low, 'close': close})

for i in range(best_n_clusters):
    Predictions[f'labels_clas_{i}'] = [j[i] for j in all_pred_labels]
    Predictions[f'distance_class_{i}'] = [float(j[i]) for j in all_pred]

Predictions.to_csv(f'{DESTINATION_ROOT}/withProfit_test_results_extr_window{EXTR_WINDOW}'
                   f'_pattern_size{PATTERN_SIZE}_{model_name}.csv')"""
