import random
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from models.torch_models import SiameseNetwork
from other_codes.not_used.data_load import test_data_load
from other_codes.old_project.old_utilits.triplet_func_for_train import (
    get_triplet_random,
    test_get_patterns_and_other_classes_with_profit,
)
from other_codes.old_project.old_utilits.triplet_func_for_train import (
    train_triplet_net,
    clusterized_pattern_save,
)
from sklearn.preprocessing import StandardScaler

print(torch.__version__)
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""" """""" """""" """""" """"" Parameters Block """ """""" """""" """""" """"""
from constants import *

#  'lr': 3.061630721283881e-06, 'batch_size': 85, 'embedding_dim': 55}
profit_value = 0.0015
epochs = 10  # количество эпох
lr = 3.061630721283881e-06  # learnig rate
embedding_dim = 55  # размер скрытого пространства
margin = 10  # маржа для лосс функции
batch_size = 85  # размер батчсайз
n_samples_to_train = 1000  # Количество триплетов для тренировки
tresh_hold = (
    15  # граница предсказаний ниже которой предсказания будут отображаться на графике
)
distance_function = lambda x, y: 1.0 - F.cosine_similarity(
    x, y
)  # функция расчета расстояния для триплет лосс
# distance_function = PairwiseDistance(p=2, eps=1e-06,)
# distance_function = l_infinity
# distance_function = euclid_dist
# distance_function = manhatten_dist


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

base_model = models.shufflenet_v2_x1_0(pretrained=True)
model_name = base_model.__class__.__name__

"""""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" """""" ""
"""""" """""" """""" """""" """"" Main Block """ """""" """""" """""" """""" """"""
"""indices = [
    i for i, x in enumerate(FILENAME) if x == "_"
]  # находим индексы вхождения '_'
ticker = FILENAME[: indices[0]]"""

"""Загрузка и подготовка данных"""
Train_df, Eval_df, train_dates = test_data_load(SOURCE_ROOT, FILENAME)


(
    patterns,
    after_pattern_Close,
    indexes_with_profit,
    train_x,
) = test_get_patterns_and_other_classes_with_profit(
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


clusterized_pattern_save(
    np.array(train_x), PATTERN_SIZE, EXTR_WINDOW, profit_value, OVERLAP
)
best_n_clusters = len(train_x)


train_triplets, labels = get_triplet_random(
    n_samples_to_train, best_n_clusters, np.array(train_x, dtype=object)
)

print(" Размер данных для обучения:", np.array(train_triplets).shape)

tensor_A = torch.Tensor(train_triplets[0])  # transform to torch tensor
tensor_P = torch.Tensor(train_triplets[1])
tensor_N = torch.Tensor(train_triplets[2])

my_dataset = TensorDataset(tensor_A, tensor_P, tensor_N)  # create your datset
my_dataloader = DataLoader(my_dataset, batch_size=batch_size)


"""def objective(trial):
    # boundaries for the optimizer's
    lr = trial.suggest_loguniform("lr", 1e-8, 1e-2)
    tbatch_size = trial.suggest_int("batch_size", 5, 245, step=10)
    embedding_dim = trial.suggest_int("embedding_dim", 5, 400, step=5)
    my_dataset = TensorDataset(tensor_A, tensor_P, tensor_N)
    my_dataloader = DataLoader(my_dataset, batch_size=tbatch_size)
    ##### If you need more parameters for optimization, it is done like this:
    # new_parameter =  trial.suggest_loguniform("new_parameter", lower_bound, upper_bound)
    tdistance_function = distance_function
    # create new model(and all parameters) every iteration
    model = SiameseNetwork( embedding_dim=embedding_dim).cuda()
    #model.latent_dim = latent_dim
    #criterion = nn.TripletMarginWithDistanceLoss(distance_function=distance_function)
    #optimizer = optim.Adam(net.parameters(), lr)
      # learning step regulates by optuna

    # To save time, we will take only 5 epochs
    _, last_epoch_loss = train_triplet_net(lr, epochs, my_dataloader, model, tdistance_function)
    return last_epoch_loss



# Create "exploration"
study = optuna.create_study(direction="minimize", study_name="Optimal lr")


study.optimize(
    objective, n_trials=10)

print(study.best_params)
optuna.visualization.plot_optimization_history(study)"""


net = SiameseNetwork(embedding_dim=embedding_dim).cuda()


"""from torchviz import make_dot -----------> Получаем схему вычислительных графов, сохраняется в основной директории
x=torch.zeros(3,1,20,15)                               в виде файла png
yhata,yhatp,yhatn = net(x.cuda(),x.cuda(),x.cuda())
make_dot((yhata,yhatp,yhatn), params=dict(net.named_parameters())).render("rnn_torchviz", format="png")"""


train_triplet_net(lr, epochs, my_dataloader, net, distance_function)
# torch.save(net,'my_model.pth')

"""Тест модели"""

eval_array = Eval_df.to_numpy()
eval_samples = [
    eval_array[i - PATTERN_SIZE : i]
    for i in range(len(eval_array))
    if i - PATTERN_SIZE >= 0
]
# eval_normlzd = [normalize(i, axis=0, norm='max') for i in eval_samples]
scaler = StandardScaler()
eval_normlzd = [scaler.fit_transform(i) for i in eval_samples]
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
all_pred = []
all_pred_labels = []
# signal = []  # лэйбл
k = 0

net.eval()
with torch.no_grad():
    for indexI, eval_arr in enumerate(tqdm(eval_normlzd)):

        sub_predictions_values = []
        sub_labels = []

        for classes in train_x:
            class_predictions = []
            sub_class_label = []
            # for i, class_sample in enumerate(classes):
            for i in range(1):
                anchor = classes[0].reshape(
                    1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1
                )

                eval_arr_r = eval_arr.reshape(
                    1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1
                )
                anchor = torch.Tensor(anchor)
                eval_arr_r = torch.Tensor(eval_arr_r)
                output1, output2, output3 = net(
                    anchor.cuda().permute(0, 3, 1, 2),
                    eval_arr_r.cuda().permute(0, 3, 1, 2),
                    eval_arr_r.cuda().permute(0, 3, 1, 2),
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

        date.append(train_dates[indexI + (PATTERN_SIZE - 1)])
        open.append(float(eval_samples[indexI][-1, [0]]))
        high.append(float(eval_samples[indexI][-1, [1]]))
        low.append(float(eval_samples[indexI][-1, [2]]))
        close.append(float(eval_samples[indexI][-1, [3]]))
        all_pred.append(sub_predictions_values)
        all_pred_labels.append(sub_labels)

        volume.append(float(eval_samples[indexI][-1, [4]]))
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

Predictions = pd.DataFrame(
    {
        "date": date,
        "open": open,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }
)


for i in range(best_n_clusters):
    Predictions[f"labels_clas_{i}"] = [j[i] for j in all_pred_labels]
    Predictions[f"distance_class_{i}"] = [float(j[i]) for j in all_pred]

Predictions.to_csv(
    f"{DESTINATION_ROOT}/withProfit_test_results_extrw{EXTR_WINDOW}_patsize{PATTERN_SIZE}_{model_name}.csv"
)


Predictions.rename(columns={"distance_class_0": "distance"}, inplace=True)

for_send = Predictions[
    ["date", "open", "high", "low", "close", "volume", "distance"]
].copy()
for_send.to_csv(
    f"{DESTINATION_ROOT}/withProfit_test_results_extrw{EXTR_WINDOW}_patsize{PATTERN_SIZE}_send.csv"
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
