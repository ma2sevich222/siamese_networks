import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from tensorflow import keras

source_root = 'source_root'
destination_root = 'outputs'
filename = 'Eval_df.csv'
buy_patterns = 'buy_patterns.csv'
eval_dates = 'Eval_dates.csv'
out_filename = 'test_results.csv'
model_name = 'Best_model.h5'

n_size = 20
treshhold = 0.05

model = keras.models.load_model(f"{source_root}/{model_name}")
Eval_df = pd.read_csv(f"{source_root}/{filename}")
Eval_dates = pd.read_csv(f"{source_root}/{eval_dates}")
# загружаем массив размечанных паттернов и результаты тестирования модели
buy_loader = np.loadtxt(f'{destination_root}/{buy_patterns}')

eval_array = Eval_df.to_numpy()
eval_samples = [eval_array[i - n_size:i] for i in range(len(eval_array)) if i - n_size >= 0]
eval_normlzd = [normalize(i, axis=0, norm='max') for i in eval_samples]
eval_normlzd = np.array(eval_normlzd).reshape(-1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1)
print(eval_normlzd.shape)

buy_patterns = buy_loader.reshape(-1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1)

Eval_str_dates = [str(i) for i in Eval_dates]
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
treshhold = treshhold

for indexI, eval in enumerate(eval_normlzd):

    buy_predictions = []

    for buy in buy_patterns:
        buy_pred = model.predict([buy.reshape(1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1),
                                  eval.reshape(1, eval_samples[0].shape[0], eval_samples[0][0].shape[0], 1)])
        buy_predictions.append(buy_pred)

    date.append(Eval_str_dates[indexI + (n_size - 1)])
    open.append(float(eval_array[indexI + (n_size - 1), [0]]))
    high.append(float(eval_array[indexI + (n_size - 1), [1]]))
    low.append(float(eval_array[indexI + (n_size - 1), [2]]))
    close.append(float(eval_array[indexI + (n_size - 1), [3]]))
    volume.append(float(eval_array[indexI + (n_size - 1), [4]]))
    Min_prediction_pattern_name.append(buy_predictions.index(min(buy_predictions)))

    min_ex = min(buy_predictions)
    distance.append(float(min_ex))

    if min_ex <= treshhold:

        signal.append(1)
    else:
        signal.append(0)

Predictions = pd.DataFrame(
    list(zip(date, open, high, low, close, volume, signal, Min_prediction_pattern_name, distance)),
    columns=['date', 'open', 'high', 'low', 'close', 'volume', 'signal', 'pattern No.', 'distance'])

Predictions.to_csv(f'{destination_root}/{out_filename}')
