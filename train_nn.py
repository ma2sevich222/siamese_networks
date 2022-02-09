import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import argrelextrema


pd.pandas.set_option('display.max_columns', None)
pd.set_option("expand_frame_repr", False)
pd.set_option("precision", 2)


source_root = 'source_root'
destination_root = 'outputs'
filename = 'VZ_15_Minutes_(with_indicators)_2018_18012022.txt'
indices = [i for i, x in enumerate(filename) if x == "_"]  # находим индексы вхождения '_'
ticker = filename[:indices[0]]

"""Загрузка и подготовка данных"""
df = pd.read_csv(f'{source_root}/{filename}')
df.rename(columns=lambda x: x.replace('>', ''), inplace=True)
df.rename(columns=lambda x: x.replace('<', ''), inplace=True)
df.rename(columns=lambda x: x.replace(' ', ''), inplace=True)
del df['ZeroLine']
columns = df.columns.tolist()

"""Формат даты в Datetime"""
print(df)
new_df = df['Date'].str.split('.', expand=True)
df['Date'] = new_df[2] + '-' + new_df[1] + '-' + new_df[0] + " " + df['Time']
df.Date = pd.to_datetime(df.Date)
df.dropna(axis=0, inplace=True)  # Удаляем наниты
df['Datetime'] = df['Date']
# df.set_index('Datetime', inplace=True)
# df.sort_index(ascending=True, inplace=False)
df = df.rename(columns={"<Volume>": "Volume"})
del df['Time'], df['Date']

"""Добавление фич"""
df['SMA'] =df.iloc[:,3].rolling(window=10).mean()
df['CMA30'] = df['Close'].expanding().mean()
df['SMA']=df['SMA'].fillna(0)
print(df)

"""Для обучения модели"""
START_TRAIN = '2018-01-01 09:00:00'
END_TRAIN = '2020-12-31 23:00:00'
"""Для тестирования модели"""
START_TEST = '2021-01-01 09:00:00'
END_TEST = '2021-12-31 23:00:00'
"""Отберем данные по максе"""
mask_train = (df.index >= START_TRAIN) & (df.index <= END_TRAIN)
Train_df = df.loc[mask_train]
mask_test = (df.index >= START_TEST) & (df.index <= END_TEST)
Eval_df = df.loc[mask_test]

# Train_df=NEW_DATA[:10000]
# Eval_df=NEW_DATA[10000:]
Train_dates = Train_df.index
Eval_dates = Eval_df.reset_index
# Train_df.drop(['Datetime'], axis = 1, inplace = True)
# Eval_df.drop(['Datetime'], axis = 1, inplace = True)
print(Train_dates)

'''Основыен параметры'''
num_classes = 2
extr_window = 40
n_size = 20  # размер мемори

def get_locals(data_df, n): # данные подаются в формате df

  data_df['index'] = data_df.index
  data_df['min']=data_df.iloc[argrelextrema(data_df.Close.values, np.less_equal,
                    order=n)[0]]['Close']
  data_df['max'] = data_df.iloc[argrelextrema(data_df.Close.values, np.greater_equal,
                    order=n)[0]]['Close']

  f = plt.figure()
  f.set_figwidth(80)
  f.set_figheight(65)
  plt.scatter(data_df.index1, data_df['min'], c='r')
  plt.scatter(data_df.index1, data_df['max'], c='g')
  plt.plot(data_df.index1, data_df['Close'])
  plt.show()

  Min_=data_df.loc[data_df['min'].isnull() == False]
  Min_.reset_index(inplace=True)
  Min_.drop(['max'],axis=1,inplace=True)

  Max_=data_df.loc[data_df['max'].isnull() == False]
  Max_.reset_index(inplace=True)
  Max_.drop(['min'],axis=1,inplace=True)

  data_df.drop(['index1', 'min','max'], axis = 1, inplace = True)
  return Min_,Max_


def get_patterns(data, min_indexes, max_indexes, n_backwatch):  # подаем дата как нумпи, индексы как лист

  negative_patterns=[]
  positive_patterns=[]
  for ind in min_indexes:
    if ind-n_backwatch>=0:
      neg=data[(ind-n_backwatch):ind]
      negative_patterns.append(neg)

  for ind in max_indexes:
    if ind-(2*n_backwatch)>= 0:
      pos=data[(ind-n_backwatch):ind]
      positive_patterns.append(pos)

  negative_patterns=np.array(negative_patterns)
  positive_patterns=np.array(positive_patterns)
  return negative_patterns, positive_patterns

Min_train_locals, Max_train__locals = get_locals(
  Train_df,
  extr_window
)

neg_patern, pos_patern = get_patterns(
  Train_df.to_numpy(),
  Min_train_locals['index1'].values.tolist(),
  Max_train__locals['index1'].values.tolist(),n_size
)

print(f'neg_patern.shape: {neg_patern.shape}\t|\tpos_patern.shape: {pos_patern.shape}')
exit()