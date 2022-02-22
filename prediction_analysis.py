import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt, rcParams
import seaborn as sns

from constants import TRESHHOLD_DISTANCE

pd.pandas.set_option('display.max_columns', None)
pd.set_option("expand_frame_repr", False)
pd.set_option("precision", 2)


source_root = 'outputs'
destination_root = 'outputs'
file_name = 'test_results_extr_window60_latent_dim5_pattern_size15.csv'
# file_name = 'test_results.csv'


df = pd.read_csv(f'{source_root}/{file_name}')
df = df.rename(columns={"pattern No.": "pattern"})
df = df.set_index('Unnamed: 0')
print(f'\nВсего распознано уникальных паттернов:\t{len(pd.unique(df["pattern"]))}')
num_patterns = pd.value_counts(df["pattern"]).to_frame()
print(f'Распределение числа распознанных паттернов:\n{num_patterns.T}\n')

fig, ax = plt.subplots()
ax.bar(num_patterns.index, num_patterns["pattern"])
ax.set_xlabel('Номер паттерна')
ax.set_ylabel('Число распознаваний')
plt.title(f'Число определенных паттернов по видам\n при treshhold_distance = {TRESHHOLD_DISTANCE}')
plt.show()



# ================================================================================================
def displot_plotting(data, pattern_list):
    filtered_df = []
    for j in zip(pattern_list):
        sample_df = data.distance
        filtered_df.append(sample_df)
        df_pattern = df[(df.pattern == j)]
        sns.displot(df_pattern["distance"]).set(title=f'pattern = {j[0]}')
        plt.show()
        # print(filtered_df)
list_of_patterns = num_patterns.index.to_list()
displot_plotting(df, list_of_patterns)


def extend_plotting(data, tresh_list, pattern_list):
    filtered_df = []
    for i, j in zip(tresh_list, pattern_list):
        sample_df = data[(data.distance <= i) & (data.pattern == j)]
        filtered_df.append(sample_df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data['close'], mode='lines', name='CLOSE'))
    for i, j, k in zip(filtered_df, tresh_list, pattern_list):
        fig.add_trace(go.Scatter(x=i['date'], y=i['close'], mode='markers', name=f"distance <= {j} / patern:{k}",
                                 marker=dict(symbol='triangle-up'
                                             , size=15)))
    fig.update_layout(title=f'BUY signals predictions for file {file_name}',
                      xaxis_title='DATE', yaxis_title='CLOSE', legend_title='Legend')
    fig.show()



#  Покажем все распозненные паттерны
for treshholg in [1, 0.8, 0.4, 0.2, 0.1, 0.01]:
    list_of_trashholds = [treshholg for _ in range(num_patterns.index.shape[0])]
    list_of_patterns = num_patterns.index.to_list()
    extend_plotting(df, list_of_trashholds, list_of_patterns)


# покажем конкретный паттерн
list_of_trashholds = [0.104]
list_of_patterns = [53]
extend_plotting(df, list_of_trashholds, list_of_patterns)
