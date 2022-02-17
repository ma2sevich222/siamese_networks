import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import seaborn as sns

from constants import TRESHHOLD_DISTANCE

pd.pandas.set_option('display.max_columns', None)
pd.set_option("expand_frame_repr", False)
pd.set_option("precision", 2)


source_root = 'outputs'
destination_root = 'outputs'
file_name = 'test_results_extr_window60_pattern_size15.csv'
# file_name = 'test_results.csv'


df = pd.read_csv(f'{source_root}/{file_name}')
df = df.rename(columns={"pattern No.": "pattern"})
del df['Unnamed: 0']
print(df)
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
def extend_plotting(data, tresh_list, pattern_list):
    filtered_df = []
    for i, j in zip(tresh_list, pattern_list):
        sample_df = data[(data.distance <= i) & (data.pattern == j)]
        filtered_df.append(sample_df)

        df_pattern = df[(df.pattern == j)]
        sns.distplot(df_pattern["distance"])
        plt.title(f'pattern = {j}:\n'
                  f'min distance = {np.round(df_pattern["distance"].min(), 4)},   '
                  f'max distance = {np.round(df_pattern["distance"].max(), 4)}')
        plt.show()
        print(filtered_df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data['close'], mode='lines', name='CLOSE'))
    for i, j, k in zip(filtered_df, tresh_list, pattern_list):
        fig.add_trace(go.Scatter(x=i['date'], y=i['close'], mode='markers', name=f"distance <= {j} / patern:{k}",
                                 marker=dict(symbol='triangle-up', size=15)))

    fig.update_layout(title=f'BUY signals predictions for file {file_name}',
                      xaxis_title='DATE', yaxis_title='CLOSE', legend_title='Legend')
    fig.show()

#  Покажем все распозненные паттерны
list_of_tr = [1.405 for _ in range(num_patterns.index.shape[0])]
extend_plotting(df, list_of_tr, num_patterns.index.to_list())

# покажем конкретный паттерн
list_of_tr = [1.41]
list_of_patt = [129]
extend_plotting(df, list_of_tr, list_of_patt)
