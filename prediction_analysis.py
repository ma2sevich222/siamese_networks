import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from matplotlib import pyplot as plt

pd.pandas.set_option('display.max_columns', None)
pd.set_option("expand_frame_repr", False)
pd.set_option("precision", 2)


source_root = 'outputs'
destination_root = 'outputs'
file_name = 'test_results_latentdim10.csv'

treshhold_distance = 0.01
pattern_num = 4

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
plt.title(f'Число определенных паттернов по видам\n при treshhold_distance = {treshhold_distance}')
plt.show()

def ploter(data, treshhold, pattern):

   Demo = data[(data.distance <= treshhold) & (data.signal == 1) & (data.pattern == pattern)]
   num_patterns = pd.value_counts(Demo["pattern"]).to_frame()
   print(f'Число распознанных паттернов #{num_patterns.index[0]}:\t{num_patterns["pattern"][pattern]} items  '
         f'\t|\tпри treshhold_distance = {treshhold}')

   fig1 = px.line(x=data['date'], y=data['close'])
   fig2 = px.scatter(x=Demo['date'], y=Demo['close'],
                     color_discrete_sequence=['green'],
                     title="Buy Signals")
   fig2.update_xaxes(type='category')
   fig2.update_traces(marker_size=12)
   fig2.update_traces(marker_symbol='triangle-up')  # https://plotly.com/python/marker-style/
   fig = go.Figure(data=fig1.data + fig2.data)
   fig.update_xaxes(title_text="Date")
   fig.update_yaxes(title_text="Close")
   fig.update_layout(title_text=f'Сигналы входа для паттерна {pattern},  '
                                f'для treshhold_distance = {treshhold}')
   fig.show()


# ploter(df, treshhold=0.03, pattern=48)
# ploter(df, treshhold=0.003, pattern=57)
# ploter(df, treshhold=0.005, pattern=4)
# ploter(df, treshhold=0.04, pattern=34)
# ploter(df, treshhold=0.005, pattern=77)

# ================================================================================================
def extend_plotting(data, tresh_list, pattern_list):
    filtered_df = []
    for i, j in zip(tresh_list, pattern_list):
        sample_df = data[(data.distance <= i) & (data.signal == 1) & (data.pattern == j)]
        filtered_df.append(sample_df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data['close'], mode='lines', name='CLOSE'))
    for i, j, k in zip(filtered_df, tresh_list, pattern_list):
        fig.add_trace(go.Scatter(x=i['date'], y=i['close'], mode='markers', name=f"distance <= {j}/patern:{k}",
                                 marker=dict(symbol='triangle-up', size=15)))

    fig.update_layout(title='BUY signals predictions', xaxis_title='DATE', yaxis_title='CLOSE', legend_title='Legend')
    fig.show()

list_of_tr = [0.001, 0.01]
list_of_patt = [4, 166]

extend_plotting(df, list_of_tr, list_of_patt)
