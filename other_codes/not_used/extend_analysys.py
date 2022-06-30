import pandas as pd
import numpy as np
from constants import *
import plotly.express as px
from other_codes.not_used.data_load import data_load_OHLC
from plotly.subplots import make_subplots
import plotly.graph_objects as go

destination_root = "outputs"
profit_test_with_clustering_file_name = (
    "withProfit_test_results_extr_window40_pattern_size10_ResNet.csv"
)
profit_value = 0.03

Train_df, Eval_df, Eval_dates_str = data_load_OHLC(SOURCE_ROOT, FILENAME)

df = pd.read_csv(f"{destination_root}/{profit_test_with_clustering_file_name}")
eval_samp_df = [
    Eval_df[i - PATTERN_SIZE : i] for i in range(len(Eval_df)) if i - PATTERN_SIZE >= 0
]

# Смотрим отношение предсказаний по каждому паттерну и общего кол-ва предсказаний.

# Класс 0
profit_columns = []
for i in range(len(df)):

    get_profit_list = []
    for j in range(1, EXTR_WINDOW + 1):

        if i + j <= len(df) - 1:

            value = df.iloc[i][df.columns.get_loc("close")]
            profit_from_value = df.iloc[i][df.columns.get_loc("close")] * profit_value

            if value + profit_from_value >= (
                df.iloc[i + j][df.columns.get_loc("close")]
            ):
                get_profit_list.append(1)
            else:
                get_profit_list.append(0)
        else:
            get_profit_list.append(0)

    profit_columns.append(get_profit_list)


for i in range(len(profit_columns[0])):
    df[f"profit_after_EXTR_WINDOW = {i+1}"] = [b[i] for b in profit_columns]

# берем класс и находим уникальные значения

dict_unig = df["labels_clas_0"].value_counts()
dict_unig = dict_unig.to_dict()
print(dict_unig)
print(dict_unig.keys())


# получили словарь из уникальных паттернов и кол-во срабатываний
# берем для примера один паттерн


# df[f'profit_after_{EXTR_WINDOW}'] = get_profit_list

print(df.loc[(df["labels_clas_0"] == 32)])

# df.iloc[i][df.columns.get_loc("marks")] +3>=max(df.iloc[i:i+j,df.columns.get_loc("marks")].values.tolist()):
label_df = df.loc[(df["labels_clas_0"] == 32)]
label_pred_window = label_df.iloc[:, -EXTR_WINDOW:]
print(label_df)
# print(label_pred_window)
print(len(label_df))
# df.iloc[:,-1:]
print(np.sum(label_pred_window.to_numpy(), axis=0))
label_summary = np.sum(label_pred_window.to_numpy(), axis=0)


fig = make_subplots(rows=len(label_df) + 1, cols=1)
fig.add_trace(go.Bar(x=[i for i in range(1, 40)], y=label_summary), 1, 1)
for i in range(len(label_df)):

    fig.add_trace(
        go.Candlestick(
            x=Eval_df.index.values,
            open=Eval_df["Open"],
            high=Eval_df["High"],
            low=Eval_df["Low"],
            close=Eval_df["Close"],
        ),
        2,
        1,
    )


fig.show()


label_summary = np.sum(label_pred_window.to_numpy(), axis=0)
mean_pred_distance = label_df["distance_class_0"].mean()
fig = px.bar(x=[i for i in range(1, 40)], y=label_summary)
fig.show()
