#######################################################
# Copyright © 2021-2099 Ekosphere. All rights reserved
# Author: Evgeny Matusevich
# Contacts: <ma2sevich222@gmail.com>
# File: forward.py
#######################################################
import pandas as pd
import plotly.express as px

file_root = "outputs/CL_2020_2022_CL_data_optune_18_06_2022_epoch_500"
filename = "intermedia_CL_2020_2022.xlsx"
final_df = pd.read_excel(f"{file_root}/{filename}")  # загружаем результаты  анализа

df_plot = final_df[
    [
        "Net Profit [$]",
        "Sharpe Ratio",
        "# Trades",
        "pattern_size",
        "extr_window",
        "overlap",
        "train_window",
        "select_dist_window",
        "forward_window",
    ]
]

fig = px.parallel_coordinates(
    df_plot,
    color="Net Profit [$]",
    labels={
        "Net Profit [$]": "Net Profit ($)",
        "Sharpe Ratio": "Sharpe Ratio",
        "# Trades": "Trades",
        "pattern_size": "pattern_size (bars)",
        "extr_window": "extr_window (bars)",
        "overlap": "overlap (bars)",
        "train_window": "train_window (bars)",
        "select_dist_window": "select_distance_window (bars)",
        "forward_window": "forward_window (bars)",
    },
    range_color=[df_plot["Net Profit [$]"].min(), df_plot["Net Profit [$]"].max()],
    color_continuous_scale=px.colors.sequential.Viridis,
    title=f"hyp_parameters_select_{filename[:-4]}",
)

fig.write_html(f"hyp_parameters_sel_{filename[:-4]}.htm")  # сохраняем в файл
fig.show()


"""file_root = "outputs/CL_2020_2022_CL_data_optune_18_06_2022_epoch_500"
filename = "intermedia_CL_2020_2022.xlsx"
final_df = pd.read_excel(f"{file_root}/{filename}")  # загружаем результаты  анализа

df_plot = final_df[
    [
        "values_0",
        "values_1",
        "pattern_size",
        "extr_window",
        "overlap",
        "train_window",
        "select_dist_window",
        "forward_window",
    ]
]

fig = px.parallel_coordinates(
    df_plot,
    color="values_0",
    labels={
        "values_0": "Net Profit ($)",
        "values_1": "Sharpe Ratio",
        "pattern_size": "pattern_size (bars)",
        "extr_window": "extr_window (bars)",
        "overlap": "overlap (bars)",
        "train_window": "train_window (bars)",
        "select_dist_window": "select_distance_window (bars)",
        "forward_window": "forward_window (bars)",
    },
    range_color=[df_plot["values_0"].min(), df_plot["values_0"].max()],
    color_continuous_scale=px.colors.sequential.Viridis,
    title=f"hyp_parameters_select_{filename[:-4]}_500_epochs",
)

fig.write_html(f"hyp_parameters_sel_{filename[:-4]}.htm")  # сохраняем в файл
fig.show()"""
