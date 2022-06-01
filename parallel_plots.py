#######################################################
# Copyright © 2021-2099 Ekosphere. All rights reserved
# Author: Evgeny Matusevich
# Contacts: <ma2sevich222@gmail.com>
# File: parallel_plots.py
################################################




import pandas as pd
import plotly.express as px

final_df = pd.read_csv('outputs/15_min_gold_backtest.csv')  # загружаем результаты  анализа

df_plot = final_df[
    ['Net Profit [$]', 'pattern_size', 'extr_window', 'overlap', 'train_window', 'buy_before', 'sell_after']]
fig = px.parallel_coordinates(df_plot, color="Net Profit [$]",
                              labels={"Net Profit [$]": "Net Profit ($)", "pattern_size": "pattern_size (bars)",
                                      "extr_window": "extr_window (bars)", "train_window": "train_window (bars)",
                                      "overlap": "overlap (bars)", "buy_before": "buy_before dist",
                                      "sell_after": "sell_after dist"},
                              range_color=[df_plot['Net Profit [$]'].min(), df_plot['Net Profit [$]'].max()],
                              color_continuous_scale=px.colors.sequential.Viridis,
                              title='Зависимость профита от параметров паттерна')

# fig.write_html("1min_gold_dep_analisys.html")  # сохраняем в файл
fig.show()
