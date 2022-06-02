import pandas as pd
import plotly.express as px

final_df = pd.read_csv(
    "forward_30_min - Sheet1 (2)(1).csv"
)  # загружаем результаты  анализа

df_plot = final_df[
    ["Net Profit [$]", "pattern_size", "extr_window", "overlap", "train_window"]
]

"""df_plot = pd.DataFrame(
    {
        "Net Profit [$]": [18245.4, 11609, -4977.8],
        "pattern_size": [20, 20, 40],
        "extr_window": [40, 20, 20],
        "overlap": [0, 0, 0],
        "train_window": [3000, 3000, 3000],
        "select_distance_window": [3000, 3000, 3000],
        "forward_window": [1000, 1000, 1000],
    }
)"""
fig = px.parallel_coordinates(
    df_plot,
    color="Net Profit [$]",
    labels={
        "Net Profit [$]": "Net Profit ($)",
        "pattern_size": "pattern_size (bars)",
        "extr_window": "extr_window (bars)",
        "overlap": "overlap (bars)",
        "train_window": "train_window (bars)",
        "select_distance_window": "select_distance_window (bars)",
        "forward_window": "forward_window (bars)",
    },
    range_color=[df_plot["Net Profit [$]"].min(), df_plot["Net Profit [$]"].max()],
    color_continuous_scale=px.colors.sequential.Viridis,
    title="hyp_parameters_select_GC_2020_2022_30min",
)

fig.write_html("hyp_parameters_sel_GC_2020_2022_30min.html")  # сохраняем в файл
fig.show()
