import pandas as pd
import plotly.express as px

file_root = "outputs/2GC_2020_2022_30min_select_and_forward"
filename = "intermediaGC_2020_2022_30min_select_and_forward.xlsx"
final_df = pd.read_excel(f"{file_root}/{filename}")  # загружаем результаты  анализа

df_plot = final_df[
    [
        "Net Profit [$]",
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
