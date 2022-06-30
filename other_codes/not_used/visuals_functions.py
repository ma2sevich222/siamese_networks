import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_prediction(data_df, sell_trash, hold_trash_per, buy_trash, filename):
    sell_periods = []
    buy_periods = []
    hold_periods = []

    for ind, i in enumerate(data_df["Distance"].values.tolist()):
        if i >= sell_trash:
            sell_periods.append(ind)
        elif i > hold_trash_per[0] and i < hold_trash_per[1]:
            hold_periods.append(ind)
        elif i <= buy_trash:
            buy_periods.append(ind)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)  # go.Figure()
    fig.add_trace(
        go.Scatter(x=data_df.Datetime, y=data_df["Close"], mode="lines", name="CLOSE")
    )

    fig.add_trace(
        go.Bar(
            x=data_df["Datetime"][sell_periods],
            y=[data_df["Close"].max() for i in range(len(sell_periods))],
            marker_color=["green" for clr in range(len(sell_periods))],
            name=f"Ожидается  рост цены, нижняя граница дистанции  = {sell_trash}",
            opacity=0.4,
        )
    )
    fig.add_trace(
        go.Bar(
            x=data_df["Datetime"][hold_periods],
            y=[data_df["Close"].max() for i in range(len(hold_periods))],
            marker_color=["yellow" for clr in range(len(hold_periods))],
            name=f"Незначительное движение цены, период  = {hold_trash_per}",
            opacity=0.4,
        )
    )
    fig.add_trace(
        go.Bar(
            x=data_df["Datetime"][buy_periods],
            y=[data_df["Close"].max() for i in range(len(buy_periods))],
            marker_color=["red" for clr in range(len(buy_periods))],
            name=f"Ожидается падение цены, верхняя граница дистанции  = {buy_trash}",
            opacity=0.4,
        )
    )

    fig.add_trace(
        go.Bar(
            x=data_df["Datetime"],
            y=data_df["Distance"],
            marker_color="crimson",
            name="Динамика изменения дистанции предсказания",
        ),
        2,
        1,
    )
    fig.update_yaxes(title_text="Дистанция предсказания", row=2, col=1)
    fig.update_layout(
        title=f"Движение цены предсказанное сетью для файла {filename} ",
        xaxis_title="DATE",
        yaxis_title="CLOSE",
        legend_title="Legend",
    )

    fig.show()
