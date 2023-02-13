import pandas as pd


file = "outputs/deep_b_des_tree_GC_2020_2022_60min_21_07_2022/91_signals_GC_2020_2022_60min_train_window5000forward_window5644_lookback_size2.csv"
df = pd.read_csv(file)
df["Signal"] = df["Signal"].astype(int)
df.to_csv(
    "91_signals_GC_2020_2022_60min_train_window5000forward_window5644_lookback_size2.csv",
    index=False,
)
