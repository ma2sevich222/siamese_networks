from other_codes.not_used.constants import SOURCE_ROOT, FILENAME
import pandas as pd


df = pd.read_csv(f"{SOURCE_ROOT}/{FILENAME}", sep=";")
df = df[~df.Datetime.duplicated(keep="first")]
df.reset_index(drop=True, inplace=True)
df[
    [
        "Datetime",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "DiffEMA",
        "SmoothDiffEMA",
        "VolatilityTunnel",
        "BuyIntense",
        "SellIntense",
    ]
] = df[
    [
        "Datetime",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "DiffEMA",
        "SmoothDiffEMA",
        "VolatilityTunnel",
        "BuyIntense",
        "SellIntense",
    ]
].convert_dtypes(
    infer_objects=True, convert_string=True
)
df[
    [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "DiffEMA",
        "SmoothDiffEMA",
        "VolatilityTunnel",
        "BuyIntense",
        "SellIntense",
    ]
].replace(",", ".", regex=True, inplace=True)
df[
    [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "DiffEMA",
        "SmoothDiffEMA",
        "VolatilityTunnel",
        "BuyIntense",
        "SellIntense",
    ]
] = df[
    [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "DiffEMA",
        "SmoothDiffEMA",
        "VolatilityTunnel",
        "BuyIntense",
        "SellIntense",
    ]
].astype(
    float
)
df["Datetime"] = pd.to_datetime(df["Datetime"], format="%d.%m.%Y %H:%M:%S")
df.set_index("Datetime", inplace=True)
# df = df[~df.index.duplicated(keep='first')]
df.sort_index(ascending=True, inplace=False)
df_duplicated = df[df.index.duplicated(keep=False)].sort_index()  # проверка дубликатов
assert df_duplicated.shape[0] == 0, "В коде существуют дубликаты!"
df.reset_index(inplace=True)
df = df.iloc[1120535:]
df = df.reset_index(drop=True)
df.to_csv("CL_2020_2022.csv")
