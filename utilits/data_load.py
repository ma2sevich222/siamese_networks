import pandas as pd
from constants import START_TRAIN, END_TRAIN, START_TEST, END_TEST

pd.pandas.set_option("display.max_columns", None)
pd.set_option("expand_frame_repr", False)
pd.set_option("precision", 2)

def data_load(SOURCE_ROOT, FILENAME):
    df = pd.read_csv(f"{SOURCE_ROOT}/{FILENAME}")
    df.rename(columns=lambda x: x.replace(">", ""), inplace=True)
    df.rename(columns=lambda x: x.replace("<", ""), inplace=True)
    df.rename(columns=lambda x: x.replace(" ", ""), inplace=True)
    del df["ZeroLine"]
    columns = df.columns.tolist()
    """Формат даты в Datetime"""
    new_df = df["Date"].str.split(".", expand=True)
    df["Date"] = new_df[2] + "-" + new_df[1] + "-" + new_df[0] + " " + df["Time"]
    df.Date = pd.to_datetime(df.Date)
    df.dropna(axis=0, inplace=True)  # Удаляем наниты
    df["Datetime"] = df["Date"]
    df.set_index("Datetime", inplace=True)
    df.sort_index(ascending=True, inplace=False)
    df = df.rename(columns={"<Volume>": "Volume"})
    del df["Time"], df["Date"]
    del df['MACD'], df['MACDAvg'], df['MACDDiff'], df['AvgExp'], df['vwap_reset'], df['AvgExp.1']
    print(df)

    """Добавление фич"""
    df["SMA"] = df.iloc[:, 3].rolling(window=10).mean()
    df["CMA30"] = df["Close"].expanding().mean()
    df["SMA"] = df["SMA"].fillna(0)
    # print(df)

    """Отберем данные по максе"""
    mask_train = (df.index >= START_TRAIN) & (df.index <= END_TRAIN)
    Train_df = df.loc[mask_train]
    mask_test = (df.index >= START_TEST) & (df.index <= END_TEST)
    Eval_df = df.loc[mask_test]
    print(f'Датасет для поиска петтернов: {Train_df.shape}, период проверки: {Eval_df.shape}')

    """Сохраняем даты, удаляем из основынх датафрэймов"""
    # Train_dates = Train_df.index.to_list()
    Eval_dates = Eval_df.index.astype(str)
    Train_df = Train_df.reset_index(drop=True)
    Eval_df = Eval_df.reset_index(drop=True)
    Eval_dates_str = [str(i) for i in Eval_dates]

    return Train_df, Eval_df, Eval_dates_str
