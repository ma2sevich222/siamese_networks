#######################################################
# Copyright © 2021-2099 Ekosphere. All rights reserved
# Author: Evgeny Matusevich
# Contacts: <ma2sevich222@gmail.com>
# File: data_load.py
#######################################################
import pandas as pd

pd.pandas.set_option("display.max_columns", None)
pd.set_option("expand_frame_repr", False)
pd.set_option("precision", 2)

'''def data_load(SOURCE_ROOT, FILENAME):
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

    return Train_df, Eval_df, Eval_dates_str'''


def data_load_OHLCV(SOURCE_ROOT, FILENAME, START_TEST, END_TEST, PATTERN_SIZE, TRAIN_WINDOW):
    df = pd.read_csv(f"{SOURCE_ROOT}/{FILENAME}")
    '''df = pd.read_csv(f"{SOURCE_ROOT}/{FILENAME}", sep=';')
    df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'DiffEMA', 'SmoothDiffEMA', 'VolatilityTunnel',
        'BuyIntense', 'SellIntense']] = df[
        ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'DiffEMA', 'SmoothDiffEMA', 'VolatilityTunnel',
         'BuyIntense', 'SellIntense']].convert_dtypes(infer_objects=True, convert_string=True)
    df[['Open', 'High', 'Low', 'Close', 'Volume', 'DiffEMA', 'SmoothDiffEMA', 'VolatilityTunnel', 'BuyIntense',
        'SellIntense']].replace(',', '.', regex=True, inplace=True)
    df[['Open', 'High', 'Low', 'Close', 'Volume', 'DiffEMA', 'SmoothDiffEMA', 'VolatilityTunnel', 'BuyIntense',
        'SellIntense']] = df[
        ['Open', 'High', 'Low', 'Close', 'Volume', 'DiffEMA', 'SmoothDiffEMA', 'VolatilityTunnel', 'BuyIntense',
         'SellIntense']].astype(float)
    # df.rename(columns=lambda x: x.replace(">", ""), inplace=True)
    # df.rename(columns=lambda x: x.replace("<", ""), inplace=True)
    # df.rename(columns=lambda x: x.replace(" ", ""), inplace=True)
    # del df["ZeroLine"]
    #columns = df.columns.tolist()'''
    """Формат даты в Datetime"""
    # new_df = df["Date"].str.split(".", expand=True)
    # df["Date"] = new_df[2] + "-" + new_df[1] + "-" + new_df[0] + " " + df["Time"]
    start_test_index = df.index[df['Datetime'] == START_TEST] - PATTERN_SIZE + 1
    end_train_index = df.index[df['Datetime'] == START_TEST] - PATTERN_SIZE
    start_train_index = df.index[df['Datetime'] == START_TEST] - PATTERN_SIZE - TRAIN_WINDOW + 1
    df['Datetime'] = pd.to_datetime(df['Datetime'], infer_datetime_format=True)
    # print(start_test_index, end_train_index, start_train_index)

    START_TEST_upd = df.loc[start_test_index, 'Datetime'].item()
    END_TRAIN = df.loc[end_train_index, 'Datetime'].item()
    START_TRAIN = df.loc[start_train_index, 'Datetime'].item()
    df.Datetime = pd.to_datetime(df.Datetime)

    # df.dropna(axis=0, inplace=True)  # Удаляем наниты
    # df["Datetime"] = df["Date"]
    df.set_index("Datetime", inplace=True)
    df.sort_index(ascending=True, inplace=False)
    # df = df.rename(columns={"<Volume>": "Volume"})
    # del df["Time"], df["Date"]
    # del df['MACD'], df['MACDAvg'], df['MACDDiff'], df['AvgExp'], df['vwap_reset'], df['AvgExp.1']

    """Отберем данные по максе"""
    mask_train = (df.index >= START_TRAIN) & (df.index <= END_TRAIN)
    Train_df = df.loc[mask_train]
    mask_test = (df.index >= START_TEST_upd) & (df.index <= END_TEST)
    Eval_df = df.loc[mask_test]
    print(f'Датасет для поиска петтернов: {Train_df.shape}, период проверки: {Eval_df.shape}')

    """Сохраняем даты, удаляем из основынх датафрэймов"""
    Train_dates = pd.DataFrame({'Datetime': Train_df.index.values})
    Eval_dates = pd.DataFrame({'Datetime': Eval_df.index.values})
    Train_df = Train_df.reset_index(drop=True)
    Eval_df = Eval_df.reset_index(drop=True)
    # del Train_df['Unnamed: 0']
    # del Eval_df['Unnamed: 0']
    # Eval_dates = Eval_dates.reset_index(drop=True)
    # Eval_dates = Eval_df.index.to_list()
    # Eval_dates_str = [str(i) for i in Eval_dates]
    print('********* Данные для обучения *********')
    print(Train_df.head())
    print('********* Данные для тестирования *********')
    print(Eval_df.head())

    return Train_df, Eval_df, Train_dates, Eval_dates


def convert_time(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)


def test_data_load(SOURCE_ROOT, FILENAME):
    df = pd.read_csv(f"{SOURCE_ROOT}/{FILENAME}")
    dates_train = pd.to_datetime(df['DATETIME'])
    # del df["ZeroLine"]

    del df["DATE"], df["T"], df['DATETIME']

    print(df)

    """Добавление фич"""
    df["SMA"] = df.iloc[:, 3].rolling(window=10).mean()
    df["CMA30"] = df["C"].expanding().mean()
    df["SMA"] = df["SMA"].fillna(0)

    df = df[['O', 'H', 'L', 'C', 'V', 'BI', 'SI', 'LO', 'SO', 'VO', 'OTO', 'PRO', 'SIGNAL', 'SMA', 'CMA30']]

    """Отберем данные по максе"""
    Train_df = df.iloc[:10000]
    dates_train = dates_train.iloc[:10000]

    Eval_df = df.iloc[10000:15000]
    print(f'Датасет для поиска петтернов: {Train_df.shape}, период проверки: {Eval_df.shape}')

    Train_df.rename(columns={"H": "High", "L": "Low", "C": "Close", "O": "Open", "V": "Volume"}, inplace=True)
    Eval_df.rename(columns={"H": "High", "L": "Low", "C": "Close", "O": "Open", "V": "Volume"}, inplace=True)
    tr = Train_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    ev = Eval_df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    return tr, ev, dates_train


def convert_time(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)


def data_load_CL(SOURCE_ROOT, FILENAME, START_TEST, END_TEST, PATTERN_SIZE, TRAIN_WINDOW):
    # df = pd.read_csv(f"{SOURCE_ROOT}/{FILENAME}")
    df = pd.read_csv(f"{SOURCE_ROOT}/{FILENAME}", sep=';')
    df = df[~df.Datetime.duplicated(keep='first')]
    df.reset_index(drop=True, inplace=True)
    df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'DiffEMA', 'SmoothDiffEMA', 'VolatilityTunnel',
        'BuyIntense', 'SellIntense']] = df[
        ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'DiffEMA', 'SmoothDiffEMA', 'VolatilityTunnel',
         'BuyIntense', 'SellIntense']].convert_dtypes(infer_objects=True, convert_string=True)
    df[['Open', 'High', 'Low', 'Close', 'Volume', 'DiffEMA', 'SmoothDiffEMA', 'VolatilityTunnel', 'BuyIntense',
        'SellIntense']].replace(',', '.', regex=True, inplace=True)
    df[['Open', 'High', 'Low', 'Close', 'Volume', 'DiffEMA', 'SmoothDiffEMA', 'VolatilityTunnel', 'BuyIntense',
        'SellIntense']] = df[
        ['Open', 'High', 'Low', 'Close', 'Volume', 'DiffEMA', 'SmoothDiffEMA', 'VolatilityTunnel', 'BuyIntense',
         'SellIntense']].astype(float)
    # df.rename(columns=lambda x: x.replace(">", ""), inplace=True)
    # df.rename(columns=lambda x: x.replace("<", ""), inplace=True)
    # df.rename(columns=lambda x: x.replace(" ", ""), inplace=True)
    # del df["ZeroLine"]
    # columns = df.columns.tolist()
    """Формат даты в Datetime"""
    # new_df = df["Date"].str.split(".", expand=True)
    # df["Date"] = new_df[2] + "-" + new_df[1] + "-" + new_df[0] + " " + df["Time"]
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d.%m.%Y %H:%M:%S')
    print(df.iloc[25, 0])
    print(df['Datetime'])
    print(df.index[df['Datetime'] == '2022-05-06 12:37:23'])
    start_test_index = df.index[df['Datetime'] == START_TEST] - PATTERN_SIZE + 1
    end_train_index = df.index[df['Datetime'] == START_TEST] - PATTERN_SIZE
    start_train_index = df.index[df['Datetime'] == START_TEST] - PATTERN_SIZE - TRAIN_WINDOW + 1
    # df['Datetime'] = pd.to_datetime(df['Datetime'], infer_datetime_format=True)  #infer_datetime_format=True

    # print(start_test_index, end_train_index, start_train_index)

    START_TEST_upd = df.loc[start_test_index, 'Datetime'].item()
    END_TRAIN = df.loc[end_train_index, 'Datetime'].item()
    START_TRAIN = df.loc[start_train_index, 'Datetime'].item()
    # df.Datetime = pd.to_datetime(df.Datetime, infer_datetime_format=True)
    (print(START_TRAIN))
    (print(END_TRAIN))
    (print(START_TEST_upd))
    (print(END_TEST))

    # df.dropna(axis=0, inplace=True)  # Удаляем наниты
    # df["Datetime"] = df["Date"]
    df.set_index("Datetime", inplace=True)
    # df = df[~df.index.duplicated(keep='first')]
    df.sort_index(ascending=True, inplace=False)
    df_duplicated = df[df.index.duplicated(keep=False)].sort_index()  # проверка дубликатов
    assert df_duplicated.shape[0] == 0, "В коде существуют дубликаты!"
    # df = df.rename(columns={"<Volume>": "Volume"})
    # del df["Time"], df["Date"]
    # del df['MACD'], df['MACDAvg'], df['MACDDiff'], df['AvgExp'], df['vwap_reset'], df['AvgExp.1']

    """Отберем данные по максе"""
    mask_train = (df.index >= START_TRAIN) & (df.index <= END_TRAIN)
    Train_df = df.loc[mask_train]
    mask_test = (df.index >= START_TEST_upd) & (df.index <= END_TEST)
    Eval_df = df.loc[mask_test]
    print(f'Датасет для поиска петтернов: {Train_df.shape}, период проверки: {Eval_df.shape}')

    """Сохраняем даты, удаляем из основынх датафрэймов"""
    Train_dates = pd.DataFrame({'Datetime': Train_df.index.values})
    Eval_dates = pd.DataFrame({'Datetime': Eval_df.index.values})
    # Eval_dates['Datetime'] = pd.to_datetime(Eval_dates['Datetime'], format='%Y%m%d%H%M%S')

    Train_df = Train_df.reset_index(drop=True)
    Eval_df = Eval_df.reset_index(drop=True)
    # del Train_df['Unnamed: 0']
    # del Eval_df['Unnamed: 0']
    # Eval_dates = Eval_dates.reset_index(drop=True)
    # Eval_dates = Eval_df.index.to_list()
    # Eval_dates_str = [str(i) for i in Eval_dates]
    print('********* Данные для обучения *********')
    print(Train_df.head())
    print('********* Данные для тестирования *********')
    print(Eval_df.head())

    return Train_df, Eval_df, Train_dates, Eval_dates
