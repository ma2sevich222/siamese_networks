"""На вход подается уже подготовленный датафрэйм данных для проверки"""
import pandas as pd


def evdata_for_visualisation(eval_data_df, pattern_size):
    eval_df = [
        eval_data_df[i - pattern_size : i]
        for i in eval_data_df.index
        if (i - pattern_size) >= 0
    ]
    return eval_df


"""На входе numpy array паттернов, на выходе датафрейм с паттернами(требуеся для визуализаций)"""


def patterns_to_df(patterns, column_list):
    paterns_df = [pd.DataFrame(i, columns=column_list) for i in patterns]
    return paterns_df
