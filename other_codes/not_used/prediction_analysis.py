import pandas as pd
from constants import *
from other_codes.not_used.data_load import test_data_load
from other_codes.old_project.old_utilits.triplet_func_for_train import (
    clusterted_patterns_load,
)
from utilits.visualisation_functios import class_full_analysys
from utilits.visualisation_functios import (
    triplet_pred_plotting,
    mid_dist_add_prediction_plots,
    extend_plotting,
    cos_similaryrty_extend_plotting,
    add_prediction_plots,
)

profit_value = 0.0015
pd.pandas.set_option("display.max_columns", None)
pd.set_option("expand_frame_repr", False)
pd.set_option("precision", 2)

# Основные параметры:

profit_tr_hold = [0, 5]  # показываются предсказания ниже этого значения
mid_dist_tr_hold = 0.5  # показываются предсказания ниже этого значения
cos_sim_tr_hold = 0.8  # показываются предсказания ВЫШЕ этого значения

pattern_root = "outputs/saved_patterns"
pattern_file_name = "buypat_extrw60_patsize20_profit0.0015_overlap0.npy"
source_root = "outputs"
# destination_root = 'outputs'
profit_test_with_clustering_file_name = (
    "withProfit_test_results_extrw60_patsize20_ShuffleNetV2.csv"
)
mid_dist_test_file_name = (
    "1mid_dist_test_results_extr_window40_pattern_size10_ResNet.csv"
)
cos_sim_test_file_name = "1Cos_sim_test_results_extr_window40_pattern_size10_ResNet.csv"

""" График предскзаний, когда паттерны отбирались с учетом профита и кластеризации"""
try:
    # df = add_prediction_plots(f'{source_root}/{profit_test_with_clustering_file_name}', profit_tr_hold)
    # num_patterns = pd.value_counts(df["pattern"]).to_frame()
    # list_of_trashholds = [profit_tr_hold for _ in range(num_patterns.index.shape[0])]
    # list_of_patterns = num_patterns.index.to_list()
    train_x = clusterted_patterns_load(pattern_root, pattern_file_name)
    Train_df, Eval_df, date_trane = test_data_load(SOURCE_ROOT, FILENAME)
    df = pd.read_csv(f"{source_root}/{profit_test_with_clustering_file_name}")
    extend_plotting(df, profit_tr_hold, profit_test_with_clustering_file_name)
    for cl in range(1):
        b_patterns = class_full_analysys(
            df,
            Eval_df,
            train_x,
            PATTERN_SIZE,
            EXTR_WINDOW,
            profit_value,
            OVERLAP,
            cl,
            save_best=False,
        )

except FileNotFoundError:
    print("Файл для анализа предсказаний с учетом профита отсутсвует")
    pass

""" График предскзаний, когда паттерны отбирались с учетом средней дисанции"""
try:
    df = mid_dist_add_prediction_plots(
        f"{source_root}/{mid_dist_test_file_name}", mid_dist_tr_hold
    )
    num_patterns = pd.value_counts(df["label"]).to_frame()
    list_of_trashholds = [mid_dist_tr_hold for _ in range(num_patterns.index.shape[0])]
    list_of_patterns = num_patterns.index.to_list()
    triplet_pred_plotting(
        df, list_of_trashholds, list_of_patterns, mid_dist_test_file_name
    )
except FileNotFoundError:
    print(
        "Файл для анализа предсказаний с выбором паттернов по средней дистанции отсутствует"
    )
    pass

""" График предскзаний, Cosine similarity"""
try:
    df = add_prediction_plots(
        f"{source_root}/{cos_sim_test_file_name}", cos_sim_tr_hold
    )
    num_patterns = pd.value_counts(df["pattern"]).to_frame()
    list_of_trashholds = [cos_sim_tr_hold for _ in range(num_patterns.index.shape[0])]
    list_of_patterns = num_patterns.index.to_list()
    cos_similaryrty_extend_plotting(
        df, list_of_trashholds, list_of_patterns, cos_sim_test_file_name
    )
except FileNotFoundError:
    print(
        "Файл для анализа предсказаний с функцией ошибки Cosine similarity отсутствует"
    )
    pass
