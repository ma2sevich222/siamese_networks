import pandas as pd
from utilits.visualisation_functios import triplet_pred_plotting

source_root = "outputs"
file_name = "mid_dist_test_results_extr_window40_pattern_size10_ResNet.csv"
df = pd.read_csv(f"{source_root}/{file_name}")
num_patterns = pd.value_counts(df["label"]).to_frame()
list_of_trashholds = [0.00008 for _ in range(num_patterns.index.shape[0])]
list_of_patterns = num_patterns.index.to_list()
triplet_pred_plotting(df, list_of_trashholds, list_of_patterns, "test")

triplet_pred_plotting(df, [0.00008], [1], "test")
