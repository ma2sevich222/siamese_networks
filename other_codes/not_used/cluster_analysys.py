from constants import *
from utilits.clustering_algos_functions import SpectralClustering_cluster_analisys

# from utilits.data_load import data_load_OHLC
from other_codes.old_project.old_utilits.triplet_func_for_train import (
    get_patterns_with_profit,
)
from other_codes.not_used.data_load import test_data_load

""" Основные параметры """
profit_value = 0.0015


"""indices = [
    i for i, x in enumerate(FILENAME) if x == "_"
]  # находим индексы вхождения '_'
ticker = FILENAME[: indices[0]]"""

"""Загрузка и подготовка данных"""
Train_df, Eval_df, train_dates = test_data_load(SOURCE_ROOT, FILENAME)
patterns, after, indexes_min = get_patterns_with_profit(
    Train_df, profit_value, EXTR_WINDOW, PATTERN_SIZE, OVERLAP, train_dates
)
range_n_clusters = [i for i in range(2, 13)]

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# KMeans_params = {'n_init': 10, 'max_iter': 300}
# KMeans_cluster_analisys(patterns, range_n_clusters, KMeans_params, profit_value, EXTR_WINDOW)

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html
# KMeans_minibatch_params = {'n_init': 10, 'max_iter': 100, 'batch_size': 10}
# MiniBatchKMeans_cluster_analisys(patterns, range_n_clusters, KMeans_minibatch_params, profit_value, EXTR_WINDOW)

# https://tslearn.readthedocs.io/en/stable/gen_modules/clustering/tslearn.clustering.TimeSeriesKMeans.html
TimeSeriesKMeans_params = {
    "n_init": 15,
    "max_iter_barycenter": 10,
    "n_jobs": 25,
    "metric": "softdtw",
}
# TimeSeriesKMeans_cluster_analisys(patterns, range_n_clusters, TimeSeriesKMeans_params, profit_value, EXTR_WINDOW)

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
AgglomerativeClustering_params = {"linkage": "average"}
# AgglomerativeClustering_cluster_analisys(patterns, range_n_clusters, AgglomerativeClustering_params, profit_value, EXTR_WINDOW)

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html
AffinityPropagation_params = {"damping": 1, "n_iter_": 800, "random_state": 10}
# AffinityPropagation_cluster_analisys(patterns, AffinityPropagation_params, profit_value, EXTR_WINDOW)

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html
OPTICS_params = {"min_samples": 10, "max_eps": 0.5}
# OPTICS_cluster_analisys(patterns, OPTICS_params, profit_value, EXTR_WINDOW)

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html
SpectralClustering_params = {"n_init": 10, "n_jobs": 10}
SpectralClustering_cluster_analisys(
    patterns, range_n_clusters, SpectralClustering_params, profit_value, EXTR_WINDOW
)

# https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
# GaussianMixture_params = {'n_init': 100, 'init_params': 'kmeans','covariance_type': 'tied'}
# GaussianMixture_cluster_analisys(patterns, range_n_clusters, GaussianMixture_params, profit_value, EXTR_WINDOW)

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
# MeanShift_params = {'n_jobs': 10, 'max_iter': 800, 'bin_seeding':False,'bandwidth':0.001}
# MeanShift_cluster_analisys(patterns, MeanShift_params, profit_value, EXTR_WINDOW)

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html
Birch_params = {"threshold": 0.5}
# Birch_cluster_analisys(patterns, range_n_clusters, Birch_params, profit_value, EXTR_WINDOW)


"""list_of_pattern_sizes = [i for i in range(5, 50, 5)] # задаем размеры паттернов
range_n_clasters = [i for i in range(2, 11)]
final_results =[]
for i in list_of_pattern_sizes:
    print(i)
    patterns = get_patterns_with_profit(Train_df, profit_value, EXTR_WINDOW, i)
    best_case = TimeSeriesKMeans_cluster_analisys(patterns, range_n_clusters, TimeSeriesKMeans_params
                     , profit_value, EXTR_WINDOW, save_stat=False)  #  save_stat для сохранения данных, по умолчанию false просто выводит график

    final_results.append(best_case)


final_results_df = pd.DataFrame(final_results)
sorted = final_results_df.sort_values(by='max_sil_score', ascending=False)
print('Отсортированная таблица результатов проверки кластеризации паттернов с разным размером PATTERN_SIZE')
display(sorted)"""
