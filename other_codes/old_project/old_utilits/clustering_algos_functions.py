import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import (
    AffinityPropagation,
    AgglomerativeClustering,
    OPTICS,
    SpectralClustering,
    MeanShift,
    Birch,
    KMeans,
    MiniBatchKMeans,
)
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans
from sklearn.preprocessing import normalize


def KMeans_cluster_analisys(
    patterns, range_n_clusters, params, profit_value, EXTR_WINDOW, save_stat=False
):

    scaler = StandardScaler()
    patterns_std = scaler.fit_transform(
        patterns.reshape(-1, patterns.shape[1] * patterns.shape[2])
    )
    pca = PCA(n_components=2)
    patterns_PCA = pca.fit_transform(patterns_std)
    silhouette_avg_list = []
    n_clasters_list = []
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(patterns_PCA) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        KMeans.set_params(params)
        clusterer = KMeans(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(patterns_PCA)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(patterns_PCA, cluster_labels)
        silhouette_avg_list.append(silhouette_avg)
        n_clasters_list.append(n_clusters)

        """print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg, 'Algorithm name - KMeans')"""

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(patterns_PCA, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[
                cluster_labels == i
            ]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot  for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            patterns_PCA[:, 0],
            patterns_PCA[:, 1],
            marker=".",
            s=150,
            lw=0,
            alpha=0.7,
            c=colors,
            edgecolor="k",
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            f"Silhouette analysis for  KMeans clustering  with : n_clusters = {n_clusters} , PATTERN_SIZE = {patterns.shape[1]}, profit_value = {profit_value},EXTR_WINDOW = {EXTR_WINDOW} ",
            fontsize=14,
            fontweight="bold",
        )
        plt.show()
    print(
        f"Best score: {max(silhouette_avg_list)} for n_clusters: {n_clasters_list[silhouette_avg_list.index(max(silhouette_avg_list))]}"
    )
    if save_stat == True:

        return {
            "pattern_size": patterns.shape[1],
            "Algorithm": "K-Means",
            "max_sil_score": max(silhouette_avg_list),
            "n_clusters": n_clasters_list[
                silhouette_avg_list.index(max(silhouette_avg_list))
            ],
            "profit_value": profit_value,
            "EXTR_WINDOW": EXTR_WINDOW,
        }


def MiniBatchKMeans_cluster_analisys(
    patterns, range_n_clusters, params, profit_value, EXTR_WINDOW, save_stat=False
):
    scaler = StandardScaler()
    patterns_std = scaler.fit_transform(
        patterns.reshape(-1, patterns.shape[1] * patterns.shape[2])
    )
    pca = PCA(n_components=2)
    patterns_PCA = pca.fit_transform(patterns_std)
    silhouette_avg_list = []
    n_clasters_list = []
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(patterns_PCA) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        MiniBatchKMeans.set_params(params)
        clusterer = MiniBatchKMeans(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(patterns_PCA)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(patterns_std, cluster_labels)
        silhouette_avg_list.append(silhouette_avg)
        n_clasters_list.append(n_clusters)

        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
            "Algorithm name - MiniBatchKMeans",
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(patterns_std, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[
                cluster_labels == i
            ]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot  for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            patterns_PCA[:, 0],
            patterns_PCA[:, 1],
            marker=".",
            s=150,
            lw=0,
            alpha=0.7,
            c=colors,
            edgecolor="k",
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            f"Silhouette analysis for  MiniBatchKMeans clustering  with : n_clusters = {n_clusters} , PATTERN_SIZE = {patterns.shape[1]}, profit_value = {profit_value},EXTR_WINDOW = {EXTR_WINDOW} ",
            fontsize=14,
            fontweight="bold",
        )
        plt.show()
    print(
        f"Best score: {max(silhouette_avg_list)} for n_clusters: {n_clasters_list[silhouette_avg_list.index(max(silhouette_avg_list))]}"
    )

    if save_stat == True:

        return {
            "pattern_size": patterns.shape[1],
            "Algorithm": "MiniBatchKMeans",
            "max_sil_score": max(silhouette_avg_list),
            "n_clusters": n_clasters_list[
                silhouette_avg_list.index(max(silhouette_avg_list))
            ],
            "profit_value": profit_value,
            "EXTR_WINDOW": EXTR_WINDOW,
        }


def TimeSeriesKMeans_cluster_analisys(
    patterns, range_n_clusters, params, profit_value, EXTR_WINDOW, save_stat=False
):
    scaler = StandardScaler()
    patterns_std = scaler.fit_transform(
        patterns.reshape(-1, patterns.shape[1] * patterns.shape[2])
    )
    pca = PCA(n_components=2)
    patterns_PCA = pca.fit_transform(patterns_std)
    silhouette_avg_list = []
    n_clasters_list = []
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(patterns_PCA) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        TimeSeriesKMeans.set_params(params)
        clusterer = TimeSeriesKMeans(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(patterns_PCA)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(patterns_std, cluster_labels)
        silhouette_avg_list.append(silhouette_avg)
        n_clasters_list.append(n_clusters)

        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
            "Algorithm name - TimeSeriesKMeans",
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(patterns_std, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[
                cluster_labels == i
            ]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot  for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            patterns_PCA[:, 0],
            patterns_PCA[:, 1],
            marker=".",
            s=150,
            lw=0,
            alpha=0.7,
            c=colors,
            edgecolor="k",
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            f"Silhouette analysis for  TimeSeriesKMeans clustering  with : n_clusters = {n_clusters} , PATTERN_SIZE = {patterns.shape[1]}, profit_value = {profit_value},EXTR_WINDOW = {EXTR_WINDOW} ",
            fontsize=14,
            fontweight="bold",
        )
        plt.show()
    print(
        f"Best score: {max(silhouette_avg_list)} for n_clusters: {n_clasters_list[silhouette_avg_list.index(max(silhouette_avg_list))]}"
    )

    if save_stat == True:

        return {
            "pattern_size": patterns.shape[1],
            "Algorithm": "TimeSeriesKMeans",
            "max_sil_score": max(silhouette_avg_list),
            "n_clusters": n_clasters_list[
                silhouette_avg_list.index(max(silhouette_avg_list))
            ],
            "profit_value": profit_value,
            "EXTR_WINDOW": EXTR_WINDOW,
        }


def AffinityPropagation_cluster_analisys(
    patterns, params, profit_value, EXTR_WINDOW, save_stat=False
):
    scaler = StandardScaler()
    patterns_std = scaler.fit_transform(
        patterns.reshape(-1, patterns.shape[1] * patterns.shape[2])
    )
    pca = PCA(n_components=2)
    patterns_PCA = pca.fit_transform(patterns_std)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    AffinityPropagation.set_params(params)
    clusterer = AffinityPropagation()
    cluster_labels = clusterer.fit_predict(patterns_PCA)
    n_clusters = len(set(cluster_labels))

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(patterns_PCA) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(patterns_std, cluster_labels)

    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
        "Algorithm name - AffinityPropagation",
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(patterns_std, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        patterns_PCA[:, 0],
        patterns_PCA[:, 1],
        marker=".",
        s=150,
        lw=0,
        alpha=0.7,
        c=colors,
        edgecolor="k",
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        f"Silhouette analysis for  AffinityPropagation clustering  with : n_clusters = {n_clusters} , PATTERN_SIZE = {patterns.shape[1]}, profit_value = {profit_value},EXTR_WINDOW = {EXTR_WINDOW} ",
        fontsize=14,
        fontweight="bold",
    )
    plt.show()
    if save_stat == True:

        return {
            "pattern_size": patterns.shape[1],
            "Algorithm": "AffinityPropagation",
            "max_sil_score": silhouette_avg,
            "n_clusters": n_clusters,
            "profit_value": profit_value,
            "EXTR_WINDOW": EXTR_WINDOW,
        }


def AgglomerativeClustering_cluster_analisys(
    patterns, range_n_clusters, params, profit_value, EXTR_WINDOW, save_stat=False
):
    scaler = StandardScaler()
    patterns_std = scaler.fit_transform(
        patterns.reshape(-1, patterns.shape[1] * patterns.shape[2])
    )
    pca = PCA(n_components=2)
    patterns_PCA = pca.fit_transform(patterns_std)
    silhouette_avg_list = []
    n_clasters_list = []
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(patterns_PCA) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        AgglomerativeClustering.set_params(params)
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(patterns_PCA)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(patterns_std, cluster_labels)
        silhouette_avg_list.append(silhouette_avg)
        n_clasters_list.append(n_clusters)

        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
            "Algorithm name - AgglomerativeClustering",
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(patterns_std, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[
                cluster_labels == i
            ]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot  for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            patterns_PCA[:, 0],
            patterns_PCA[:, 1],
            marker=".",
            s=150,
            lw=0,
            alpha=0.7,
            c=colors,
            edgecolor="k",
        )

        # Labeling the clusters
        # centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        """ax2.scatter(
            #centers[:, 0],
            #centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k") """

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            f"Silhouette analysis for  AgglomerativeClustering clustering  with : n_clusters = {n_clusters} , PATTERN_SIZE = {patterns.shape[1]}, profit_value = {profit_value},EXTR_WINDOW = {EXTR_WINDOW} ",
            fontsize=14,
            fontweight="bold",
        )
        plt.show()
    print(
        f"Best score: {max(silhouette_avg_list)} for n_clusters: {n_clasters_list[silhouette_avg_list.index(max(silhouette_avg_list))]}"
    )

    if save_stat == True:

        return {
            "pattern_size": patterns.shape[1],
            "Algorithm": "AgglomerativeClustering",
            "max_sil_score": max(silhouette_avg_list),
            "n_clusters": n_clasters_list[
                silhouette_avg_list.index(max(silhouette_avg_list))
            ],
            "profit_value": profit_value,
            "EXTR_WINDOW": EXTR_WINDOW,
        }


def OPTICS_cluster_analisys(
    patterns, params, profit_value, EXTR_WINDOW, save_stat=False
):
    scaler = StandardScaler()
    patterns_std = scaler.fit_transform(
        patterns.reshape(-1, patterns.shape[1] * patterns.shape[2])
    )
    pca = PCA(n_components=2)
    patterns_PCA = pca.fit_transform(patterns_std)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    OPTICS.set_params(params)
    clusterer = OPTICS()
    cluster_labels = clusterer.fit_predict(patterns_std)

    n_clusters = len(set(cluster_labels))

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(patterns_PCA) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(patterns_std, cluster_labels)

    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
        "Algorithm name - OPTICS",
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(patterns_std, cluster_labels)

    y_lower = 10
    for i in range(min(cluster_labels), n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        patterns_PCA[:, 0],
        patterns_PCA[:, 1],
        marker=".",
        s=150,
        lw=0,
        alpha=0.7,
        c=colors,
        edgecolor="k",
    )

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        f"Silhouette analysis for  OPTICS clustering  with : n_clusters = {n_clusters} , PATTERN_SIZE = {patterns.shape[1]}, profit_value = {profit_value},EXTR_WINDOW = {EXTR_WINDOW} ",
        fontsize=14,
        fontweight="bold",
    )
    plt.show()

    if save_stat == True:

        return {
            "pattern_size": patterns.shape[1],
            "Algorithm": "OPTICS",
            "max_sil_score": silhouette_avg,
            "n_clusters": n_clusters,
            "profit_value": profit_value,
            "EXTR_WINDOW": EXTR_WINDOW,
        }


def SpectralClustering_cluster_analisys(
    patterns, range_n_clusters, params, profit_value, EXTR_WINDOW, save_stat=False
):
    scaler = StandardScaler()
    patterns_std = scaler.fit_transform(
        patterns.reshape(-1, patterns.shape[1] * patterns.shape[2])
    )
    pca = PCA(n_components=2)
    patterns_PCA = pca.fit_transform(patterns_std)
    silhouette_avg_list = []
    n_clasters_list = []
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(patterns_PCA) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        SpectralClustering.set_params(params)
        clusterer = SpectralClustering(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(patterns_std)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(patterns_std, cluster_labels)
        silhouette_avg_list.append(silhouette_avg)
        n_clasters_list.append(n_clusters)

        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
            "Algorithm name - SpectralClustering",
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(patterns_std, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[
                cluster_labels == i
            ]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot  for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            patterns_PCA[:, 0],
            patterns_PCA[:, 1],
            marker=".",
            s=150,
            lw=0,
            alpha=0.7,
            c=colors,
            edgecolor="k",
        )

        # Labeling the clusters
        # centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        """ax2.scatter(
            #centers[:, 0],
            #centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k") """

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            f"Silhouette analysis for  SpectralClustering clustering  with : n_clusters = {n_clusters} , PATTERN_SIZE = {patterns.shape[1]}, profit_value = {profit_value},EXTR_WINDOW = {EXTR_WINDOW} ",
            fontsize=14,
            fontweight="bold",
        )
        plt.show()
    print(
        f"Best score: {max(silhouette_avg_list)} for n_clusters: {n_clasters_list[silhouette_avg_list.index(max(silhouette_avg_list))]}"
    )

    if save_stat == True:

        return {
            "pattern_size": patterns.shape[1],
            "Algorithm": "SpectralClustering",
            "max_sil_score": max(silhouette_avg_list),
            "n_clusters": n_clasters_list[
                silhouette_avg_list.index(max(silhouette_avg_list))
            ],
            "profit_value": profit_value,
            "EXTR_WINDOW": EXTR_WINDOW,
        }


def GaussianMixture_cluster_analisys(
    patterns, range_n_clusters, params, profit_value, EXTR_WINDOW, save_stat=False
):
    scaler = StandardScaler()
    patterns_std = scaler.fit_transform(
        patterns.reshape(-1, patterns.shape[1] * patterns.shape[2])
    )
    pca = PCA(n_components=2)
    patterns_PCA = pca.fit_transform(patterns_std)
    silhouette_avg_list = []
    n_clasters_list = []
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(patterns_PCA) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        GaussianMixture.set_params(params)
        clusterer = GaussianMixture(n_components=n_clusters)
        cluster_labels = clusterer.fit_predict(patterns_std)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(patterns_std, cluster_labels)
        silhouette_avg_list.append(silhouette_avg)
        n_clasters_list.append(n_clusters)

        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
            "Algorithm name - GaussianMixture",
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(patterns_std, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[
                cluster_labels == i
            ]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot  for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            patterns_PCA[:, 0],
            patterns_PCA[:, 1],
            marker=".",
            s=150,
            lw=0,
            alpha=0.7,
            c=colors,
            edgecolor="k",
        )

        # Labeling the clusters
        # centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        """ax2.scatter(
            #centers[:, 0],
            #centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k") """

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            f"Silhouette analysis for  GaussianMixture clustering  with : n_clusters = {n_clusters} , PATTERN_SIZE = {patterns.shape[1]}, profit_value = {profit_value},EXTR_WINDOW = {EXTR_WINDOW} ",
            fontsize=14,
            fontweight="bold",
        )
        plt.show()
    print(
        f"Best score: {max(silhouette_avg_list)} for n_clusters: {n_clasters_list[silhouette_avg_list.index(max(silhouette_avg_list))]}"
    )

    if save_stat == True:

        return {
            "pattern_size": patterns.shape[1],
            "Algorithm": "GaussianMixture",
            "max_sil_score": max(silhouette_avg_list),
            "n_clusters": n_clasters_list[
                silhouette_avg_list.index(max(silhouette_avg_list))
            ],
            "profit_value": profit_value,
            "EXTR_WINDOW": EXTR_WINDOW,
        }


def MeanShift_cluster_analisys(
    patterns, params, profit_value, EXTR_WINDOW, save_stat=False
):
    scaler = StandardScaler()
    patterns_std = scaler.fit_transform(
        patterns.reshape(-1, patterns.shape[1] * patterns.shape[2])
    )
    pca = PCA(n_components=2)
    patterns_PCA = pca.fit_transform(patterns_std)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    MeanShift.set_params(params)
    clusterer = MeanShift()
    cluster_labels = clusterer.fit_predict(patterns_std)

    n_clusters = len(set(cluster_labels))

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(patterns_PCA) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(patterns_std, cluster_labels)

    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
        "Algorithm name - MeanShift",
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(patterns_std, cluster_labels)

    y_lower = 10
    for i in range(min(cluster_labels), n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        patterns_PCA[:, 0],
        patterns_PCA[:, 1],
        marker=".",
        s=150,
        lw=0,
        alpha=0.7,
        c=colors,
        edgecolor="k",
    )

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        f"Silhouette analysis for  MeanShift clustering  with : n_clusters = {n_clusters} , PATTERN_SIZE = {patterns.shape[1]}, profit_value = {profit_value},EXTR_WINDOW = {EXTR_WINDOW} ",
        fontsize=14,
        fontweight="bold",
    )
    plt.show()

    if save_stat == True:

        return {
            "pattern_size": patterns.shape[1],
            "Algorithm": "MeanShift",
            "max_sil_score": silhouette_avg,
            "n_clusters": n_clusters,
            "profit_value": profit_value,
            "EXTR_WINDOW": EXTR_WINDOW,
        }


def Birch_cluster_analisys(
    patterns, range_n_clusters, params, profit_value, EXTR_WINDOW, save_stat=False
):
    scaler = StandardScaler()
    patterns_std = scaler.fit_transform(
        patterns.reshape(-1, patterns.shape[1] * patterns.shape[2])
    )
    pca = PCA(n_components=2)
    patterns_PCA = pca.fit_transform(patterns_std)
    silhouette_avg_list = []
    n_clasters_list = []
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(patterns_PCA) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        Birch.set_params(params)
        clusterer = Birch(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(patterns_PCA)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(patterns_std, cluster_labels)
        silhouette_avg_list.append(silhouette_avg)
        n_clasters_list.append(n_clusters)

        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
            "Algorithm name - Birch",
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(patterns_std, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[
                cluster_labels == i
            ]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot  for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            patterns_PCA[:, 0],
            patterns_PCA[:, 1],
            marker=".",
            s=150,
            lw=0,
            alpha=0.7,
            c=colors,
            edgecolor="k",
        )

        # Labeling the clusters
        """centers = clusterer.subcluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")"""

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            f"Silhouette analysis for  Birch clustering  with : n_clusters = {n_clusters} , PATTERN_SIZE = {patterns.shape[1]}, profit_value = {profit_value},EXTR_WINDOW = {EXTR_WINDOW} ",
            fontsize=14,
            fontweight="bold",
        )
        plt.show()
    print(
        f"Best score: {max(silhouette_avg_list)} for n_clusters: {n_clasters_list[silhouette_avg_list.index(max(silhouette_avg_list))]}"
    )

    if save_stat == True:

        return {
            "pattern_size": patterns.shape[1],
            "Algorithm": "Birch",
            "max_sil_score": max(silhouette_avg_list),
            "n_clusters": n_clasters_list[
                silhouette_avg_list.index(max(silhouette_avg_list))
            ],
            "profit_value": profit_value,
            "EXTR_WINDOW": EXTR_WINDOW,
        }
