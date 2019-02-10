from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_samples
from matplotlib import cm
import numpy as np
import pandas as pd


class ClusteringAlgorithm:

    @staticmethod
    def k_means(data: pd.DataFrame, n: int):
        km = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
        clusters = km.fit_predict(data)
        return clusters

    @staticmethod
    def agglomerative(data: pd.DataFrame, n: int, affinity: str='euclidean', linkage: str='ward'):
        ac = AgglomerativeClustering(n_clusters=n, affinity=affinity, linkage=linkage)
        clusters = ac.fit_predict(data)
        return clusters

    @staticmethod
    def show_k_means_elbow(data: pd.DataFrame, max_n_cluster: int=20):
        distortions = []
        for i in range(1, max_n_cluster):
            km = KMeans(n_clusters=i,
                        init='k-means++',
                        n_init=10,
                        max_iter=300,
                        tol=1e-04,
                        random_state=0)

            km.fit(data)
            distortions.append(km.inertia_)
        plt.plot(range(1, max_n_cluster), distortions, marker='o')
        plt.title('Score Elbow for k-means Clustering')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Distortions')
        plt.show()

    @staticmethod
    def show_clusters_3d(data: pd.DataFrame, clusters):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(data['c1'],
                   data['c2'],
                   data['c3'],
                   c=clusters)

        ax.grid()
        plt.show()

    @staticmethod
    def show_dendrogram(data: pd.DataFrame, mark_dist, method='ward', metric='euclidean'):
        linked = linkage(data.values,
                         method=method,
                         metric=metric)

        plt.figure(figsize=(10, 7))
        dendrogram(linked,
                   orientation='top',
                   distance_sort='descending',
                   no_labels=True
                   )
        plt.ylabel('Euclidean distance')
        plt.xlabel('Samples')
        plt.title('US Census Dendrograms')
        plt.hlines(y=mark_dist, xmin=0, xmax=len(data.index)*10, linewidth=1, color='y')
        plt.show()

    @staticmethod
    def show_silhouette(data: pd.DataFrame, clusters, metric='euclidean'):
        silhouette_vals = silhouette_samples(data, clusters, metric=metric)
        y_ax_lower, y_ax_upper = 0, 0
        cluster_labels = np.unique(clusters)
        n_clusters = cluster_labels.shape[0]
        yticks = []
        for i, c in enumerate(cluster_labels):
            c_silhouette_vals = silhouette_vals[clusters == c]
            c_silhouette_vals.sort()
            y_ax_upper += len(c_silhouette_vals)
            color = cm.jet(float(i) / n_clusters)
            plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0,
                     edgecolor='none', color=color)

            yticks.append((y_ax_lower + y_ax_upper) / 2)
            y_ax_lower += len(c_silhouette_vals)

        silhouette_avg = np.mean(silhouette_vals)
        plt.axvline(silhouette_avg, color="red", linestyle="--")
        plt.yticks(yticks, cluster_labels + 1)
        plt.ylabel('Clusters')
        plt.xlabel('Silhouette value')
        plt.tight_layout()
        print('Silhouette ration = ' + str(silhouette_avg))
        plt.show()
