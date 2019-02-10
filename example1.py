from final.data_source import DataSource
from final.clustering_algorithm import ClusteringAlgorithm
from final.pca_helper import PCAHelper

data = DataSource()
data.read(100000)
data.preprocess()
data.show_covariance_matrix()

data = PCAHelper(data.us_census)
data.show_pca_elbow(0.415)
data.do_pca(3)
data.show_principal_component_bar()

df = data.df
ClusteringAlgorithm.show_k_means_elbow(df, 20)
clusters = ClusteringAlgorithm.k_means(df, 5)
ClusteringAlgorithm.show_clusters_3d(df, clusters)
ClusteringAlgorithm.show_silhouette(df, clusters)
