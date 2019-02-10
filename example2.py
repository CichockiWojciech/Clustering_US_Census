from final.data_source import DataSource
from final.clustering_algorithm import ClusteringAlgorithm
from final.pca_helper import PCAHelper

data = DataSource()
data.read(5000)
data.preprocess()

data = PCAHelper(data.us_census)
data.show_pca_elbow(0.95)
data.do_pca(38)
data.show_principal_component_bar()

df = data.df
ClusteringAlgorithm.show_dendrogram(df, 99)
clusters = ClusteringAlgorithm.agglomerative(df, 10)
ClusteringAlgorithm.show_clusters_3d(df, clusters)
ClusteringAlgorithm.show_silhouette(df, clusters)
