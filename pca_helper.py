import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class PCAHelper:

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.pca = None

    def do_pca(self, n: int):
        self.pca = PCA(n_components=n)
        columns = []
        for i in range(1, n+1):
            columns.append('c{}'.format(i))
        X_pca = self.pca.fit_transform(self.df)
        self.df = pd.DataFrame(data=X_pca, columns=columns)

    def show_pca_elbow(self, mark_variance: float=0.95):
        dimensions = range(1, len(self.df.columns))
        variance_ratio = []
        for n in dimensions:
            pca = PCA(n_components=n)
            pca.fit_transform(self.df)
            explained_variance = sum(pca.explained_variance_ratio_)
            print('n {} - > {}'.format(n, explained_variance))
            variance_ratio.append(explained_variance)

        plt.plot(dimensions, variance_ratio, marker='o')
        plt.xlabel('dimension number')
        plt.ylabel('explained variance ratio')
        plt.hlines(y=mark_variance, xmin=0, xmax=len(self.df.columns), linewidth=3, color='r')
        plt.show()

    def show_principal_component_bar(self):
        explained_variance_ration = self.pca.explained_variance_ratio_
        plt.bar(self.df.columns, explained_variance_ration)
        plt.ylabel('Explained variance ration ')
        plt.xlabel('Principal components')
        plt.title('Principal component analysis (PCA)')
        plt.show()
