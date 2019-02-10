import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


class DataSource:
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt'
    path = 'USCensus1990.data.txt'

    def __init__(self):
        self.us_census = None
        self.pca = None

    def read(self, n_rows):
        path = DataSource.url
        if os.path.isfile(DataSource.path):
            path = DataSource.path

        if n_rows is None:
            self.us_census = pd.read_csv(path)
        else:
            self.us_census = pd.read_csv(path, nrows=n_rows)

    def preprocess(self):
        # remove caseid feature
        features = set(self.us_census)
        features.remove('caseid')
        us_census = pd.DataFrame(self.us_census, columns=features)

        # standard data
        from sklearn import preprocessing
        self.us_census = pd.DataFrame(preprocessing.scale(us_census), columns=us_census.columns)

    def show_covariance_matrix(self):
        corr = self.us_census.corr()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0, len(self.us_census.columns), 1)
        ax.set_xticks(ticks)
        plt.xticks(rotation=90)
        ax.set_yticks(ticks)
        ax.set_xticklabels(self.us_census.columns)
        ax.set_yticklabels(self.us_census.columns)
        plt.show()

