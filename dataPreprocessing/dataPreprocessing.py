import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from plotter import Plotter


class DataPreprocessing(Plotter):

    def __init__(self):
        self.num_components_before = None
        self.num_components_after = None
        self.pca = None
        Plotter.__init__(self)

    @staticmethod
    def scale(X):
        return X / 255.0

    def dimensionality_reduction(self, X_train, X_test, explained_variance):

        pca = PCA(explained_variance)
        pca.fit(X_train)

        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        self.pca = pca
        self.num_components_before = X_train.shape[1]
        self.num_components_after = X_train_pca.shape[1]

        print(f"b: {self.num_components_before}, a: {self.num_components_after}")
        return X_train_pca, X_test_pca

    def plot(self, X, X_pca, y, plot_type):

        if plot_type == 'pca_images':
            X_new = self.pca.inverse_transform(X_pca)
            self.plot_pca(X, X_new, y)
        if plot_type == 'eigenfaces':
            pass


