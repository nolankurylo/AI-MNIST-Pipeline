import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from utils import Utils


class DataPreprocessing(Utils):

    def __init__(self):
        self.num_components_before = None
        self.num_components_after = None
        self.pca = None
        Utils.__init__(self)

    @staticmethod
    def scale(X):
        """ Scales every pixel value by the maximum (255)
        :param X: input dataframe
        :return: scaled dataframe
        """
        return X / 255.0

    def dimensionality_reduction(self, X_train, X_test, explained_variance=0.8):
        """ Given a desired explained_variance ratio, finds the number of principle components to explain that variance
        and returns the dimensionally reduced training and testing dataframes: Principle Component Analysis
        :param X_train: training set of 784 components
        :param X_test: testing set of 784 components
        :param explained_variance: desired ratio that the remaining components should still represent after
        dimensionality reduction
        :return: PCA transformed training and testing sets of 784 - n components
        """

        pca = PCA(explained_variance)
        pca.fit(X_train)

        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)

        self.pca = pca
        self.num_components_before = X_train.shape[1]
        self.num_components_after = X_train_pca.shape[1]

        return X_train_pca, X_test_pca

    def plot(self, *args, plot_type):
        """ Plot entry function to call Plotter child class
        :param X: original X
        :param X_pca: dimensionally reduced X
        :param plot_type: Plotter function to call
        :return: None
        """
        if plot_type == 'pca_images':
            X_new = self.pca.inverse_transform(args[1])
            self.plot_pca(args[0], X_new, args[2])
        if plot_type == 'eigenfaces':
            self.plot_eigenvalues(self.pca)


