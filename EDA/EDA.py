import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotter import Plotter


class EDA(Plotter):

    def __init__(self):
        Plotter.__init__(self)

    @staticmethod
    def descriptive_statistics(X):
        """ Obtains descriptive statistics regarding the dataframe X from Pandas
        :param X: input dataframe
        :return: dataframe containing X's statistics
        """
        stats = X.describe()
        return stats

    @staticmethod
    def find_constant_pixels(X):
        """ Searches the 784 pixel components of dataframe X to see if any are purely constant black or white
        :param X: input dataframe
        :return: a list containing the constant pixel features
        """
        pixel_features = X.columns
        constant_features = []
        for feature in pixel_features:
            if X[feature].eq(0).all() or X[feature].eq(255).all():
                constant_features.append(feature)
        return constant_features

    def plot(self, arg1, arg2, plot_type):
        """ Plot entry function to call Plotter child class
        :param arg1: Plotter arg based on function to call
        :param arg2: Plotter arg based on function to call
        :param plot_type: Plotter function to call
        :return: None
        """
        if plot_type == "display_images":
            self.display_images(X=arg1, y=arg2)
        if plot_type == "classes_bar_plot":
            self.classes_bar_plot(train_y=arg1, test_y=arg2)

