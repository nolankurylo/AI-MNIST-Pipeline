import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotter import Plotter


class EDA(Plotter):

    def __init__(self):
        Plotter.__init__(self)

    @staticmethod
    def descriptive_statistics(X):
        stats = X.describe()
        return stats

    @staticmethod
    def find_constant_pixels(X):
        pixel_features = X.columns
        constant_features = []
        for feature in pixel_features:
            if len(X[feature].unique()) == 1:
                constant_features.append(feature)
        return constant_features

    def plot(self, arg1, arg2, plot_type):
        if plot_type == "display_images":
            self.display_images(X=arg1, y=arg2)
        if plot_type == "classes_bar_plot":
            self.classes_bar_plot(train_y=arg1, test_y=arg2)

