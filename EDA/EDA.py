import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class EDA:

    def __init__(self):
        pass

    @staticmethod
    def classes_bar_plot(train_y, test_y):
        plt.figure(figsize=(20,10))
        plt.subplot(1, 2, 1)
        train_y.value_counts().plot(kind='bar')
        plt.title("Number of Training Images by Class")
        plt.xlabel('Class Name')
        plt.ylabel('# Images')

        plt.subplot(1, 2, 2)
        test_y.value_counts().plot(kind='bar')
        plt.title("Number of Testing Images by Class")
        plt.xlabel('Class Name')
        plt.ylabel('# Images')

    @staticmethod
    def display_images(X, y):
        plt.figure(figsize=(10, 20))

        for i in range(10):
            plt.subplot(5, 2, i + 1)
            img = X.iloc[np.where(y == i)[0][0]].values
            plt.imshow(img.reshape(28, 28), cmap='gray')
            plt.title(f"Image from Class: {i}")
        plt.show()

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


