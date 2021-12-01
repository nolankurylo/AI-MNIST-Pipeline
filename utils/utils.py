import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class Utils:

    def __init__(self):
        pass

    def plot_pca(self, X, X_new, y):
        plt.figure(figsize=(10, 20))
        for i in range(0, 10):  # print 3 random images and compare them to the PCA reduced images
            idx = np.where(y == i)[0][0]
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 2, 1)
            plt.title("Original Image - " + str(self.num_components_before) + " dimensions")
            X_image = X.iloc[idx].values.reshape(28, 28)
            plt.imshow(X_image, cmap='gray')
            plt.subplot(1, 2, 2)
            plt.title("Projected Image - " + str(self.num_components_after) + " dimensions")
            X_image_pca = X_new[idx].reshape(28, 28)
            plt.imshow(X_image_pca, cmap='gray')
            plt.suptitle(f"PCA Transformation for image from Class: {i}")
            plt.show()

    def display_images(self, X, y):
        plt.figure(figsize=(10, 20))

        for i in range(10):
            plt.subplot(5, 2, i + 1)
            img = X.iloc[np.where(y == i)[0][0]].values
            plt.imshow(img.reshape(28, 28), cmap='gray')
            plt.title(f"Image from Class: {i}")
        plt.show()

    def classes_bar_plot(self, y_train, y_test):
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        y_train.value_counts().plot(kind='bar')
        plt.title("Number of Training Images by Class")
        plt.xlabel('Class Name')
        plt.ylabel('# Images')
        plt.subplot(1, 2, 2)
        y_test.value_counts().plot(kind='bar')
        plt.title("Number of Testing Images by Class")
        plt.xlabel('Class Name')
        plt.ylabel('# Images')

    @staticmethod
    def plot_eigenvalues(pca):
        plt.figure(figsize=(10, 20))

        for i in range(5):
            plt.subplot(3, 2, i + 1)
            plt.suptitle("Top 5 Principal Components (Eigenfaces)")
            plt.title("Principal Component: " + str(i + 1))
            plt.imshow(pca.components_[i].reshape(28, 28), cmap='bone')
        plt.show()
