from dataAcquisition import DataAcquisition
from dataPreprocessing import DataPreprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    data_acquirer = DataAcquisition()
    preprocessor = DataPreprocessing()


    # X_train, X_test, y_train, y_test = data_acquirer.acquire_data()
    data_acquirer = DataAcquisition()

    X_train = pd.read_csv("data/train_X.csv")
    y_train = pd.read_csv("data/train_y.csv")
    X_test = pd.read_csv("data/test_X.csv")
    y_test = pd.read_csv("data/test_y.csv")

    print("Scale")
    X_train = preprocessor.scale(X_train)
    X_test = preprocessor.scale(X_test)

    X_train_pca, X_test_pca = preprocessor.dimensionality_reduction(X_train, X_test, 0.8)
    preprocessor.plot(X_train, X_train_pca, y_train, plot_type='pca_images')

    # img = new_X.iloc[0].values
    # img2 = X_train.iloc[0].values
    # plt.imshow(img.reshape(28, 28), cmap='gray')
    # plt.imshow(img2.reshape(28, 28), cmap='gray')
    # plt.show()