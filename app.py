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

    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv")

    X_train = preprocessor.scale(X_train)
    X_test = preprocessor.scale(X_test)

    X_train_pca, X_test_pca = preprocessor.dimensionality_reduction(X_train, X_test, 0.8)
