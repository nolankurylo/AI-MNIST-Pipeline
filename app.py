from dataAcquisition import DataAcquisition
from dataPreprocessing import DataPreprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist



if __name__ == '__main__':
    data_acquirer = DataAcquisition()
    preprocessor = DataPreprocessing()

    X_train, X_test, y_train, y_test = data_acquirer.acquire_data()
    data_acquirer = DataAcquisition()

    X_train.to_csv("data/X_train.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)
    print(X_train.shape)

    X_train = preprocessor.scale(X_train)
    X_test = preprocessor.scale(X_test)

    X_train_pca, X_test_pca = preprocessor.dimensionality_reduction(X_train, X_test, 0.8)
    print("DONE")
