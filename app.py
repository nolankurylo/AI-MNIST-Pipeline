from dataAcquisition import DataAcquisition
from dataPreprocessing import DataPreprocessing
import pandas as pd
import numpy as np



if __name__ == '__main__':
    data_acquirer = DataAcquisition()
    preprocessor = DataPreprocessing()

    # X_train, X_test, y_train, y_test = data_acquirer.acquire_data()
    data_acquirer = DataAcquisition()

    X_train = pd.read_csv("data/train_X.csv")
    y_train = pd.read_csv("data/train_y.csv")
    X_test = pd.read_csv("data/test_X.csv")
    y_test = pd.read_csv("data/test_y.csv")


    constant_features = preprocessor.find_constant_pixels(X_train)
    print(constant_features)