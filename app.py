import pandas as pd
import pickle as pk
from dataAcquisition import DataAcquisition
from dataPreprocessing import DataPreprocessing
from modelTraining import ModelTraining
from keras.datasets import mnist


if __name__ == '__main__':
    acquirer = DataAcquisition()
    preprocessor = DataPreprocessing()

    X_train, X_test, y_train, y_test = acquirer.acquire_data()

    X_train = preprocessor.scale(X_train)
    X_test = preprocessor.scale(X_test)

    trainer = ModelTraining(X_train, y_train, X_test, y_test)

    model = trainer.neural_network()

    print("DONE")
