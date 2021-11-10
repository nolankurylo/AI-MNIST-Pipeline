import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from keras.layers import Dense, Dropout, BatchNormalization, InputLayer, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, Adam


class ModelTraining:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def neural_network(self):
        X_train = self.X_train.values.reshape(-1,28,28,1)
        print(X_train.shape)
        y_train = to_categorical(self.y_train, 10)

        model = Sequential([
            # InputLayer(input_shape=(28, 28, 1)),  # input layer
            # Conv2D(32, (3,3), activation='relu', kernel_initializer='he_uniform'),  # convolutional layer
            # MaxPooling2D((2,2)),  # pooling layer
            Flatten(input_shape=(28, 28, 1)),  # reshape
            Dense(128, activation='relu'),  # fully connected layer
            Dense(10, activation="softmax")  # output layer
        ])

        opt = Adam(lr=0.01)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=256)
        return model

