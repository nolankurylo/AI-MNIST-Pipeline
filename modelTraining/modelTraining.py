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
        print(self.X_train.shape)
        X_train = self.X_train.values.reshape(-1,28,28,1)
        X_test = self.X_test.values.reshape(-1,28,28,1)

        y_train = to_categorical(self.y_train, 10)
        y_test = to_categorical(self.y_test, 10)
        print(X_train.shape)
        model = Sequential([
            InputLayer(input_shape=(28, 28, 1)),  # input layer
            Conv2D(16, (3,3), activation='relu'),  # convolutional layer
            MaxPooling2D((2,2)),  # pooling layer
            Conv2D(32, (3, 3), activation='relu'),  # convolutional layer
            MaxPooling2D((2, 2)),  # pooling layer
            # Conv2D(64, (3, 3), activation='relu'),  # convolutional layer
            # MaxPooling2D((2, 2)),  # pooling layer
            # Conv2D(64, (3, 3), activation='relu'),  # convolutional layer
            # MaxPooling2D((2, 2)),  # pooling layer
            # Conv2D(64, (3, 3), activation='relu'),  # convolutional layer
            MaxPooling2D((2, 2)),  # pooling layer
            Flatten(),  # reshape
            Dense(512, activation='relu'),  # fully connected layer
            # Dense(64, activation='relu'),  # fully connected layer


            Dense(10, activation="softmax")  # output layer
        ])
        print(model.summary())

        opt = Adam(lr=3e-5)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64)
        return model

