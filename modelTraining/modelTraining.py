import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, InputLayer, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import xgboost as xgb


class ModelTraining:

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def xgboost(self):
        model = xgb.XGBClassifier()
        model.fit(self.X_train, self.y_train, verbose=True)
        return model

    def neural_network(self):
        print(self.X_train.shape)
        X_train = self.X_train.reshape(-1,28,28,1)
        X_test = self.X_test.reshape(-1,28,28,1)

        y_train = to_categorical(self.y_train, 10)
        y_test = to_categorical(self.y_test, 10)
        print(X_train.shape)
        model = Sequential([
            InputLayer(input_shape=(28, 28, 1)),  # input layer
            Conv2D(16, (3,3), activation='relu'),  # convolutional layer
            MaxPooling2D((2,2)),  # pooling layer
            # Conv2D(32, (3, 3), activation='relu'),  # convolutional layer
            # MaxPooling2D((2, 2)),  # pooling layer
            # Conv2D(64, (3, 3), activation='relu'),  # convolutional layer
            # MaxPooling2D((2, 2)),  # pooling layer
            # Conv2D(64, (3, 3), activation='relu'),  # convolutional layer
            # MaxPooling2D((2, 2)),  # pooling layer
            # Conv2D(64, (3, 3), activation='relu'),  # convolutional layer
            # MaxPooling2D((2, 2)),  # pooling layer
            Flatten(),  # reshape
            Dense(100, activation='relu', kernel_initializer='he_uniform'),  # fully connected layer
            # Dense(64, activation='relu'),  # fully connected layer


            Dense(10, activation="softmax")  # output layer
        ])
        print(model.summary())

        opt = Adam(lr=3e-5)
        # opt = SGD(lr=0.01, momentum=0.9)

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=64, callbacks=[es])
        return model




    def support_vector_machine(self):
        # y_train = to_categorical(self.y_train, 10)
        # y_test = to_categorical(self.y_test, 10)
        y_train = self.y_train
        print(y_train)
        hyperparams = dict(
            C=[345],
            gamma=[0.1, 0.001]
        )

        search = GridSearchCV(estimator=svm.SVC(kernel='linear'), param_grid=hyperparams, verbose=10)
        search.fit(self.X_train, y_train)
        print("doone")
        model = search.best_estimator_
        return model

