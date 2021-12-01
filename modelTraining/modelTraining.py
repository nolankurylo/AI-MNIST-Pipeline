import tensorflow as tf
from tensorflow.keras.layers import Dropout, BatchNormalization, Input, Conv2D, Flatten, Dense, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import xgboost as xgb


class ModelTraining:
    EPOCHS = 100
    BATCH_SIZE = 512

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def xgboost(self):
        """
        XGBoost classifier that was used for initial model testing
        :return:
        """
        model = xgb.XGBClassifier()
        model.fit(self.X_train, self.y_train, verbose=True)
        return model

    def neural_network(self):
        """ Single input deep neural network with batch normalization and dropout for overfitting
        :return:
        """

        X_train = self.X_train.values.reshape(-1, 28, 28, 1)
        X_test = self.X_test.values.reshape(-1, 28, 28, 1)

        y_train = to_categorical(self.y_train, 10)
        y_test = to_categorical(self.y_test, 10)

        model_input = Input(shape=(28, 28, 1), name="input_layer")
        model_output = Conv2D(32, (3, 3), activation='relu', name="conv_layer")(model_input)
        model_output = MaxPooling2D((2, 2), name="pooling_layer")(model_output)
        model_output = Flatten(name="flatten_layer")(model_output)  # reshape
        model_output = Dense(512, activation='relu', name="dense1_layer")(model_output)  # fully connected layer
        model_output = Dropout(0.5, name="dropout1_layer")(model_output)
        model_output = BatchNormalization(name="bn1_layer")(model_output)
        model_output = Dense(256, activation='relu', name="dense2_layer")(model_output)  # fully connected layer
        model_output = Dropout(0.5, name="dropout2_layer")(model_output)
        model_output = BatchNormalization(name="bn2_layer")(model_output)
        model_output = Dense(10, activation="softmax", name="dense3_layer")(model_output)
        model = Model(inputs=[model_input], outputs=[model_output], name="final_model")

        initial_learning_rate = 0.0001
        final_learning_rate = 0.000001
        learning_rate_decay_factor = (final_learning_rate / initial_learning_rate) ** (1 / self.EPOCHS)
        steps_per_epoch = int(self.X_train.shape[0] / self.BATCH_SIZE)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=steps_per_epoch,
            decay_rate=learning_rate_decay_factor,
            staircase=True)

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40)
        mc = ModelCheckpoint('model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

        optimizer = Adam(learning_rate=lr_schedule)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(
            x=X_train,
            y=y_train,
            validation_data=(X_test, y_test),
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            callbacks=[es, mc]
        )

        return model
