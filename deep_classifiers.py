import preprocess
import pandas as pd
import numpy as np
import utils
from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import Dense, BatchNormalization, Activation
from keras.optimizers import Adamax

def train_deep_small_classifier(X_train, y_train, X_test, y_test, lr=0.01):
    classifier = Sequential()
    classifier.add(Dense(activation="elu", input_dim=X_train.shape[1], units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="elu", units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="elu", units=8, kernel_initializer="uniform"))
    classifier.add(Dense(activation="elu", units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
    optimizer = Adamax(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    class_weigths = utils.define_class_weight(y_train)
    print(class_weigths)
    classifier.fit(X_train, y_train, batch_size=64, epochs=300,
                   class_weight=class_weigths,
                   validation_data=(X_test, y_test),
                   callbacks=utils.callbacks_definition()
                   )
    return classifier


def train_deep_classifier(X_train, y_train, X_test, y_test, lr=0.01):
    classifier = Sequential()
    classifier.add(Dense(activation="elu", input_dim=X_train.shape[1], units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="elu", units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="elu", units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="elu", units=10, kernel_initializer="uniform"))
    classifier.add(Dense(activation="elu", units=10, kernel_initializer="uniform"))
    classifier.add(Dense(activation="elu", units=8, kernel_initializer="uniform"))
    classifier.add(Dense(activation="elu", units=8, kernel_initializer="uniform"))
    classifier.add(Dense(activation="elu", units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="elu", units=6, kernel_initializer="uniform"))
    classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))
    optimizer = Adamax(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    class_weigths = utils.define_class_weight(y_train)
    print(class_weigths)
    classifier.fit(X_train, y_train, batch_size=64, epochs=300,
                   class_weight=class_weigths,
                   validation_data=(X_test, y_test),
                   callbacks=utils.callbacks_definition()
                   )
    return classifier


def load_model_classifier(file_weights):
    model = load_model(file_weights)
    return model


def predict(classifier, X_test):
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5).astype('int32')
    y_pred = np.reshape(y_pred, y_pred.shape[0])
    return y_pred


if __name__ == '__main__':

    # load data -----------------------------------------------
    df_data = pd.read_csv('data/train.csv')

    # preprocess ----------------------------------------------
    preprocess.preprocess_all(df_data)

    # split train and test set --------------------------------
    X_train, y_train, X_test, y_test = utils.get_X_Y_train_test(df_data)
    print(y_test.value_counts(normalize=False))

    # train ---------------------------------------------------
    X_train = X_train.to_numpy(copy=True)
    y_train = y_train.to_numpy(copy=True)
    X_test = X_test.to_numpy(copy=True)
    y_test = y_test.to_numpy(copy=True)

    classifier = train_deep_classifier(X_train, y_train, X_test, y_test)
    #classifier = load_model_classifier('outputs/96_2019-06-02_17:16:58_model.hdf5')

    # Predicting the Test set results
    y_pred = predict(classifier, X_test)

    # evaluate -------------------------------------------------------
    utils.test_model(y_test, y_pred)




