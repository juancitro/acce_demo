from sklearn.model_selection import train_test_split
import preprocess
import pandas as pd
import numpy as np
import utils
from matplotlib import pyplot as plt


def plot_scatter(df, f1, f2):
    plt.scatter(df[f1], df[f2], c='r')
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.savefig('figs/{}-{}.png'.format(f1,f2))
    plt.show()

def plot_scatter_all(df):
    for index1, f1 in enumerate(df):
        for index2, f2 in enumerate(df):
            if index2>index1:
                plot_scatter(df,f1,f2)


if __name__ == '__main__':
    df_data = pd.read_csv('data/train.csv')
    #preprocess.encode_all_categorical_features(df_data)
    print(df_data.head())
    preprocess.preprocess_all(df_data)
    #get_value_counts_features(df_data)
    print(df_data.head())

    plot_scatter_all(df_data)

    #X_train, y_train, X_test, y_test = utils.get_X_Y_train_test(df_data)
    #print(y_train.value_counts(normalize=True))
    #print(y_test.value_counts(normalize=True))

    #print("\nX_train:\n")
    #print(X_train.head())
    # print(X_train.shape)
    # print("\nX_test:\n")
    # print(X_test.head())
    # print(X_test.shape)