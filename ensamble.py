import deep_classifiers
import sk_learn_classifiers
import pandas as pd
import preprocess
import utils

def get_X_predictions_classifiers(deep_classifier, rf_classifier, xgboost_classifier, X):
    deep_classifier_predictions_train = deep_classifiers.predict(deep_classifier, X)
    rf_classifier_predictions_train = sk_learn_classifiers.predict(rf_classifier, X)
    xgboost_classifier_predictions_train = sk_learn_classifiers.predict(xgboost_classifier, X)
    df_X = pd.DataFrame({
        'd_class': deep_classifier_predictions_train,
        'rf_class': rf_classifier_predictions_train,
        'xg_class': xgboost_classifier_predictions_train
    })
    return df_X


if __name__ == '__main__':
    # load data -----------------------------------------------
    df_data = pd.read_csv('data/train.csv')

    # preprocess ----------------------------------------------
    preprocess.preprocess_all(df_data)

    # split train and test set --------------------------------
    X_train, y_train, X_test, y_test = utils.get_X_Y_train_test(df_data)
    print(y_test.value_counts(normalize=False))

    deep_classifier = deep_classifiers.load_model_classifier('outputs/96_2019-06-02_17:16:58_model.hdf5')
    #rf_classifier = sk_learn_classifiers.train_random_forest(X_train, y_train)
    rf_classifier = sk_learn_classifiers.load_model('outputs/random_forest2')
    #xgboost_classifier = sk_learn_classifiers.train_xgboost(X_train, y_train)
    xgboost_classifier = sk_learn_classifiers.load_model('outputs/xgboost1')
    df_X_train_preds_classifiers = get_X_predictions_classifiers(deep_classifier, rf_classifier, xgboost_classifier, X_train)
    df_X_test_preds_classifiers = get_X_predictions_classifiers(deep_classifier, rf_classifier, xgboost_classifier, X_test)

    X_train_ensemble = df_X_train_preds_classifiers.to_numpy(copy=True)
    y_train = y_train.to_numpy(copy=True)
    X_test_ensemble = df_X_test_preds_classifiers.to_numpy(copy=True)
    y_test = y_test.to_numpy(copy=True)

    # train ---------------------------------------------------
    #ensemble_classifier = data_science.train_deep_classifier(X_train_ensemble, y_train, X_test_ensemble, y_test, lr=0.0001)
    #ensemble_classifier = data_science.train_deep_small_classifier(X_train_ensemble, y_train, X_test_ensemble, y_test, lr=0.0001)
    ensemble_classifier = deep_classifiers.load_model_classifier('outputs/2019-06-03_14:31:16_model.hdf5')
    y_pred_ensemble = deep_classifiers.predict(ensemble_classifier, X_test_ensemble)

    y_pred_deep = deep_classifiers.predict(deep_classifier, X_test)
    y_pred_rf = sk_learn_classifiers.predict(rf_classifier, X_test)
    y_pred_xgboost = sk_learn_classifiers.predict(xgboost_classifier, X_test)

    utils.test_model(y_test, y_pred_deep)
    utils.test_model(y_test, y_pred_rf)
    utils.test_model(y_test, y_pred_xgboost)
    utils.test_model(y_test, y_pred_ensemble)


