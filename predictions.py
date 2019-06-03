import deep_classifiers
import sk_learn_classifiers
import pandas as pd
import preprocess
import ensamble
import numpy as np

def predict(df_data):
    preprocess.preprocess_features(df_data)
    df_data = df_data.drop(['ID', 'se_fue'], axis=1)
    deep_classifier = deep_classifiers.load_model_classifier('outputs/96_2019-06-02_17:16:58_model.hdf5')
    rf_classifier = sk_learn_classifiers.load_model('outputs/random_forest2')
    xgboost_classifier = sk_learn_classifiers.load_model('outputs/xgboost1')
    df_X_preds = ensamble.get_X_predictions_classifiers(deep_classifier, rf_classifier, xgboost_classifier, df_data)
    X_inputs_ens = df_X_preds.to_numpy(copy=True)
    ensemble_classifier = deep_classifiers.load_model_classifier('outputs/2019-06-03_14:31:16_model.hdf5')
    y_pred_ensemble = deep_classifiers.predict(ensemble_classifier, X_inputs_ens)
    return y_pred_ensemble


if __name__ == '__main__':
    df_data = pd.read_csv('data/test.csv')
    print(df_data.head())
    y_pred = predict(df_data)
    print(np.unique(y_pred, return_counts=True))
    df_test = pd.DataFrame({
        'ID': df_data['ID'].astype('int32'),
        'se_fue': y_pred
    })
    df_test.to_csv('data/submission_test.csv',index=False)
    print(df_test.head())