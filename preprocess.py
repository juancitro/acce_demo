import pandas as pd
from sklearn import preprocessing

"""
ID,
nivel_de_satisfaccion,
ultima_evaluacion,
cantidad_proyectos,
promedio_horas_mensuales_trabajadas,
años_en_la_empresa,
tuvo_un_accidente_laboral,
promociones_ultimos_5_anios,
area,
salario,
se_fue
"""

def preprocess_features(df):
    #df.drop(['ID'], axis=1)
    encode_all_categorical_features(df)
    normalize_all(df)

def preprocess_all(df):
    preprocess_features(df)
    custom_encode_numeric(df, 'se_fue', {'si': 1, 'no': 0})

def encode_all_categorical_features(df):
    encode_feature_numeric(df, 'area')
    custom_encode_numeric(df, 'salario', {'bajo': 0, 'medio': 0.5, 'alto': 1})

def normalize_all(df):
    #for column in df:
    #    normalize_feature(df, column)
    normalize_feature(df, 'cantidad_proyectos')
    normalize_feature(df, 'promedio_horas_mensuales_trabajadas')
    normalize_feature(df, 'años_en_la_empresa')
    normalize_feature(df, 'area')


def encode_feature_numeric(df, feature_name):
    le = preprocessing.LabelEncoder()
    le = le.fit(df[feature_name])
    df[feature_name] = le.transform(df[feature_name])
    return le

def normalize_feature(df, feature_name):
    min_max_scaler = preprocessing.MinMaxScaler()
    df[[feature_name]] = min_max_scaler.fit_transform(df[[feature_name]])

def custom_encode_numeric(df, feature_name, dict_matching):
    df[feature_name]=df[feature_name].replace(dict_matching)


