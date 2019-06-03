import utils
import pandas as pd
import preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
from sklearn.model_selection import GridSearchCV

def train_random_forest(X_train, y_train):
    param_dist = {'criterion': 'gini', 'max_depth': 10, 'max_features': 'sqrt',
                  'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 9}
    #classifier = RandomForestClassifier()
    classifier = RandomForestClassifier(**param_dist)
    # parameters = {'n_estimators': [4, 6, 9],
    #               'max_features': ['log2', 'sqrt', 'auto'],
    #               'criterion': ['entropy', 'gini'],
    #               'max_depth': [2, 3, 5, 10],
    #               'min_samples_split': [2, 3, 5],
    #               'min_samples_leaf': [1, 5, 8]
    #               }
    # acc_scorer = make_scorer(accuracy_score)
    # grid_obj = GridSearchCV(classifier, parameters, scoring=acc_scorer)
    # grid_obj = grid_obj.fit(X_train, y_train)
    # best_score = grid_obj.best_score_
    # best_params = grid_obj.best_params_
    # print("Best score: {}".format(best_score))
    # print("Best params: ")
    # for param_name in sorted(best_params.keys()):
    #     print('%s: %r' % (param_name, best_params[param_name]))
    # classifier = grid_obj.best_estimator_
    classifier.fit(X_train, y_train)
    # Save model
    joblib.dump(classifier, 'outputs/random_forest.joblib')
    return classifier


def train_xgboost(X_train, y_train):
    #param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
    #param_dist = {'objective': 'binary:logistic', 'n_estimators': 2}
    param_dist = {'colsample_bytree': 0.6, 'gamma': 0, 'learning_rate': 0.2,
                  'max_depth': 10, 'min_child_weight': 1}
    classifier = XGBClassifier(**param_dist)
    # classifier = XGBClassifier()
    # parameters =   {'learning_rate': [0.001, 0.01, 0.1, 0.2, 0,3],
    #                 'max_depth': [6, 10, 15, 20],
    #                 'min_child_weight' : [ 1, 3, 5, 7 ],
    #                 'gamma': [0, 0.25, 0.5, 1.0],
    #                 'colsample_bytree': [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #                 }
    # acc_scorer = make_scorer(accuracy_score)
    # grid_obj = GridSearchCV(classifier, parameters, scoring=acc_scorer)
    # grid_obj = grid_obj.fit(X_train, y_train)
    # best_score = grid_obj.best_score_
    # best_params = grid_obj.best_params_
    # print("Best score: {}".format(best_score))
    # print("Best params: ")
    # for param_name in sorted(best_params.keys()):
    #     print('%s: %r' % (param_name, best_params[param_name]))
    # classifier = grid_obj.best_estimator_
    classifier.fit(X_train, y_train)
    joblib.dump(classifier, 'outputs/xgboost.joblib')
    return classifier

def train_logistic(X_train, y_train):
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    return classifier


def load_model(filename):
    classifier = joblib.load('{}.joblib'.format(filename))
    return classifier


def predict(classifier, X_test):
    y_pred = classifier.predict(X_test)
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
    classifier = train_random_forest(X_train, y_train)
    #classifier = train_xgboost(X_train, y_train)
    #classifier = load_model('outputs/random_forest2)
    #classifier = load_model('outputs/xgboost1')

    # Predicting the Test set results
    y_pred = predict(classifier, X_test)
    print(y_pred)

    # evaluate -------------------------------------------------------
    utils.test_model(y_test, y_pred)