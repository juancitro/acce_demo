from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from keras.callbacks import CSVLogger, ModelCheckpoint
import numpy as np
from datetime import datetime
from sklearn.utils import class_weight as sk_class_weight

def callbacks_definition():
    str_date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    callbacks = []
    #tbCallBack = TensorBoard(log_dir='./outputs/logsTB/'+str_date, histogram_freq=0, write_graph=True, write_images=True)
    #callbacks.append(tbCallBack)
    #history = History()
    #callbacks.append(history)
    csv_logger = CSVLogger('outputs/{}_training.log'.format(str_date))
    callbacks.append(csv_logger)
    model_checkpoint = ModelCheckpoint(##'outputs/checkpoints/'+str_date+'_weights.{epoch:02d}-loss:{val_loss:.2f}.hdf5',
                                       'outputs/' + str_date + '_model.hdf5',
                                       verbose=0, save_best_only=True, monitor='val_loss'
                                       )
    callbacks.append(model_checkpoint)
    return callbacks


def define_class_weight(y_train):
    weights = sk_class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
    return weights

def get_value_counts_features(df, normalize=True):
    for column in df:
        print(df[column].value_counts(normalize=normalize))

def get_X_Y_train_test(df_data):
    Y = df_data['se_fue']
    X = df_data.drop(['ID','se_fue'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    return X_train, y_train, X_test, y_test


def test_model(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion matrix', cm)
    precision, recall, f_score, support = precision_recall_fscore_support(y_test, y_pred, average='micro')
    print('Precision', precision)
    print('Recall', recall)
    print('F-score', f_score)
    print('Accuracy', accuracy_score(y_test, y_pred))