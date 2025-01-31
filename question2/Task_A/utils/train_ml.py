# import multiple classifiers from sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

import logging

import os
import joblib


def train_ml_model(data: pd.DataFrame, target: pd.Series, model_name: str):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    # initialize the model
    if model_name == 'rf':
        model = RandomForestClassifier(n_estimators=100, max_depth=20)
    elif model_name == 'svm':
        model = SVC(kernel='poly', degree=10)
    elif model_name == 'dt':
        model = DecisionTreeClassifier(max_depth=20)
    elif model_name == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)
    else:
        raise ValueError('Invalid model name')

    # train the model
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)*100

    y_pred_train = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)*100

    return model, train_accuracy, test_accuracy


def load_data(data_path: str):
    data_csv = pd.read_csv(data_path)
    target = data_csv.pop('label')

    # the data in first column of data_csv are embeddings in form of string of a list
    # convert the string to floats and spread in columns
    # data_csv['embedding'] = data_csv['embedding'].apply(lambda x: x[1:-1].split()).apply(pd.to_numeric)
    # data = data_csv.apply(pd.to_numeric)
    # data_csv['embedding'] = data_csv['embedding'].apply(ast.literal_eval)
    # embeddings_df = pd.DataFrame(data_csv['embedding'].tolist())
    # print(embeddings_df.info())

    return data_csv, target


def main(window_type, model_name):
    logging.info(f'Training {model_name} model on embeddings for window: {window_type}')

    logging.info('Loading data')
    data, target = load_data(f'train_data/embeddings/{window_type}_embeddings.csv')
    logging.info('Data loaded')

    logging.info('Training model')
    model, train_accuracy, test_accuracy = train_ml_model(data, target, model_name)
    logging.info('Model trained')

    logging.info(f'Training accuracy: {train_accuracy}, Test accuracy: {test_accuracy}')

    logging.info('Saving model')
    if not os.path.exists(f'models/{model_name}'):
        os.makedirs(f'models/{model_name}')
    joblib.dump(model, f'models/{model_name}/{window_type}.pkl')    
    logging.info('Model saved successfully\n')


# if __name__ == '__main__':
#     main()