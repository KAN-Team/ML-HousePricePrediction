import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from HousePricePrediction.Classification import ClassificationDataPreProcessing
from HousePricePrediction.Regression import RegressionDataPreProcessing


def test_regression():
    loaded_gradient_model_filename = 'regression_saved_gradient_model.sav'
    loaded_ridge_model_filename = 'regression_saved_ridge_model.sav'
    dataset_name = 'House_Data_Regression.csv'
    RegressionDataPreProcessing.start_preprocessing(dataset_name=dataset_name)

    print('\n================================================')
    print('...Regression Script starts...\n')

    df = pd.read_csv('Regression_Preprocessed_House_Data.csv')
    X = df.iloc[:, 0:5]  # Features
    Y = df.iloc[:, -1]  # Label
    print('The test sample length After Pre-processing: {}'.format(len(X)))

    print('...Loading models from Pickle file starts...\n')
    loaded_gradient_model = pickle.load(open(loaded_gradient_model_filename, 'rb'))
    loaded_ridge_model = pickle.load(open(loaded_ridge_model_filename, 'rb'))
    print('...Loading models from Pickle file ends...\n')

    predictions = loaded_gradient_model.predict(X)
    print("Accuracy For GradientBoosting Model: {}%".format(r2_score(Y, predictions)*100))
    print("MSE For GradientBoosting Model: {}".format(metrics.mean_squared_error(predictions, np.asarray(Y))))
    print("-------------------")

    predictions = loaded_ridge_model.predict(X)
    print("Accuracy For Ridge Model: {}%".format(r2_score(Y, predictions)*100))
    print("MSE For Ridge Model: {}".format(metrics.mean_squared_error(predictions, np.asarray(Y))))

    print('\n...Regression Script ends...')
    print('================================================\n')


def test_classification():
    loaded_poly_model_filename = 'classification_saved_poly_model.sav'
    loaded_decision_tree_model_filename = 'classification_saved_decisionTree_model.sav'
    loaded_logistic_model_filename = 'classification_saved_logistic_model.sav'
    dataset_name = 'House_Data_Classification.csv'
    ClassificationDataPreProcessing.start_preprocessing(dataset_name=dataset_name)

    print('\n================================================')
    print('...Classification Script starts...\n')

    df = pd.read_csv('Classification_Preprocessed_House_Data.csv')
    X = df.iloc[:, :2]  # we only take the first two features.
    Y = df.iloc[:, -1]
    print('The test sample length After Pre-processing: {}'.format(len(X)))

    print('...Loading models from Pickle file starts...\n')
    loaded_poly_model = pickle.load(open(loaded_poly_model_filename, 'rb'))
    loaded_decision_tree_model = pickle.load(open(loaded_decision_tree_model_filename, 'rb'))
    loaded_logistic_model = pickle.load(open(loaded_logistic_model_filename, 'rb'))
    print('...Loading models from Pickle file ends...\n')

    predictions = loaded_poly_model.predict(X)
    print("Accuracy For Polynomial SVM Model: {}%".format(accuracy_score(Y, predictions)*100))
    print("MSE For Polynomial SVM Model: {}".format(metrics.mean_squared_error(predictions, np.asarray(Y))))
    print("-------------------")

    predictions = loaded_decision_tree_model.predict(X)
    print("Accuracy For Decision Tree Model: {}%".format(accuracy_score(Y, predictions) * 100))
    print("MSE For Decision Tree Model: {}".format(metrics.mean_squared_error(predictions, np.asarray(Y))))
    print("-------------------")

    predictions = loaded_logistic_model.predict(X)
    print("Accuracy For Logistic Regression Model: {}%".format(accuracy_score(Y, predictions) * 100))
    print("MSE For Logistic Regression Model: {}".format(metrics.mean_squared_error(predictions, np.asarray(Y))))

    print('\n...Classification Script ends...')
    print('================================================\n')


if __name__ == "__main__":
    choice = input("Do you want to test Regression or Classification (R, C): ")
    if choice == 'R' or choice == 'r':
        test_regression()
    elif choice == 'C' or choice == 'c':
        test_classification()
    else:
        print('Wrong Choice, Please Select R or C')
