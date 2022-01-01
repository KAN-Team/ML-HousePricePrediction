import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
from Classification import ClassificationTestPreProcessing
from Regression import RegressionTestPreProcessing


def tst_regression():
    print('\n================================================')
    print('...START: test_regression()...\n')

    # Specifying Test Data
    test_data_path = 'Regression/SavedData/Sample_Test_Data.csv'
    # Reading Test Data
    test_data = pd.read_csv(test_data_path)
    # Start Preprocessing on Test Data
    X_test, Y_test = RegressionTestPreProcessing.start_preprocessing(dataset=test_data)

    print('[The test sample length After Pre-processing: {}]\n'.format(len(X_test)))

    # Loading models from pickle
    print('...Loading models from Pickle file starts...')
    loaded_gradient_model = pickle.load(open('Regression/SavedData/regression_boosting_model.sav', 'rb'))
    loaded_forest_model = pickle.load(open('Regression/SavedData/regression_forest_model.sav', 'rb'))
    loaded_ridge_model = pickle.load(open('Regression/SavedData/regression_ridge_model.sav', 'rb'))
    print('...Loading models from Pickle file ends...\n')

    # Prediction & Performance
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Gradient Boosting Regressor Performance
    prediction = loaded_gradient_model.predict(X_test)
    print('GradientBoosting Model Mean Square Error: \t', metrics.mean_squared_error(np.asarray(Y_test), prediction))
    print("GradientBoosting Model Accuracy(%): \t\t " + str(r2_score(Y_test, prediction) * 100) + "%")
    print("-------------------")

    # Random Forest Regressor Performance
    prediction = loaded_forest_model.predict(X_test)
    print('RandomForest Model Mean Square Error:\t\t', metrics.mean_squared_error(np.asarray(Y_test), prediction))
    print("RandomForest Model Accuracy(%): \t\t\t " + str(r2_score(Y_test, prediction) * 100) + "%")
    print("-------------------")

    # Ridge Regressor Performance
    prediction = loaded_ridge_model.predict(X_test)
    print('Ridge Model Mean Square Error:\t\t\t\t', metrics.mean_squared_error(np.asarray(Y_test), prediction))
    print("Ridge Model Accuracy: \t\t\t\t\t\t " + str(r2_score(Y_test, prediction) * 100) + "%")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    print('\n...END: test_regression()...')
    print('================================================')


def tst_classification():
    print('\n================================================')
    print('...START: test_classification()...\n')

    # Specifying Test Data
    test_data_path = 'Classification/House_Data_Classification.csv'
    # Reading Test Data
    test_data = pd.read_csv(test_data_path)
    # Start Preprocessing on Test Data
    X_test, Y_test = ClassificationTestPreProcessing.start_preprocessing(dataset=test_data)

    print('[The test sample length After Pre-processing: {}]\n'.format(len(X_test)))

    print('...Loading models from Pickle file starts...\n')
    loaded_decision_tree_model = pickle.load(open('Classification/SavedData/classification_decision_tree_model.sav', 'rb'))
    loaded_poly_model = pickle.load(open('Classification/SavedData/classification_polynomial_svm_model.sav', 'rb'))
    loaded_logistic_model = pickle.load(open('Classification/SavedData/classification_logistic_model.sav', 'rb'))
    print('...Loading models from Pickle file ends...\n')

    # Prediction & Performance
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    # Performance
    predictions = loaded_decision_tree_model.predict(X_test)
    print("Accuracy For Decision Tree Model: {}%".format(accuracy_score(Y_test, predictions) * 100))
    print("MSE For Decision Tree Model: {}".format(metrics.mean_squared_error(predictions, np.asarray(Y_test))))
    print("-------------------")

    predictions = loaded_poly_model.predict(X_test)
    print("Accuracy For Polynomial SVM Model: {}%".format(accuracy_score(Y_test, predictions)*100))
    print("MSE For Polynomial SVM Model: {}".format(metrics.mean_squared_error(predictions, np.asarray(Y_test))))
    print("-------------------")

    predictions = loaded_logistic_model.predict(X_test)
    print("Accuracy For Logistic Regression Model: {}%".format(accuracy_score(Y_test, predictions) * 100))
    print("MSE For Logistic Regression Model: {}".format(metrics.mean_squared_error(predictions, np.asarray(Y_test))))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    print('\n...END: test_classification()...')
    print('================================================\n')


if __name__ == "__main__":
    choice = input("Do you want to test Regression or Classification (R, C): ")
    if choice == 'R' or choice == 'r':
        tst_regression()
    elif choice == 'C' or choice == 'c':
        tst_classification()
    else:
        print('Wrong Choice, Please Select R or C')
