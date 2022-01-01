import numpy as np
import pandas as pd
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from HousePricePrediction.Classification.ClassificationDataPreProcessing import start_preprocessing


def read_data():
    # Reading house data
    dataframe = pd.read_csv('House_Data_Classification.csv')
    X = dataframe.iloc[:, :-1]
    Y = dataframe.iloc[:, -1:]

    # Splitting dataframe into train and test data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
    start_preprocessing(x_train, y_train, x_test, y_test)


#read_data()

# Loading house data (train/test data)
train_data = pd.read_csv('SavedData/Classification_Preprocessed_Train_House_Data.csv')
test_data = pd.read_csv('SavedData/Classification_Preprocessed_Test_House_Data.csv')

X_train = train_data.iloc[:, :-1]     # Features
X_test = test_data.iloc[:, :-1]
Y_train = train_data['PriceRate']     # Labels
Y_test = test_data['PriceRate']

# Apply Logistic Regression Classifier on the selected features
logistic_start_time = time.time()

C = 0.1
log_reg_model = LogisticRegression(C=C, max_iter=1000, random_state=3).fit(X_train, Y_train)
pickle.dump(log_reg_model, open('SavedData/classification_logistic_model.sav', 'wb'))

logistic_end_time = time.time()


# make prediction using the model
predictions = log_reg_model.predict(X_test)
accuracy = metrics.accuracy_score(Y_test, y_pred=predictions)*100

print("-----------------------------------------------------------")
print('Logistic Regression Classifier MSE:\t\t\t\t', metrics.mean_squared_error(np.asarray(Y_test), predictions))
print("Logistic Regression Classifier Accuracy:\t\t {}%".format(accuracy))
print("Logistic Regression Classifier Training Time:\t {} seconds.".format(logistic_end_time - logistic_start_time))
print("-----------------------------------------------------------")
