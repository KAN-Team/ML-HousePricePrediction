import numpy as np
import pandas as pd
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from ClassificationDataProcessing import start_preprocessing


# load the data
saved_model_filename = 'classification_saved_logistic_model.sav'
start_preprocessing(dataset_name='House_Data_Classification.csv')


data = pd.read_csv('Classification_Preprocessed_House_Data.csv')
X = data.iloc[:, 0:2]     # Features
Y = data['PriceRate']     # Label


logistic_start_time = time.time()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
C = 0.5
log_reg_model = LogisticRegression(C=C, max_iter=1000, random_state=3).fit(X_train, y_train)
pickle.dump(log_reg_model, open(saved_model_filename, 'wb'))

logistic_end_time = time.time()


# make prediction using the model
predictions = log_reg_model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred=predictions)*100

print("-----------------------------------------------------------")
print('Logistic Regression Classifier MSE:\t\t\t\t', metrics.mean_squared_error(np.asarray(y_test), predictions))
print("Logistic Regression Classifier Accuracy:\t\t {}%".format(accuracy))
print("Logistic Regression Classifier Training Time:\t {} seconds.".format(logistic_end_time - logistic_start_time))
print("-----------------------------------------------------------")
