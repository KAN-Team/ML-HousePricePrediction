import time
import numpy as np
import pandas as pd
import pickle
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from RegressionDataPreProcessing import start_preprocessing


def read_data():
    # Reading house data
    dataframe = pd.read_csv('House_Data_Regression.csv')
    X = dataframe.iloc[:, :-1]
    Y = dataframe.iloc[:, -1]

    # Splitting dataframe into train and test data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
    start_preprocessing(x_train, y_train, x_test, y_test)


# read_data()

# Loading house data (train/test data)
train_data = pd.read_csv('SavedData/Regression_Preprocessed_Train_House_Data.csv')
test_data = pd.read_csv('SavedData/Regression_Preprocessed_Test_House_Data.csv')

X_train = train_data.iloc[:, :-1]     # Features
X_test = test_data.iloc[:, :-1]
Y_train = train_data['SalePrice']     # Labels
Y_test = test_data['SalePrice']

# Apply Ridge Regression on the selected features
random_forest_start_time = time.time()  # alarm start time

model = RandomForestRegressor()     # Default parameters are fine
model.fit(X_train, Y_train)
pickle.dump(model, open('SavedData/regression_forest_model.sav', 'wb'))
prediction = model.predict(X_test)

random_forest_end_time = time.time()    # alarm finish time

runtime = random_forest_end_time - random_forest_start_time

# Performance
print("~~~~~ Random Forest Regressor ~~~~~")
print('Mean Square Error: ', metrics.mean_squared_error(np.asarray(Y_test), prediction))
print("Model Accuracy(%): \t" + str(r2_score(Y_test, prediction)*100) + "%")
print(f"Train Runtime: \t\t{round(runtime, 2)} seconds.")
print("~~~~  ~~~~~  ~~~~~  ~~~~~  ~~~~~  ~~~~~  ~~~~~  ~~~~~ ")
