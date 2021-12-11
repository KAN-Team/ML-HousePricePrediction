import time
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from DataProcessing import start_preprocessing

# Load house data

start_preprocessing()
data = pd.read_csv('Preprocessed_House_Data.csv')

X = data.iloc[:, 0:5]     # Features
Y = data['SalePrice']     # Label

# Apply Ridge Regression on the selected features
ridge_start_time = time.time()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
cls = RidgeCV(alphas=range(1, 40))
cls.fit(X_train, Y_train)
prediction = cls.predict(X_test)

time.sleep(1)
ridge_end_time = time.time()

# Performance
print('Mean Square Error: ', metrics.mean_squared_error(np.asarray(Y_test), prediction))
print("\tAccuracy: \t\t" + str(r2_score(Y_test, prediction)*100))
print(f"Runtime of Ridge regression is {ridge_end_time - ridge_start_time}")
