import time
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from DataProcessing import start_preprocessing

# Load house data

start_preprocessing()
data = pd.read_csv('Preprocessed_House_Data.csv')

X = data.iloc[:, 0:5]     # Features
Y = data['SalePrice']     # Label

# Apply MultiLinear Regression on the selected features
multi_reg_start_time = time.time()
cls = linear_model.LinearRegression()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
cls.fit(X_train, Y_train)
prediction = cls.predict(X_test)
time.sleep(1)
multi_reg_end_time = time.time()

# Performance
print('Mean Square Error: ', metrics.mean_squared_error(np.asarray(Y_test), prediction))
print("\tAccuracy: \t\t" + str(r2_score(Y_test, prediction)*100))
print(f"Runtime of multi regression is {multi_reg_end_time - multi_reg_start_time}")
