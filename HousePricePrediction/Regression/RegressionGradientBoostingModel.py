import time
import numpy as np
import pandas as pd
import pickle
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from RegressionDataPreProcessing import start_preprocessing

# Load house data

saved_model_filename = 'regression_saved_gradient_model.sav'
start_preprocessing(dataset_name='House_Data_Regression.csv')
data = pd.read_csv('Regression_Preprocessed_House_Data.csv')

X = data.iloc[:, 0:5]     # Features
Y = data['SalePrice']     # Label

# Apply GradientBoostingRegressor on the selected features
gradient_boosting_start_time = time.time()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
cls = GradientBoostingRegressor()
cls.fit(X_train, Y_train)
pickle.dump(cls, open(saved_model_filename, 'wb'))
prediction = cls.predict(X_test)

time.sleep(1)
gradient_boosting_end_time = time.time()

runtime = gradient_boosting_end_time - gradient_boosting_start_time

# Performance
print('Mean Square Error: ', metrics.mean_squared_error(np.asarray(Y_test), prediction))
print("Model Accuracy(%): \t" + str(r2_score(Y_test, prediction)*100) + "%")
print(f"Runtime of Gradient Boosting Regressor is: {round(runtime, 2)} seconds.")
