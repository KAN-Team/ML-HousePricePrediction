import numpy as np
import pandas as pd
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from ClassificationDataProcessing import start_preprocessing

# load the data
saved_model_filename = 'classification_saved_poly_model.sav'
start_preprocessing(dataset_name='House_Data_Classification.csv')


data = pd.read_csv('Classification_Preprocessed_House_Data.csv')
X = data.iloc[:, 0:2]    # Features
Y = data.iloc[:, -1]     # Label


poly_start_time = time.time()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
C = 0.1  # SVM regularization parameter
poly_svm = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, y_train)
pickle.dump(poly_svm, open(saved_model_filename, 'wb'))

poly_end_time = time.time()

# make prediction using the model
predictions = poly_svm.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred=predictions)*100

print("-----------------------------------------------------------\n")
print("accuracy for polynomial classifier: {}%".format(accuracy))
print("Training Time for polynomial classifier: {}s".format(poly_end_time - poly_start_time))
print('MSE for polynomial classifier: ', metrics.mean_squared_error(np.asarray(y_test), predictions))
