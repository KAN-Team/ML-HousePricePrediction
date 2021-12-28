import numpy as np
import pandas as pd
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from ClassificationDataProcessing import start_preprocessing


# load the data
saved_model_filename = 'classification_saved_decisionTree_model.sav'
start_preprocessing(dataset_name='House_Data_Classification.csv')

data = pd.read_csv('Classification_Preprocessed_House_Data.csv')

X = data.iloc[:, :2]     # Features
Y = data.iloc[:, -1]     # Label
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=3, shuffle=True)


DT_start_time = time.time()

# the decision tree model
dec_tree_model = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=10).fit(X_train, y_train)
pickle.dump(dec_tree_model, open(saved_model_filename, 'wb'))

DT_end_time = time.time()


# make prediction using the model
predictions = dec_tree_model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred=predictions)*100

print("-----------------------------------------------------------\n")
print("accuracy for Decision Tree classifier: {}%".format(accuracy))
print("Training Time for Decision Tree classifier: {}s".format(DT_end_time - DT_start_time))
print('MSE for Decision Tree classifier: ', metrics.mean_squared_error(np.asarray(y_test), predictions))
