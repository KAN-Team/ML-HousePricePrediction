import numpy as np
import pandas as pd
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from Classification.ClassificationDataPreProcessing import start_preprocessing


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


DT_start_time = time.time()

# the decision tree model
dec_tree_model = DecisionTreeClassifier(max_depth=3, max_leaf_nodes=15).fit(X_train, Y_train)
pickle.dump(dec_tree_model, open('SavedData/classification_decision_tree_model.sav', 'wb'))

DT_end_time = time.time()


# make prediction using the model
predictions = dec_tree_model.predict(X_test)
accuracy = metrics.accuracy_score(Y_test, y_pred=predictions)*100

print("-----------------------------------------------------------")
print('Decision Tree Classifier MSE:\t\t\t', metrics.mean_squared_error(np.asarray(Y_test), predictions))
print("Decision Tree Classifier Accuracy:\t\t {}%".format(accuracy))
print("Decision Tree Classifier Training Time:\t {} seconds.".format(DT_end_time - DT_start_time))
print("-----------------------------------------------------------")
