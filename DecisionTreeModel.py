import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import precision_score

# load the data
data = pd.read_csv('House_Data_Classification.csv')
print(data.describe())
print(data.columns)

X = data.copy()
y = data['PriceRate']

# divide data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# the decision tree model
dec_tree_model = DecisionTreeClassifier() #here I can put the max depth for the tree
print(dec_tree_model)

# train the model
dec_tree_model = dec_tree_model.fit(X_train, y_train)

# make prediction using the model
predictions = dec_tree_model.predict(X_test)
print(predictions)

# show the difference between tree that doesn't have stopping criteria and the one that does
dec_tree_model.predict_proba(X_test)

# evaluate the model "accuracy score"
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predictions))

# evaluate the model "precision score"
print("DecisionTrees's Precision: ", precision_score(y_test, predictions))