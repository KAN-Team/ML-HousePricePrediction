import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

data = pd.read_csv('House_Data_Classification.csv')
dataset = pd.DataFrame(data=data['data'], columns=data['feature_names'])

print(data.describe())
#print(data.columns)

X = data.copy()
y = data

X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2)


# the decision tree model
clf = DecisionTreeClassifier() #here I can put the max depth for the tree
clf = clf.fit(X_train, y_train)

clf.get_params()

predictions = clf.predict(X_test)
print(predictions)

# show the difference between tree that doesn't have stopping criteria and the one that does
clf.predict_proba(X_test)

accuracy_score(y_test, predictions)

precision_score(y_test, predictions)