import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

# load the data
data = pd.read_csv('House_Data_Classification.csv')
print(data)
print(data['PriceRate'])

X = data.iloc[:, 0:5]     # Features
Y = data['PriceRate']     # Label

# divide data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)

# generate the model
svm_model = SVC(kernel="linear")

# train the model
svm_model.fit(X_train, y_train)

# make prediction using the model
predictions = svm_model.predict(X_test)
print("accuracy:", metrics.accuracy_score(y_test, y_pred=predictions))

# precision score
print("precision:", metrics.precision_score(y_test, y_pred=predictions))

# recall score
print("recall:", metrics.recall_score(y_test, y_pred=predictions))
print(metrics.classification_report(y_test, y_pred=predictions))