import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import math

# load the data
data = pd.read_csv('Classification_Preprocessed_House_Data.csv')
print(data.head())

X = data.iloc[:, 0:5]     # Features
Y = data['PriceRate']     # Label

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
print(X_train.shape)

# generate the model
log_reg_model = LogisticRegression()

# train the model
log_reg_model.fit(X_train, y_train)

# make prediction using the model
predictions = log_reg_model.predict(X_test)

#log_reg_model.score(X_train, y_test)

log_reg_model.coef_     #w transpose

log_reg_model.intercept_    # B

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def prediction_function(x):
    z = log_reg_model.coef_ * x + log_reg_model.intercept_
    y = sigmoid(z)
    return y