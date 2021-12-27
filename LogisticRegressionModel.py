import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import math

df = pd.read_csv('Classification_Preprocessed_House_Data.csv')

print(df.head())

x_train, x_test, y_train, y_test = train_test_split(df[['PriceRate']], df.PriceRate, test_size=0.2)

model = LogisticRegression()
model.fit(x_train, y_train)

y_preicted = model.predict(x_test)
model.score(x_test, y_test)

model.coef_     #w transpose

model.intercept_    # B

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def prediction_function(x):
    z = model.coef_ * x + model.intercept_
    y = sigmoid(z)
    return y