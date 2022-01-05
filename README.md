# House Price Prediction
Growing unaffordability of housing has become one of the major challenges for metropolitan cities around the world. In order to gain a better understanding of the commercialized housing market we are currently facing; we want to figure out what are the top influential factors of the housing price. Apart from the more obvious driving forces such as the inflation and the scarcity of land, there are also several variables that are worth looking into. The task here is to reach a model that can closely predict the pricing of a house.

# Preprocessing
Data preprocessing is a process of preparing the raw data and making it suitable for a machine learning model. It is the first and crucial step while creating a machine learning model.<br>
Before we deep into our preprocessing techniques, there are a few things to clarify:
  - We are passing two separate objects to the preprocessing methods one is representing the training data and the other represents the unseen data for testing. All the preprocessing techniques are mainly applied to the training data while the testing data are just keeping up with the training data
  - We are saving all the models used in preprocessing using the ‘pickle’ library, for later use.

### 1.1 Dropping useless Features
Starting the Pre-processing by neglecting the unnecessary columns either which don't affect the target value (such as case index) or contain too many missing values.
- Dropping ID column as it only refers to row index
- Eliminating those which have more than 80% of total data missing or showing no values. For this dataset, we are Dropping PoolQC, MiscFeature, and Fence. Their Missing values percentages were very high, 99%, 96%, and 81% respectively. Which they cover almost the entire column, eliminating these features won't cause any significant loss of values or even affects the prediction process. The right evaluation was also considered as it is important to be aware of dealing with such missing values because there may be a chance of losing important data.

### 1.2 Solving missing values
Imputing missing data is very important in the Feature Engineering process. There are many techniques to impute the missing values in such an accurate manner. What attracted us were the Iterative Imputer (MICE) and the mean replacement techniques. <br>
The Iterative Imputer is such an effective algorithm which refers to a process where each feature is modeled as a function of the other features, e.g. a regression problem where missing values are predicted. Each feature is imputed sequentially, one after the other, allowing prior imputed values to be used as part of a model in predicting subsequent features and it uses the mean as the initial_strategy. In our trials it was found that the Iterative Imputation is not stable with changing the random seed. But it was an experience that has to be tested.
- Numeric Data is imputed using the mean strategy
- Categorical Data is imputed using the most-frequented value.

### 1.3 Label encoding
House Pricing Dataset contains 35 categorical features which we can’t ignore (drop) or even let them as they are. We started the encoding using LabelEncoder and OneHotEncoder. <br>
OneHotEncoder converts each categorical value into a new categorical column and assigns a binary value of 1 or 0 to those columns. But after applying that and then observing the correlation (1.4 Feature Selection) was found that all the correlated features are non-categorical data, then for preserving memory there was no need to use OneHotEncoder. So with the help of sklearn, we used LabelEncoder instead of OneHotEncoder to encode the textual data in a sensible manner.

### 1.4 Feature Selection
By using the Pearson Coefficient of Correlation we are getting the top 50% correlation features with the target (SalePrice). <br>
There are features (from the top features) for Example 1stFlrSF and TotalBsmtSF, both are strongly correlated with each other and also have a strong correlation with the target. So for preserving the memory and runtime, there is no need to consider both features, we consider selecting from each strongly correlated features the most correlated one with the target (SalePrice) and dropping the rest.

### 1.5 Feature Scaling
Feature scaling is a method used to normalize the range of independent variables or features of data. It makes the flow of gradient descent smooth and helps algorithms quickly reach the minima of the cost function. <br>
We used Standardization which is a very effective technique that re-scales a feature value so that it has distribution with 0 mean value and variance equals 1. <br>
Although Decision trees and ensemble methods do not require feature scaling to be performed as they are not sensitive to the variance in the data, we kept it as a general process to lower the headache that is going to happen in saving the data as a side effect.
