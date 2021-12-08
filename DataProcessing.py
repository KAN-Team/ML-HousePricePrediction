import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# load data
df_train = pd.read_csv('House_Data.csv')
print(df_train.shape)

pd.set_option("Display.max_columns", None)
df_train.head()
n = len(df_train['Id'])
print(n)

# we start the preprocessing of data by looking out for missing values in each explainatory variables.
na_values = df_train.isnull().sum().sort_values(ascending=False)
na_percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
na_values_percent = pd.concat([na_values, na_percent], axis=1, keys=['Total', 'Percent'])
print(na_values_percent.head(20))

# drop columns with high missing values which have more than 45% of total data missing or showing no values.
df = df_train.drop(['PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu'], axis=1)
print(df.shape)

# LotFrontage has 211 missing values which is significant number, so I'll replace it with the median
df['LotFrontage'].fillna(df['LotFrontage'].median(), inplace=True)

# check the column 'LotFrontage' again if there are any other missing values
print(df['LotFrontage'].isnull().sum())
print(df.columns)


print(df['GarageCond'].value_counts())
# there are 81 missing values in GarageCond column.
print(df['GarageCond'].isnull().sum())
# replace the missing values with 'No Garage'
df['GarageCond'].fillna('No Garage', inplace=True)
# check the column again if there are any other missing values.
print(df['GarageCond'].isnull().sum())

# replace the null values of the garage related variables
df['GarageType'].fillna('No Garage', inplace=True)
df['GarageYrBlt'].fillna('Unknown', inplace=True)
df['GarageFinish'].fillna('No Garage', inplace=True)
df['GarageQual'].fillna('No Garage', inplace=True)

# check the column again if there are any other missing values.
df['GarageType'].isnull().sum()
df['GarageYrBlt'].isnull().sum()
df['GarageFinish'].isnull().sum()
df['GarageQual'].isnull().sum()

print(df['GarageType'].value_counts())
print(df['GarageYrBlt'].value_counts())
print(df['GarageFinish'].value_counts())
print(df['GarageQual'].value_counts())


print(df['BsmtExposure'].value_counts())

# replace the missing values with 'No Basement'
df['BsmtExposure'].fillna('No Basement', inplace=True)
df['BsmtFinType2'].fillna('No Basement', inplace=True)
df['BsmtFinType1'].fillna('No Basement', inplace=True)
df['BsmtCond'].fillna('No Basement', inplace=True)
df['BsmtQual'].fillna('No Basement', inplace=True)

print(df['BsmtExposure'].value_counts())

# check the columns again if there are any other missing values.
df['BsmtExposure'].isnull().sum()
df['BsmtFinType2'].isnull().sum()
df['BsmtFinType1'].isnull().sum()
df['BsmtCond'].isnull().sum()
df['BsmtQual'].isnull().sum()


print(df['MasVnrType'].value_counts())
df["MasVnrType"].fillna('Unknown', inplace=True)
print(df['MasVnrType'].value_counts())
print(df['MasVnrType'].value_counts())

# check the column again if there are any other missing values.
df['MasVnrType'].isnull().sum()

# replace the null values with the median
df['MasVnrArea'].fillna(df['MasVnrArea'].median(), inplace=True)
print(df['MasVnrType'].isnull().sum())


# replace null values with 'No Fireplace'
#df["FireplaceQu"].value_counts()
#df["FireplaceQu"].fillna('No Fireplace', inplace=True)
#df["FireplaceQu"].value_counts()
#df["FireplaceQu"].isnull().sum()
#df["FireplaceQu"].value_counts()

df

print(df.columns)

df.to_csv("houseprice_datapreprocessing.csv", index=0)

# check distribution of target variable
sns.displot(df.SalePrice)
plt.show()

# transform the target variable
sns.displot(np.log(df.SalePrice))
plt.show()