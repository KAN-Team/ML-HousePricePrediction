import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder


def log_dataframe_info(dataframe, rows=20):
    null_values = dataframe.isnull().sum().sort_values(ascending=False)
    null_percent = (dataframe.isnull().sum() / dataframe.isnull().count() * 100).sort_values(ascending=False)
    print("[Log: Info] NULL values percentages...")
    null_values_percent = pd.concat([null_values, null_percent], axis=1, keys=['Total', 'Percent'])
    print(null_values_percent.head(rows))
    print("Data Shape: %s\n" % (dataframe.shape,))


def replace_with_most_frequent(dataframe, col=None):
    print("%s column Processing..." % col)
    print("[Log: Info] %s value_counts()" % col)
    top_frequent_5 = dataframe[col].value_counts()
    print(top_frequent_5.head(5))
    # Entered col has missing values which we're replacing with the most entered value.
    most_freq_value = top_frequent_5[:1].idxmax()
    dataframe[col].fillna(value=most_freq_value, inplace=True)
    # Checking the column again if there are any other missing values.
    print("Number of NULL values in %s after processing: %d\n" % (col, dataframe[col].isnull().sum()))


def generate_cleaned_file(features, sale_price):
    print("================================================")
    print("...generate_cleaned_file starts...\n")

    dataframe = features
    dataframe['SalePrice'] = sale_price.values
    dataframe.to_csv("Preprocessed_House_Data.csv", index=0)

    print('Preprocessed House_Data file has been generated...\n')
    print("...generate_cleaned_file ends...")
    print("================================================\n")


def features_scaling(features):
    print("================================================")
    print("...features_scaling starts...\n")

    features = np.array(features)
    normalized_features = np.zeros((features.shape[0], features.shape[1]))

    for i in range(features.shape[1]):
        normalized_features[:, i] = ((features[:, i] - min(features[:, i])) /
                                     (max(features[:, i]) - min(features[:, i])))

    features = pd.DataFrame(normalized_features)

    print("...features_scaling ends...")
    print("================================================\n")
    return features


def features_selection(dataframe):
    print("================================================")
    print("...features_selection starts...\n")

    # Getting the features Correlation
    correlation = dataframe.corr()
    # Top 50% Correlation features with the SalePrice
    top_features = correlation.index[abs(correlation['SalePrice'] > 0.5)]

    # Showing the Correlation plot
    plt.subplots(figsize=(8, 8))
    top_correlation = dataframe[top_features].corr()
    sns.heatmap(top_correlation, annot=True)
    # plt.show()

    # selected_features = ['OverallQual', YearRemodAdd, 'GrLivArea', 'TotalBsmtSF', 'GarageArea']
    top_features = top_features.delete([2, 4, 6, 7, 8, 10])

    print(top_features)
    selected_features = dataframe[top_features]

    print("...features_selection ends...")
    print("================================================")
    return selected_features


def label_encoding(dataframe):
    print("================================================")
    print("...label_encoding starts...\n")
    # print(dataframe.head(5))

    columns = ['Utilities', 'Street', 'LotShape', 'MSZoning', 'LotConfig', 'LandSlope', 'Neighborhood', 'BldgType',
            'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual',
            'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'GarageType',
            'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']

    for col in columns:
        lbl = LabelEncoder()
        lbl.fit(list(dataframe[col].values))
        dataframe[col] = lbl.transform(list(dataframe[col].values))

    print("...label_encoding ends...")
    print("================================================\n")
    return dataframe


def solve_missing_values(dataframe):
    print("================================================")
    print("...solve_missing_values starts...\n")

    # we start the preprocessing of data by looking out for missing values in each explanatory variables.
    log_dataframe_info(dataframe)

    # LotFrontage column Processing...
    print("LotFrontage column Processing...")
    # LotFrontage has 200 missing values which we're replacing with the previous used value in the same column.
    dataframe['LotFrontage'].fillna(method='ffill', limit=3, inplace=True)
    # Checking the 'LotFrontage' column again if there are any other missing values
    print("Number of NULL values in LotFrontage after processing: %d\n" % dataframe['LotFrontage'].isnull().sum())

    # GarageCond column Processing...
    replace_with_most_frequent(dataframe, col='GarageCond')

    # GarageType column Processing...
    replace_with_most_frequent(dataframe, col='GarageType')

    # GarageFinish column Processing...
    replace_with_most_frequent(dataframe, col='GarageFinish')

    # GarageQual column Processing...
    replace_with_most_frequent(dataframe, col='GarageQual')

    # BsmtFinType2 column Processing...
    replace_with_most_frequent(dataframe, col='BsmtFinType2')

    # BsmtExposure column Processing...
    replace_with_most_frequent(dataframe, col='BsmtExposure')

    # BsmtQual column Processing...
    replace_with_most_frequent(dataframe, col='BsmtQual')

    # BsmtCond column Processing...
    replace_with_most_frequent(dataframe, col='BsmtCond')

    # BsmtFinType1 column Processing...
    replace_with_most_frequent(dataframe, col='BsmtFinType1')

    # MasVnrArea column Processing...
    replace_with_most_frequent(dataframe, col='MasVnrArea')

    # MasVnrType column Processing...
    replace_with_most_frequent(dataframe, col='MasVnrType')

    # TotRmsAbvGrd column Processing...
    replace_with_most_frequent(dataframe, col='TotRmsAbvGrd')

    print("...solve_missing_values ends...")
    print("================================================\n")
    return dataframe


def drop_unwanted_data(dataframe):
    print("================================================")
    print("...drop_unwanted_data starts...\n")

    # dropping id column
    print("[Log: Info] Removing unwanted columns...")
    dataframe = dataframe.iloc[:, 1:78]

    # dropping rows with high missing values which have more than 10% of total data missing or showing no values.
    # print("Dropping rows with null percentages more than 10%...")
    # dataframe.dropna(axis=0, how='any', thresh=70, inplace=True)
    # log_dataframe_info(dataframe, rows=10)

    # dropping columns with high missing values which have more than 40% of total data missing or showing no values.
    print("Dropping columns with null percentages more than 40%...")
    dataframe = dataframe.drop(['PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu'], axis=1)
    log_dataframe_info(dataframe, rows=14)

    print("Data Shape after rows/columns removal: %s\n" % (dataframe.shape,))

    print("...drop_unwanted_data ends...")
    print("================================================\n")
    return dataframe


def read_data():
    print("================================================")
    print("read_data starts...\n")

    # load data
    print("[Log: Info] Reading House_Data.csv...")
    dataframe = pd.read_csv('House_Data.csv')

    # display data info
    print("Data Original Shape: %s" % (dataframe.shape,))
    pd.set_option("Display.max_columns", None)
    n = len(dataframe['Id'])
    print("Examples Count: %d\n" % n)

    print("read_data ends...")
    print("================================================\n")
    return dataframe


def start_preprocessing():
    df = read_data()
    df = drop_unwanted_data(df)
    df = solve_missing_values(df)
    df = label_encoding(df)
    fts = features_selection(df)
    fts = features_scaling(fts)
    generate_cleaned_file(fts, df['SalePrice'])
    return fts


if __name__ == "__main__":
    start_preprocessing()
