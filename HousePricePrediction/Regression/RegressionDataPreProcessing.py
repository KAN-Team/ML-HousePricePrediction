import pandas as pd
import seaborn as sns
import pickle
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
pd.options.mode.chained_assignment = None


# ************************************************* #
def log_dataframe_info(dataframe, rows=20):
    null_values = dataframe.isnull().sum().sort_values(ascending=False)
    null_percent = (dataframe.isnull().sum() / dataframe.isnull().count() * 100).sort_values(ascending=False)
    print("[Log: Info] NULL values percentages...")
    null_values_percent = pd.concat([null_values, null_percent], axis=1, keys=['Total', 'Percent'])
    print(null_values_percent.head(rows))
    print("Data Shape: %s\n" % (dataframe.shape,))


def get_most_frequent(dataframe, col=None):
    print("[Log: Info] %s value_counts()" % col)
    top_frequent = dataframe[col].value_counts()
    print(top_frequent.head(5))
    print()
    most_freq_value = top_frequent[:1].idxmax()
    return most_freq_value
# ************************************************* #


def generate_cleaned_files(X_train, Y_train, X_test, Y_test):
    print("================================================")
    print("...generate_cleaned_files starts...\n")

    # Gathering Features with Target
    train_data = pd.concat([X_train, Y_train], axis=1)
    test_data = pd.concat([X_test, Y_test], axis=1)

    train_data.to_csv('SavedData/Regression_Preprocessed_Train_House_Data.csv', index=0)
    print('Preprocessed Train House_Data file has been generated -> \'SavedData\' Folder...')
    test_data.to_csv('SavedData/Regression_Preprocessed_Test_House_Data.csv', index=0)
    print('Preprocessed Test House_Data file has been generated -> \'SavedData\' Folder...\n')

    print("...generate_cleaned_file ends...")
    print("================================================\n")


def features_scaling(X_train, X_test):
    print("================================================")
    print("...features_scaling starts...\n")

    # fit on training data column
    scaler = StandardScaler().fit(X_train)

    # transform the training data columns
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)

    # transform the testing data columns
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    print("X_test after Standardization:")
    print(X_test)

    pickle.dump(scaler, open('SavedData/features_scaling.sav', 'wb'))
    print("-> Scaled Weights Saved into \'SavedData/features_scaling.sav\'\n")

    print("...features_scaling ends...")
    print("================================================\n")
    return X_train, X_test


def features_selection(X_train, Y_train, X_test):
    print("================================================")
    print("...features_selection starts...\n")

    # Gathering Features with Target
    train_data = pd.concat([X_train, Y_train], axis=1)

    # Getting the train data features Correlation
    correlation = train_data.corr()
    # Top 50% Correlation features with the SalePrice
    top_features = correlation.index[abs(correlation['SalePrice'] > 0.5)]

    # Showing the Correlation plot
    plt.subplots(figsize=(8, 8))
    top_correlation = train_data[top_features].corr()
    sns.heatmap(top_correlation, annot=True)
    # plt.show()

    # frequented selected_features = ['OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea', 'GarageArea']
    top_features = top_features.delete(-1)

    print("Top Features to be Selected:")
    print(top_features)

    pickle.dump(top_features, open('SavedData/selected_features.sav', 'wb'))
    print("-> Selected Features Saved into \'SavedData/selected_features.sav\'\n")

    selected_train_features = X_train[top_features]
    selected_test_features = X_test[top_features]
    print("...features_selection ends...")
    print("================================================\n")
    return selected_train_features, selected_test_features


def label_encoding(X_train, X_test):
    print("================================================")
    print("...label_encoding starts...\n")

    columns = ['Utilities', 'Street', 'LotShape', 'MSZoning', 'LotConfig', 'LandSlope', 'Neighborhood', 'BldgType',
            'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual',
            'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
            'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'GarageType',
            'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']

    all_possible_categories = pd.concat([X_train, X_test], axis=0)
    columns_lbl_encoder = {}
    for col in columns:
        lbl = LabelEncoder()
        lbl.fit(list(all_possible_categories[col].values))
        X_train[col] = lbl.transform(list(X_train[col].values))
        X_test[col] = lbl.transform(list(X_test[col].values))
        columns_lbl_encoder[col] = lbl

    pickle.dump(columns_lbl_encoder, open('SavedData/label_encoding.sav', 'wb'))

    print("-> Encoding Data Saved into \'SavedData/label_encoding.sav\'\n")
    print("...label_encoding ends...")
    print("================================================\n")
    return X_train, X_test


def solve_missing_values(X_train, X_test):
    print("================================================")
    print("...solve_missing_values starts...\n")

    X_train_numeric_idx = X_train.select_dtypes(include='number').columns
    X_train_categoric_idx = X_train.select_dtypes('object').columns

    X_train_numeric = X_train[X_train_numeric_idx]
    X_train_categoric = X_train[X_train_categoric_idx]
    X_test_numeric = X_test[X_train_numeric_idx]
    X_test_categoric = X_test[X_train_categoric_idx]

    columns_mean_values = {}
    for col in X_train_numeric.columns:
        col_mean = X_train_numeric[col].mean()
        X_train_numeric[col].fillna(value=col_mean, inplace=True)
        X_test_numeric[col].fillna(value=col_mean, inplace=True)
        columns_mean_values[col] = col_mean

    for col in X_train_categoric.columns:
        most_freq_value = get_most_frequent(X_train_categoric, col)
        X_train_categoric[col].fillna(value=most_freq_value, inplace=True)
        X_test_categoric[col].fillna(value=most_freq_value, inplace=True)
        columns_mean_values[col] = most_freq_value

    # Save missing values replacement values of the train data
    pickle.dump(columns_mean_values, open('SavedData/missing_values.sav', 'wb'))

    X_train = pd.concat([X_train_numeric, X_train_categoric], axis=1)
    X_test = pd.concat([X_test_numeric, X_test_categoric], axis=1)

    log_dataframe_info(X_train, rows=5)
    print("-> Imputing Data Saved into \'SavedData/missing_values.sav\'\n")
    print("...solve_missing_values ends...")
    print("================================================\n")
    return X_train, X_test


def drop_useless_features(X_train, X_test):
    print("================================================")
    print("...drop_useless_features starts...\n")

    # dropping id column
    print("[Log: Info] Removing ID column...")
    X_train = X_train.iloc[:, 1:]
    X_test = X_test.iloc[:, 1:]

    # dropping columns with missing values more than 80% of total Train data or showing no values.
    print("[Log: Info] Dropping features with null percentages more than 80%...")
    # log_dataframe_info(X_train, rows=14)  # uncomment this line to show null values percentages...
    X_train = X_train.drop(['PoolQC', 'MiscFeature', 'Fence'], axis=1)
    X_test = X_test.drop(['PoolQC', 'MiscFeature', 'Fence'], axis=1)
    print("Train Data Shape after columns removal: %s" % (X_train.shape,))
    print("Test Data Shape after columns removal: %s\n" % (X_test.shape,))

    print("...drop_useless_features ends...")
    print("================================================\n")
    return X_train, X_test


def start_preprocessing(X_train, Y_train, X_test, Y_test):
    print("================================================================================")
    print("...Regression Data Pre-processing starts...\n")
    X_train, X_test = drop_useless_features(X_train, X_test)
    X_train, X_test = solve_missing_values(X_train, X_test)
    X_train, X_test = label_encoding(X_train, X_test)
    X_train, X_test = features_selection(X_train, Y_train, X_test)
    X_train, X_test = features_scaling(X_train, X_test)
    generate_cleaned_files(X_train, Y_train, X_test, Y_test)
    print("...Regression Data Pre-processing ends...")
    print("================================================================================\n")
