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


def balance_test_data(train_col, col, test_col):
    most_frequent = get_most_frequent(train_col, col)
    for idx, val in enumerate(test_col):
        if val not in train_col[col].values:
            test_col.values[idx] = most_frequent
    return test_col
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


def features_scaling(X_test):
    print("================================================")
    print("...features_scaling starts...")

    # Loading selected_features.sav for selecting top features
    scaler = pickle.load(open('Regression/SavedData/features_scaling.sav', 'rb'))
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    print("...features_scaling ends...")
    print("================================================\n")
    return X_test


def features_selection(X_test):
    print("================================================")
    print("...features_selection starts...")

    # Loading selected_features.sav for selecting top features
    selected_features = pickle.load(open('Regression/SavedData/selected_features.sav', 'rb'))
    selected_features = X_test[selected_features]

    print("...features_selection ends...")
    print("================================================\n")
    return selected_features


def label_encoding(X_test):
    print("================================================")
    print("...label_encoding starts...\n")

    columns_lbl_encoder = pickle.load(open('Regression/SavedData/label_encoding.sav', 'rb'))

    for col in columns_lbl_encoder:
        lbl = columns_lbl_encoder[col]  # LabelEncoder() object
        X_test[col] = lbl.transform(list(X_test[col].values))

    print("...label_encoding ends...")
    print("================================================\n")
    return X_test


def solve_missing_values(X_test):
    print("================================================")
    print("...solve_missing_values starts...")

    # Loading missing_values.sav for imputing NA or missing values
    columns_mean_values = pickle.load(open('Regression/SavedData/missing_values.sav', 'rb'))

    for col in columns_mean_values:  # foreach item in dictionary
        X_test[col].fillna(value=columns_mean_values[col], inplace=True)

    print("...solve_missing_values ends...")
    print("================================================\n")
    return X_test


def drop_useless_features(X_test):
    print("================================================")
    print("...drop_useless_features starts...")

    # dropping id column
    X_test = X_test.iloc[:, 1:]

    # dropping columns which we dropped during training
    X_test = X_test.drop(['PoolQC', 'MiscFeature', 'Fence'], axis=1)

    print("...drop_useless_features ends...")
    print("================================================\n")
    return X_test


def start_preprocessing(dataset):
    print("================================================================================")
    print("...Regression Data Pre-processing starts...\n")
    X_test = dataset.iloc[:, :-1]
    Y_test = dataset.iloc[:, -1]
    X_test = drop_useless_features(X_test)
    X_test = solve_missing_values(X_test)
    # X_test = label_encoding(X_test)
    X_test = features_selection(X_test)
    X_test = features_scaling(X_test)
    print("...Regression Data Pre-processing ends...")
    print("================================================================================\n")
    return X_test, Y_test
