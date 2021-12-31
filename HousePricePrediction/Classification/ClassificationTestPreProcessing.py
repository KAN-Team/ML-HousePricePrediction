import pandas as pd
import pickle
pd.options.mode.chained_assignment = None


def features_scaling(X_test):
    print("================================================")
    print("...features_scaling starts...")

    # Loading selected_features.sav for selecting top features
    scaler = pickle.load(open('Classification/SavedData/features_scaling.sav', 'rb'))
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    print("...features_scaling ends...")
    print("================================================\n")
    return X_test


def features_selection(X_test):
    print("================================================")
    print("...features_selection starts...")

    # Loading selected_features.sav for selecting top features
    selected_features = pickle.load(open('Classification/SavedData/selected_features.sav', 'rb'))
    selected_features = X_test[selected_features]

    print("...features_selection ends...")
    print("================================================\n")
    return selected_features


def label_encoding(X_test, Y_test):
    print("================================================")
    print("...label_encoding starts...\n")

    columns_lbl_encoder = pickle.load(open('Classification/SavedData/label_encoding.sav', 'rb'))

    for col in columns_lbl_encoder:
        lbl = columns_lbl_encoder[col]  # LabelEncoder() object
        if col == 'PriceRate':
            Y_test[col] = lbl.transform(list(Y_test[col].values))
        else:
            X_test[col] = lbl.transform(list(X_test[col].values))

    print("...label_encoding ends...")
    print("================================================\n")
    return X_test, Y_test


def solve_missing_values(X_test):
    print("================================================")
    print("...solve_missing_values starts...")

    # Loading missing_values.sav for imputing NA or missing values
    columns_mean_values = pickle.load(open('Classification/SavedData/missing_values.sav', 'rb'))

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
    print("...Classification Data Pre-processing starts...\n")
    X_test = dataset.iloc[:, :-1]
    Y_test = dataset.iloc[:, -1:]
    X_test = drop_useless_features(X_test)
    X_test = solve_missing_values(X_test)
    X_test, Y_test = label_encoding(X_test, Y_test)
    X_test = features_selection(X_test)
    X_test = features_scaling(X_test)
    print("...Classification Data Pre-processing ends...")
    print("================================================================================\n")
    return X_test, Y_test
