from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.data.load import load_data

def split_data(df):
    X = df.drop(columns=["PlacementStatus"])
    y = df["PlacementStatus"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        Y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    return X_train, X_test, y_train, y_test

def identify_features(X):
    numerical_features = X.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()
    categorical_features = X.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()
    return numerical_features, categorical_features

def standardize_data(X_train, X_test, numerical_features):
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    #Fit only on training data
    X_train[numerical_features] = scaler.fit_transform(
        X_train[numerical_features]
    )
    #Use the same scaler for test data
    X_test[numerical_features] = scaler.transform(
        X_test[numerical_features]
    )
    return X_train, X_test, scaler

if __name__ == "__main__":

    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df)

    numerical_features, categorical_features = identify_features(X_train)

    print("Numerical Features:")
    print(numerical_features)

    print("\nCategorical Features:")
    print(categorical_features)

    X_train, X_test, scaler = standardize_data(
        X_train,
        X_test,
        numerical_features
    )

    print("\nTraining:")
