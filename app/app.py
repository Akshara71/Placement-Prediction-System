from flask import Flask, render_template, request
from src.data.load_data import load_data, get_summary
from src.data.preprocess import (
    split_data,
    identify_features,
    standardize_data,
    one_hot_encode_data
)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/dataset')
def dataset():
    df = load_data()
    summary = get_summary(df)

    view = request.args.get('view', 'head')
    try:
        n = int(request.args.get('n', 5))
    except ValueError:
        n = 5
    n = max(1, min(n, summary['rows']))

    if view == 'tail':
        rows_df = df.tail(n)
    elif view == 'random':
        rows_df = df.sample(n)
    else:
        view = 'head'
        rows_df = df.head(n)

    return render_template(
        "load_dataset.html",
        summary=summary,
        rows_html=rows_df.to_html(index=False, classes="data-table"),
        view=view,
        n=n
    )


@app.route('/eda')
def eda():
    return render_template("eda.html")


@app.route('/preprocess')
def preprocess():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    numerical_features, categorical_features = identify_features(X_train)
    X_train, X_test, scaler = standardize_data(X_train, X_test, numerical_features)
    X_train, X_test, encoder = one_hot_encode_data(X_train, X_test, categorical_features)

    return render_template(
        "preprocess.html",
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        train_shape=X_train.shape,
        test_shape=X_test.shape,
        train_preview=X_train.head().to_html(index=False, classes="data-table"),
        test_preview=X_test.head().to_html(index=False, classes="data-table")
    )


@app.route('/predict')
def predict():
    return "Prediction form coming soon."


if __name__ == "__main__":
    app.run(debug=True)