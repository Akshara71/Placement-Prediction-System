from flask import Flask, render_template

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


# Placeholder route for the prediction form — build this out next
# once the ML model and input fields are ready.
@app.route('/predict')
def predict():
    return "Prediction form coming soon."


if __name__ == "__main__":
    app.run(debug=True)