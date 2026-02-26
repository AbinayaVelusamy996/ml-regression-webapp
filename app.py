from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)


data = pd.read_csv("house_price_regression_dataset.csv")

X = data.drop("House_Price", axis=1)
y = data["House_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.01),
    "Polynomial Regression": Pipeline([
        ("poly", PolynomialFeatures(degree=2)),
        ("lr", LinearRegression())
    ]),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
}

metrics = {}
trained_models = {}


for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)


    preds = np.maximum(preds, 0)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    metrics[name] = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

    trained_models[name] = model


best_model = max(metrics, key=lambda x: metrics[x]["R2"])


plt.figure()
plt.bar(metrics.keys(), [metrics[m]["R2"] for m in metrics])
plt.xticks(rotation=45)
plt.title("Model Comparison (R2 Score)")
plt.tight_layout()
plt.savefig("static/model_comparison.png")
plt.close()


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None

    if request.method == "POST":
        values = [
            float(request.form[col]) for col in X.columns
        ]

        model_name = request.form["model"]
        model = trained_models[model_name]

        prediction = model.predict([values])[0]
        prediction = max(prediction, 0)

    return render_template(
        "predict.html",
        columns=X.columns,
        models=models.keys(),
        prediction=prediction
    )

@app.route("/compare")
def compare():
    return render_template(
        "compare.html",
        metrics=metrics,
        best_model=best_model
    )

if __name__ == "__main__":
    app.run(debug=True)
