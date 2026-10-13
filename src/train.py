import mlflow
import mlflow.sklearn

import pandas as pd


from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor


def load_data():
    df = pd.read_csv(
        "industrial-machine-condition-classification/student_dataset.csv"
    )

    X = df.drop(columns=["unit_id", "condition"])
    y = df["condition"]

    return X, y


def train():
    X, y = load_data()

    preprocessor = build_preprocessor(X)

    model = XGBClassifier(
        random_state=42,
        eval_metric="mlogloss",
        n_jobs=4
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    mlflow.set_experiment("machine_condition_classification")

    with mlflow.start_run(run_name="xgboost_baseline"):

        results = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring="accuracy",
            return_train_score=True,
            n_jobs=1
        )

        train_acc = results["train_score"].mean()
        val_acc = results["test_score"].mean()
        val_std = results["test_score"].std()

        mlflow.log_param("model","XGBoost")
        mlflow.log_param("n_splits", 5)
        mlflow.log_param("random_state", 42) 

        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("cv_accuracy", val_acc)
        mlflow.log_metric("cv_std", val_std)

        print("Train Accuracy:", train_acc)
        print("Validation Accuracy:", val_acc)
        print("Validation Std:", val_std)

        pipeline.fit(X, y)

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="machine_condition_model",
            input_example=X.head(5),
            skops_trusted_types=[
                "numpy.dtype",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier"
            ]
        )

        print("Model URI:", model_info.model_uri)
    

if __name__ == "__main__":
    train()
