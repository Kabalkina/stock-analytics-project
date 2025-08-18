# scripts/2_model.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib


def train_model(data_path: str, model_path: str = "models/rf_model.pkl") -> dict:
    """
    Train a RandomForest model on prepared dataset.

    Parameters
    ----------
    data_path : str
        Path to processed dataset (CSV).
    model_path : str
        Path to save trained model.

    Returns
    -------
    dict
        Dictionary containing model and evaluation metrics.
    """
    df = pd.read_csv(data_path)

    # Features and target
    X = df.drop(columns=["buy_signal"])  # adjust if needed
    y = df["buy_signal"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Define model and grid search
    param_grid = {
        "n_estimators": [100, 300],
        "max_depth": [5, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"]
    }

    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=3, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # Save model
    joblib.dump(best_model, model_path)

    # Evaluate
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {"model": best_model, "report": report}


if __name__ == "__main__":
    results = train_model("data/processed_data.csv")
    print("Model trained. Report:", results["report"])
