
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import joblib

from scripts.data import DataRepo

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("model_pipeline")

class Model:
    # quarterly data
    quarterly_df: pd.DataFrame

    # full dataset with dummies
    df: pd.DataFrame

    # dataframes for ML
    train_df: pd.DataFrame
    test_df: pd.DataFrame

    X_train: pd.DataFrame

    #attributes
    CATEGORICAL: list
    TO_PREDICT: list
    USED_BUY_SIGNAL_LABEL: list
    TO_DROP: list

    #performance
    MODEL: RandomForestClassifier
    FEATURE_IMPORTANCES: pd.Series
    CLASSIFICATION_REPORT: dict
    CONFUSION_MATRIX: pd.DataFrame


    def __init__(self, quarterly_data):
        # Accept DataRepo or DataFrame
        if hasattr(quarterly_data, "quarterly_data"):
            self.quarterly_df = quarterly_data.quarterly_data.copy(deep=True)
        else:
            self.quarterly_df = quarterly_data.copy(deep=True)

        # Initialize df as a working copy
        self.df = self.quarterly_df.copy(deep=True)
        self.df['date'] = pd.to_datetime(self.df['date'], errors='coerce')


    def define_categories(self):
        self.CATEGORICAL = ['ticker', 'sector']
        self.TO_PREDICT = ['buy_signal']
        self.USED_BUY_SIGNAL_LABEL = ['qma_2', 'qma_4', 'return_2q', 'sp500_qoq','momentum_3q', 'trailingPE', 'interest_us_qoq']
        self.TO_DROP = ['companyName', 'date', 'year', 'quarter', 'close', 'region'] + self.USED_BUY_SIGNAL_LABEL + self.CATEGORICAL
        # TODO remove region afterwards

    def get_dummies(self):
        dummy_variables = pd.get_dummies(self.df[self.CATEGORICAL], dtype='int32')
        self.df = pd.concat([self.df, dummy_variables], axis=1)

    def handle_nans(self):
        # Handle NaNs in numeric features
        imputer = SimpleImputer(strategy="median")

        numeric_features = self.df.drop(self.CATEGORICAL + self.TO_PREDICT + self.TO_DROP, axis=1).columns
        self.df[numeric_features] = imputer.fit_transform(self.df[numeric_features])

    def split_data(self):
        # 1. Sort chronologically
        self.df = self.df.sort_values(['date', 'ticker']).reset_index(drop=True)

        # 2. Automatic 80/20 date cutoff

        start_date = self.df['date'].min()
        end_date = self.df['date'].max()
        cutoff_date = start_date + (end_date - start_date) * 0.8

        self.train_df = self.df[self.df['date'] < cutoff_date]
        self.test_df = self.df[self.df['date'] >= cutoff_date]


    def train_model(self):
        """
        Train a RandomForest model on prepared dataset.

        """

        #  Features and target
        X_train = self.train_df.drop(self.CATEGORICAL + self.TO_PREDICT + self.TO_DROP, axis=1)
        y_train = self.train_df[self.TO_PREDICT[0]]

        X_test = self.test_df.drop(self.CATEGORICAL + self.TO_PREDICT + self.TO_DROP, axis=1)
        y_test = self.test_df[self.TO_PREDICT[0]]

        #for visual in dashboard
        self.X_train = X_train

        # Handle NaNs only for numeric features
        #imputer = SimpleImputer(strategy="median")
        #X_train = imputer.fit_transform(X_train)
        #X_test = imputer.transform(X_test)

        # Define model and grid search
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }

        rf = RandomForestClassifier(random_state=42, n_jobs=-1)

        # TimeSeriesSplit for CV
        tscv = TimeSeriesSplit(n_splits=5)

        logger.info("Starting RandomForest training with GridSearchCV")

        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            scoring='accuracy',
            cv=tscv,
            n_jobs=-1,
            verbose=2,
            return_train_score=True
        )

        grid_search.fit(X_train, y_train)
        logger.info("Best parameters found: %s", grid_search.best_params_)
        logger.info("Best CV Score: %s", grid_search.best_score_)

        best_model = grid_search.best_estimator_
        best_model.fit(X_train, y_train)
        self.MODEL = best_model

        # Feature importances
        importances = grid_search.best_estimator_.feature_importances_
        features = X_train.columns
        self.FEATURE_IMPORTANCES = pd.Series(importances, index=features).sort_values(ascending=False).head(10)

        # Save the model to a file
        model_filename = 'random_forest_model.joblib'
        script_dir = os.path.dirname(os.path.abspath(__file__))  # .../scripts
        data_dir = os.path.join(script_dir, "../Data")
        path = os.path.join(data_dir, model_filename)
        joblib.dump(best_model, path)

        logger.info("Model saved to %s", path)

        # Evaluate
        y_pred = best_model.predict(X_test)

        self.CLASSIFICATION_REPORT= classification_report(y_test, y_pred, output_dict=True)
        self.CONFUSION_MATRIX = confusion_matrix(y_test, y_pred)

        acc = accuracy_score(y_test, y_pred)
        logger.info(f"Test Accuracy: {acc:.4f}")
        logger.debug(f"Classification Report: {self.CLASSIFICATION_REPORT}")



    def make_predictions(self):

        logger.info("Making predictions on full dataset")
        X_all = self.df.drop(self.CATEGORICAL + self.TO_PREDICT + self.TO_DROP, axis=1)
        #X_all = self.imputer.transform(X_all)
        y_pred_all = self.MODEL.predict(X_all)
        self.df['pred_rf'] = y_pred_all



    def save_data_for_simulation(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))  # .../scripts
        data_dir = os.path.join(script_dir, "../Data")
        path = os.path.join(data_dir, "rf_model_pred.csv")

        rf_model_pred = self.df[['date', 'year', 'quarter', 'ticker', 'close','buy_signal', 'pred_rf']]
        rf_model_pred.to_csv(path, index=False)
        logger.info(f"Saved predictions to {path}")
        return rf_model_pred
