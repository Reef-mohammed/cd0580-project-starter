"""
Library of functions to predict credit card customer churn.

The pipeline loads the bank data, runs exploratory data analysis (EDA),
encodes categorical features, trains a random forest and a logistic
regression model, and saves the EDA plots, evaluation plots and trained
models to disk.

Run the full pipeline from the command line with:
    python churn_library.py

Author: Reef Mohammed
Date: 2026-11-06
"""

import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import constants as const

# "Agg" renders straight to image files, so no window or display is needed.
plt.switch_backend("Agg")

sns.set_theme()
logger = logging.getLogger(__name__)


def create_output_directories():
    """
    Create the output directories used by the project if they do not exist.

    input:
            None
    output:
            None
    """
    for directory in (const.EDA_DIR, const.RESULTS_DIR,
                      const.MODELS_DIR, const.LOGS_DIR):
        os.makedirs(directory, exist_ok=True)


def import_data(pth):
    """
    Return a dataframe for the csv found at pth, with a binary churn column.

    The response column (const.RESPONSE, "Churn") is 1 for customers whose
    Attrition_Flag is "Attrited Customer" and 0 otherwise.

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    df = pd.read_csv(pth, index_col=0)
    df[const.RESPONSE] = (
        df["Attrition_Flag"] != "Existing Customer").astype(int)
    logger.info("Loaded %d rows and %d columns from %s",
                df.shape[0], df.shape[1], pth)
    return df


def _save_figure(output_pth):
    """
    Save the current matplotlib figure to output_pth and close it.

    input:
            output_pth: path of the image file to write
    output:
            None
    """
    plt.tight_layout()
    plt.savefig(output_pth)
    plt.close()


def perform_eda(df):
    """
    Perform EDA on df and save the figures to the images/eda folder.

    Saves a univariate categorical plot (churn, marital status), univariate
    quantitative plots (customer age, total transaction count) and a
    bivariate plot (correlation heatmap).

    input:
            df: pandas dataframe that includes the response column
    output:
            None
    """
    create_output_directories()

    plt.figure(figsize=(20, 10))
    df[const.RESPONSE].hist()
    plt.title("Churn distribution (0 = existing, 1 = churned)")
    _save_figure(const.EDA_IMAGES["churn"])

    plt.figure(figsize=(20, 10))
    df["Customer_Age"].hist()
    plt.title("Customer age distribution")
    _save_figure(const.EDA_IMAGES["customer_age"])

    plt.figure(figsize=(20, 10))
    df["Marital_Status"].value_counts(normalize=True).plot(kind="bar")
    plt.title("Marital status (share of customers)")
    _save_figure(const.EDA_IMAGES["marital_status"])

    plt.figure(figsize=(20, 10))
    sns.histplot(df["Total_Trans_Ct"], stat="density", kde=True)
    plt.title("Total transaction count distribution")
    _save_figure(const.EDA_IMAGES["total_trans_ct"])

    plt.figure(figsize=(20, 10))
    sns.heatmap(df.select_dtypes(include="number").corr(),
                annot=False, cmap="Dark2_r", linewidths=2)
    plt.title("Correlation heatmap")
    _save_figure(const.EDA_IMAGES["heatmap"])

    logger.info("EDA figures saved to %s", const.EDA_DIR)


def encoder_helper(df, category_lst, response):
    """
    Turn each categorical column into a new column holding the churn rate
    (mean of the response) for that category - cell 15 of the notebook.

    For example, "Gender" becomes "Gender_Churn", where every "F" row holds
    the churn rate of all female customers.

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
    output:
            df: pandas dataframe with new encoded columns
    """
    df = df.copy()
    for category in category_lst:
        category_means = df.groupby(category)[response].mean()
        df[f"{category}_{response}"] = df[category].map(category_means)
    return df


def perform_feature_engineering(df, response):
    """
    Encode the categorical columns, select the model features and split the
    data into training and testing sets.

    input:
              df: pandas dataframe
              response: string of response name
    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    encoded_df = encoder_helper(df, const.CAT_COLUMNS, response)
    x_data = encoded_df[const.KEEP_COLUMNS]
    y_data = encoded_df[response]
    return train_test_split(x_data, y_data,
                            test_size=const.TEST_SIZE,
                            random_state=const.RANDOM_STATE)


def _report_image(model_name, train_results, test_results, output_pth):
    """
    Draw the train and test classification reports of one model as text in a
    figure and save it.

    input:
            model_name: name shown in the figure titles
            train_results: tuple (training response values, training predictions)
            test_results: tuple (test response values, test predictions)
            output_pth: path of the image file to write
    output:
            None
    """
    font = {"fontsize": 10, "fontproperties": "monospace"}
    plt.figure(figsize=(6, 6))
    plt.text(0.01, 1.25, f"{model_name} Train", **font)
    plt.text(0.01, 0.7, classification_report(*train_results), **font)
    plt.text(0.01, 0.6, f"{model_name} Test", **font)
    plt.text(0.01, 0.05, classification_report(*test_results), **font)
    plt.axis("off")
    plt.savefig(output_pth, bbox_inches="tight")
    plt.close()


# The signature below is fixed by the project specification.
# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                *,
                                output_dir=const.RESULTS_DIR):
    """
    Produce the classification report for training and testing results and
    store the report as an image in the images/results folder.

    input:
            y_train: training response values
            y_test: test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
            output_dir: folder to write the two report images to
    output:
            None
    """
    os.makedirs(output_dir, exist_ok=True)
    _report_image("Random Forest",
                  (y_train, y_train_preds_rf), (y_test, y_test_preds_rf),
                  os.path.join(output_dir, "rf_results.png"))
    _report_image("Logistic Regression",
                  (y_train, y_train_preds_lr), (y_test, y_test_preds_lr),
                  os.path.join(output_dir, "logistic_results.png"))


def feature_importance_plot(model, x_data, output_pth):
    """
    Create and store the feature importances plot in output_pth.

    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure
    output:
            None
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [x_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=45, ha="right")
    _save_figure(output_pth)


def roc_curve_plot(rfc_model, lr_model, x_test, y_test, output_pth):
    """
    Plot the ROC curves of both models on one figure and save it.

    input:
            rfc_model: trained random forest model
            lr_model: trained logistic regression model
            x_test: X testing data
            y_test: y testing data
            output_pth: path to store the figure
    output:
            None
    """
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    curve_style = {"alpha": 0.8}
    RocCurveDisplay.from_estimator(rfc_model, x_test, y_test, ax=axis,
                                   name="Random Forest",
                                   curve_kwargs=curve_style)
    RocCurveDisplay.from_estimator(lr_model, x_test, y_test, ax=axis,
                                   name="Logistic Regression",
                                   curve_kwargs=curve_style)
    _save_figure(output_pth)


def train_models(x_train, x_test, y_train, y_test):
    """
    Train the models, store the model results (images and scores) and store
    the models.

    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    create_output_directories()

    # Random forest, tuned with a randomized hyperparameter search
    cv_rfc = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=const.RANDOM_STATE),
        param_distributions=const.RF_PARAM_GRID,
        n_iter=const.RF_SEARCH_ITERATIONS,
        cv=const.CV_FOLDS,
        random_state=const.RANDOM_STATE,
        n_jobs=-1,
        error_score="raise",
    )
    cv_rfc.fit(x_train, y_train)
    rfc_model = cv_rfc.best_estimator_
    logger.info("Random forest trained, best params: %s",
                cv_rfc.best_params_)

    # Logistic regression; scaling the features helps the solver converge
    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000)),
    ])
    lr_model.fit(x_train, y_train)
    logger.info("Logistic regression trained")

    # Predictions and evaluation
    classification_report_image(y_train, y_test,
                                lr_model.predict(x_train),
                                rfc_model.predict(x_train),
                                lr_model.predict(x_test),
                                rfc_model.predict(x_test))
    roc_curve_plot(rfc_model, lr_model, x_test, y_test,
                   const.RESULT_IMAGES["roc"])
    feature_importance_plot(rfc_model, x_train,
                            const.RESULT_IMAGES["feature_importance"])
    logger.info("Result images saved to %s", const.RESULTS_DIR)

    # Save the models so they can be loaded later with joblib.load
    joblib.dump(rfc_model, const.RFC_MODEL_PATH)
    joblib.dump(lr_model, const.LR_MODEL_PATH)
    logger.info("Models saved to %s", const.MODELS_DIR)


def main():
    """
    Run the full churn pipeline: import, EDA, feature engineering, training.

    input:
            None
    output:
            None
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    data = import_data(const.DATA_PATH)
    perform_eda(data)
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        data, const.RESPONSE)
    train_models(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    main()
