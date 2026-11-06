"""
Constants shared by the churn prediction pipeline and its tests.

Keeping paths, column lists and hyperparameters here (instead of inside
functions) means they can be changed in one place without touching the logic.

Author: Reef Mohammed
Date: 2026-11-06
"""

# ---------------------------------------------------------------- paths
DATA_PATH = "./data/bank_data.csv"
EDA_DIR = "./images/eda"
RESULTS_DIR = "./images/results"
MODELS_DIR = "./models"
LOGS_DIR = "./logs"
LOG_FILE = "./logs/churn_library.log"

RFC_MODEL_PATH = f"{MODELS_DIR}/rfc_model.pkl"
LR_MODEL_PATH = f"{MODELS_DIR}/logistic_model.pkl"

# ---------------------------------------------------------------- columns
RESPONSE = "Churn"

CAT_COLUMNS = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category",
]

QUANT_COLUMNS = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",
]

# Model features = numeric columns + one encoded column per categorical column
KEEP_COLUMNS = QUANT_COLUMNS + [f"{col}_{RESPONSE}" for col in CAT_COLUMNS]

# ---------------------------------------------------------------- modelling
TEST_SIZE = 0.3
RANDOM_STATE = 42

RF_PARAM_GRID = {
    "n_estimators": [200, 300],
    "max_features": ["sqrt"],
    "max_depth": [5, 8, 10],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [2, 4],
    "criterion": ["gini"],
}
RF_SEARCH_ITERATIONS = 12
CV_FOLDS = 3

# ---------------------------------------------------------------- output files
EDA_IMAGES = {
    "churn": f"{EDA_DIR}/churn_distribution.png",
    "customer_age": f"{EDA_DIR}/customer_age_distribution.png",
    "marital_status": f"{EDA_DIR}/marital_status_distribution.png",
    "total_trans_ct": f"{EDA_DIR}/total_transaction_distribution.png",
    "heatmap": f"{EDA_DIR}/heatmap.png",
}

RESULT_IMAGES = {
    "rf_report": f"{RESULTS_DIR}/rf_results.png",
    "lr_report": f"{RESULTS_DIR}/logistic_results.png",
    "roc": f"{RESULTS_DIR}/roc_curve_result.png",
    "feature_importance": f"{RESULTS_DIR}/feature_importances.png",
}
