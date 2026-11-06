# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree (Udacity)
- **Repository platform:** GitHub — <https://github.com/Reef-mohammed/cd0580-project-starter>

---

## Project Description

A bank wants to know which credit card customers are likely to leave
(**churn**) so that it can contact them before they do. This project predicts
churn from customer demographics and card-usage behaviour.

The starting point was a Jupyter notebook (`churn_notebook.ipynb`) that already
solved the data-science problem. This project refactors that notebook into a
clean, production-style Python package that is:

- **modular** – each step of the pipeline is a separate, documented function;
- **tested and logged** – every function has a test, and results are written
  to a log file;
- **PEP 8 compliant** – both scripts score 10/10 with `pylint`;
- **runnable from the command line** – no notebook needed.

The machine-learning pipeline:

1. **Loads** the customer data and creates a binary `Churn` target
   (1 = "Attrited Customer", 0 = "Existing Customer").
2. **Explores** the data (EDA) and saves the plots.
3. **Engineers features**: each categorical column is encoded as the churn rate
   of its category (mean-target encoding), then the data is split 70% / 30%
   into training and testing sets.
4. **Trains two models**:
   - a **Random Forest**, tuned with `RandomizedSearchCV` (12 candidates, 3-fold CV);
   - a **Logistic Regression**, with feature scaling (`StandardScaler`).
5. **Evaluates** both models (classification reports, ROC curves, feature
   importances) and **saves** the plots and the trained models.

On the test set, the Random Forest reaches an ROC AUC of about **0.99** and the
Logistic Regression about **0.92**.

---

## Files and Data Description

### Main Files

```
.
├── churn_library.py                   # The pipeline: EDA, features, training, evaluation
├── churn_script_logging_and_tests.py  # Tests for every function in churn_library.py, with logging
├── constants.py                       # Paths, column lists and model settings used by both scripts
├── churn_notebook.ipynb               # Original notebook that this project refactors
├── guide.ipynb                        # Udacity setup and troubleshooting tips
├── requirements.txt                   # Python dependencies
├── data/
│   └── bank_data.csv                  # Input dataset
├── images/
│   ├── eda/                           # EDA plots (created by the pipeline)
│   └── results/                       # Model evaluation plots (created by the pipeline)
├── logs/
│   └── churn_library.log              # Test log (created by the test script)
└── models/                            # Trained models (created by the pipeline)
```

- **`churn_library.py`** – the main library. Its functions follow the order
  `import_data` → `perform_eda` → `perform_feature_engineering` (which calls
  `encoder_helper`) → `train_models` (which calls `classification_report_image`,
  `roc_curve_plot` and `feature_importance_plot`). Run it directly to execute
  the whole pipeline.
- **`churn_script_logging_and_tests.py`** – calls each function in
  `churn_library.py`, checks the results with assertions, and logs `SUCCESS`
  or `ERROR` for each test.
- **`constants.py`** – a single place to change file paths, the list of
  categorical and numeric columns, the train/test split and the random-forest
  hyperparameter grid.
- **`churn_notebook.ipynb`** – the original exploratory notebook, kept for
  reference.

### Data

`data/bank_data.csv` contains **10,127 bank customers** and 21 columns
(plus an unnamed index column):

| Group | Columns |
|---|---|
| Identifier | `CLIENTNUM` (not used for modelling) |
| Target | `Attrition_Flag` – "Existing Customer" or "Attrited Customer" (about 16% churned) |
| Categorical (5) | `Gender`, `Education_Level`, `Marital_Status`, `Income_Category`, `Card_Category` |
| Numeric (14) | `Customer_Age`, `Dependent_count`, `Months_on_book`, `Total_Relationship_Count`, `Months_Inactive_12_mon`, `Contacts_Count_12_mon`, `Credit_Limit`, `Total_Revolving_Bal`, `Avg_Open_To_Buy`, `Total_Amt_Chng_Q4_Q1`, `Total_Trans_Amt`, `Total_Trans_Ct`, `Total_Ct_Chng_Q4_Q1`, `Avg_Utilization_Ratio` |

The models use the 14 numeric columns plus the 5 encoded categorical columns
(`Gender_Churn`, `Education_Level_Churn`, …) – 19 features in total.

---

## Installation

Requires **Python 3.10 or newer**. The project was developed and tested with
Python 3.14.5, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.4, matplotlib 3.11.2,
seaborn 0.13.2, joblib 1.6.0, pylint 4.0.4 and autopep8 2.3.2.

```bash
git clone https://github.com/Reef-mohammed/cd0580-project-starter.git
cd cd0580-project-starter

python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

All commands below must be run from the project root folder, because the
scripts use paths relative to it (for example `./data/bank_data.csv`).

---

## Running the Files

### 1. Run the pipeline

```bash
python churn_library.py
```

This loads the data, saves the EDA plots, encodes the features, trains both
models (about 20 seconds on a laptop), saves the evaluation plots, and saves
the trained models. Progress messages are printed to the console.

### 2. Run the tests and create the log

```bash
python churn_script_logging_and_tests.py
```

This runs one test for each of the 7 functions in `churn_library.py`. Each
test:

- deletes the files that function is expected to create, so it cannot pass
  because of a file left over from an earlier run;
- calls the function and checks the result with assertions – for example, that
  the dataframe is not empty, that the encoded columns hold churn rates between
  0 and 1, that the image files were created, and that the saved models can be
  loaded and used to make predictions;
- writes `Testing <function>: SUCCESS` (level `INFO`) or
  `Testing <function>: ERROR` (level `ERROR`) to `logs/churn_library.log`.

The log file is overwritten on each run. If a test fails, the script stops
with the error and the reason is in the log. The tests also retrain the
models, so the script takes about 25 seconds.

> Run the tests with `python`, not `pytest` – the test functions receive the
> function under test as an argument, which pytest would try to treat as a
> fixture.

### 3. Check code quality

```bash
pylint churn_library.py
pylint churn_script_logging_and_tests.py
autopep8 --in-place --aggressive --aggressive churn_library.py
```

---

## Expected Outputs

| Location | File | Contents |
|---|---|---|
| `images/eda/` | `churn_distribution.png` | Univariate, categorical: count of churned vs. existing customers |
| | `marital_status_distribution.png` | Univariate, categorical: share of customers by marital status |
| | `customer_age_distribution.png` | Univariate, quantitative: histogram of customer age |
| | `total_transaction_distribution.png` | Univariate, quantitative: transaction-count histogram with density curve |
| | `heatmap.png` | Bivariate: correlation heatmap of all numeric columns |
| `images/results/` | `rf_results.png` | Random forest classification report (train and test) |
| | `logistic_results.png` | Logistic regression classification report (train and test) |
| | `roc_curve_result.png` | ROC curves of both models on the test set |
| | `feature_importances.png` | Random forest feature importances |
| `models/` | `rfc_model.pkl` | Best random forest, saved with `joblib` |
| | `logistic_model.pkl` | Logistic regression pipeline (scaler + model), saved with `joblib` |
| `logs/` | `churn_library.log` | Log of the latest test run |

Load a saved model in another program with:

```python
import joblib
model = joblib.load("models/rfc_model.pkl")
predictions = model.predict(features)  # features: the 19 columns in constants.KEEP_COLUMNS
```

---

## Notes

- **Logging:** `churn_library.py` logs its progress with Python's `logging`
  module. When the test script runs, those messages and the test results go
  to the same file, `logs/churn_library.log`.
- **Changes from the notebook:**
  - the repeated encoding code (one block per column) is now a single loop
    in `encoder_helper`, so adding a categorical column only needs a change
    to `CAT_COLUMNS` in `constants.py`;
  - the notebook's random-forest report swapped the "Train" and "Test" labels;
    this is fixed;
  - `RocCurveDisplay` styling was updated for scikit-learn 1.7+;
  - the SHAP explainer plot was left out – the feature-importance plot covers
    the same question, and SHAP is not needed to run the project.
- **Known limitation – target leakage:** as in the notebook, the churn rate
  for each category is calculated on the full dataset *before* the train/test
  split, so information from the test set is used in the training features.
  A production version should calculate these rates on the training data only.
- **Class imbalance:** only about 16% of customers churned. Recall for the
  churn class (about 0.76 for the random forest on the test set) is a more
  useful measure than overall accuracy.
