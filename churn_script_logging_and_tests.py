"""
Tests and logging for the functions in churn_library.py.

Each test calls one library function, checks its output with assertions
(returned data is not empty, expected files were created), and writes an
INFO message on success or an ERROR message on failure to
logs/churn_library.log.

Run from the command line with:
    python churn_script_logging_and_tests.py

Author: Reef Mohammed
Date: 2026-11-06
"""

import logging
import os

import joblib

import churn_library as cls
import constants as const

# The log folder must exist before logging can open a file inside it.
os.makedirs(const.LOGS_DIR, exist_ok=True)
logging.basicConfig(
    filename=const.LOG_FILE,
    level=logging.INFO,
    filemode="w",  # start a fresh log on every run
    format="%(asctime)s %(name)s - %(levelname)s - %(message)s",
)


def _remove_files(paths):
    """
    Delete files left over from earlier runs, so a test can prove that the
    function under test created them again.

    input:
            paths: iterable of file paths
    output:
            None
    """
    for pth in paths:
        if os.path.exists(pth):
            os.remove(pth)


def _assert_files_created(paths):
    """
    Assert that every file in paths exists and is not empty.

    input:
            paths: iterable of file paths
    output:
            None
    """
    for pth in paths:
        assert os.path.isfile(pth), f"{pth} was not created"
        assert os.path.getsize(pth) > 0, f"{pth} is empty"


def test_import(import_data):
    """
    Test import_data: the csv loads and the dataframe has rows, columns and
    the binary response column.

    input:
            import_data: the churn_library.import_data function
    output:
            None
    """
    try:
        df = import_data(const.DATA_PATH)
        logging.info("Testing import_data: SUCCESS - file loaded")
    except FileNotFoundError as err:
        logging.error("Testing import_data: ERROR - file %s was not found",
                      const.DATA_PATH)
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        assert set(df[const.RESPONSE].unique()) <= {0, 1}
        logging.info("Testing import_data: SUCCESS - dataframe has %d rows "
                     "and %d columns", df.shape[0], df.shape[1])
    except AssertionError as err:
        logging.error("Testing import_data: ERROR - the dataframe is empty "
                      "or the %s column is not binary", const.RESPONSE)
        raise err


def test_eda(perform_eda):
    """
    Test perform_eda: every EDA image is created.

    input:
            perform_eda: the churn_library.perform_eda function
    output:
            None
    """
    image_paths = list(const.EDA_IMAGES.values())
    _remove_files(image_paths)
    df = cls.import_data(const.DATA_PATH)

    try:
        perform_eda(df)
    except Exception as err:
        logging.error("Testing perform_eda: ERROR - the function raised %r",
                      err)
        raise err

    try:
        _assert_files_created(image_paths)
        logging.info("Testing perform_eda: SUCCESS - %d EDA images saved "
                     "to %s", len(image_paths), const.EDA_DIR)
    except AssertionError as err:
        logging.error("Testing perform_eda: ERROR - %s", err)
        raise err


def test_encoder_helper(encoder_helper):
    """
    Test encoder_helper: one encoded column is added per categorical column,
    with churn rates between 0 and 1, and the input dataframe is unchanged.

    input:
            encoder_helper: the churn_library.encoder_helper function
    output:
            None
    """
    df = cls.import_data(const.DATA_PATH)
    original_columns = list(df.columns)

    try:
        encoded_df = encoder_helper(df, const.CAT_COLUMNS, const.RESPONSE)
    except KeyError as err:
        logging.error("Testing encoder_helper: ERROR - column %s is missing",
                      err)
        raise err

    try:
        assert encoded_df.shape[0] == df.shape[0]
        for category in const.CAT_COLUMNS:
            new_column = f"{category}_{const.RESPONSE}"
            assert new_column in encoded_df.columns, \
                f"{new_column} was not created"
            assert encoded_df[new_column].notna().all(), \
                f"{new_column} has missing values"
            assert encoded_df[new_column].between(0, 1).all(), \
                f"{new_column} has values outside 0-1"
        assert list(df.columns) == original_columns, \
            "the input dataframe was modified"
        logging.info("Testing encoder_helper: SUCCESS - %d encoded columns "
                     "added", len(const.CAT_COLUMNS))
    except AssertionError as err:
        logging.error("Testing encoder_helper: ERROR - %s", err)
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    """
    Test perform_feature_engineering: the four splits are not empty, have the
    expected feature columns and together contain every row.

    input:
            perform_feature_engineering: the
                churn_library.perform_feature_engineering function
    output:
            None
    """
    df = cls.import_data(const.DATA_PATH)

    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df, const.RESPONSE)
    except Exception as err:
        logging.error("Testing perform_feature_engineering: ERROR - the "
                      "function raised %r", err)
        raise err

    try:
        for split in (x_train, x_test, y_train, y_test):
            assert len(split) > 0, "a train/test split is empty"
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        assert len(x_train) + len(x_test) == len(df), "rows were lost"
        assert list(x_train.columns) == const.KEEP_COLUMNS, \
            "unexpected feature columns"
        logging.info("Testing perform_feature_engineering: SUCCESS - "
                     "%d training rows and %d testing rows",
                     len(x_train), len(x_test))
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: ERROR - %s", err)
        raise err


def test_train_models(train_models):
    """
    Test train_models: both models and every result image are saved, and the
    saved models can be loaded and used to predict.

    input:
            train_models: the churn_library.train_models function
    output:
            None
    """
    model_paths = [const.RFC_MODEL_PATH, const.LR_MODEL_PATH]
    image_paths = list(const.RESULT_IMAGES.values())
    _remove_files(model_paths + image_paths)

    df = cls.import_data(const.DATA_PATH)
    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
        df, const.RESPONSE)

    try:
        train_models(x_train, x_test, y_train, y_test)
    except Exception as err:
        logging.error("Testing train_models: ERROR - the function raised %r",
                      err)
        raise err

    try:
        _assert_files_created(model_paths + image_paths)
        for pth in model_paths:
            predictions = joblib.load(pth).predict(x_test)
            assert len(predictions) == len(x_test), \
                f"{pth} did not predict every test row"
        logging.info("Testing train_models: SUCCESS - %d models and %d "
                     "result images saved, models load and predict",
                     len(model_paths), len(image_paths))
    except AssertionError as err:
        logging.error("Testing train_models: ERROR - %s", err)
        raise err


def test_classification_report_image(classification_report_image):
    """
    Test classification_report_image: both report images are created from
    the saved models' predictions.

    input:
            classification_report_image: the
                churn_library.classification_report_image function
    output:
            None
    """
    image_paths = [const.RESULT_IMAGES["rf_report"],
                   const.RESULT_IMAGES["lr_report"]]
    _remove_files(image_paths)

    df = cls.import_data(const.DATA_PATH)
    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
        df, const.RESPONSE)
    rfc_model = joblib.load(const.RFC_MODEL_PATH)
    lr_model = joblib.load(const.LR_MODEL_PATH)

    try:
        classification_report_image(y_train, y_test,
                                    lr_model.predict(x_train),
                                    rfc_model.predict(x_train),
                                    lr_model.predict(x_test),
                                    rfc_model.predict(x_test))
        _assert_files_created(image_paths)
        logging.info("Testing classification_report_image: SUCCESS - "
                     "report images saved")
    except (AssertionError, ValueError) as err:
        logging.error("Testing classification_report_image: ERROR - %s", err)
        raise err


def test_feature_importance_plot(feature_importance_plot):
    """
    Test feature_importance_plot: the plot is saved for the random forest.

    input:
            feature_importance_plot: the
                churn_library.feature_importance_plot function
    output:
            None
    """
    output_pth = const.RESULT_IMAGES["feature_importance"]
    _remove_files([output_pth])

    df = cls.import_data(const.DATA_PATH)
    x_train, _, _, _ = cls.perform_feature_engineering(df, const.RESPONSE)
    rfc_model = joblib.load(const.RFC_MODEL_PATH)

    try:
        feature_importance_plot(rfc_model, x_train, output_pth)
        _assert_files_created([output_pth])
        logging.info("Testing feature_importance_plot: SUCCESS - plot saved "
                     "to %s", output_pth)
    except (AssertionError, AttributeError) as err:
        logging.error("Testing feature_importance_plot: ERROR - %s", err)
        raise err


def main():
    """
    Run every test in pipeline order. train_models runs before the two plot
    tests because they load the models it saves.

    input:
            None
    output:
            None
    """
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
    test_classification_report_image(cls.classification_report_image)
    test_feature_importance_plot(cls.feature_importance_plot)
    logging.info("All tests finished: SUCCESS")
    print(f"All tests passed. See {const.LOG_FILE} for details.")


if __name__ == "__main__":
    main()
