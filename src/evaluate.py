# =========================
# Model Evaluation Module
# =========================
# This file contains functions used to evaluate the performance
# of the trained neural network model on unseen test data.

import logging
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model using test data.

    Parameters:
    model: Trained machine learning model
    X_test: Test feature set
    y_test: True target labels for the test set

    Returns:
    dict: Dictionary containing predictions and evaluation metrics

    Purpose:
    This function measures how well the trained model performs
    on unseen data using classification metrics such as accuracy,
    confusion matrix, and classification report.
    """
    try:
        # Generate predicted class labels for the test set
        y_pred = model.predict(X_test)

        # Store evaluation metrics and predictions in a dictionary
        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
            "y_pred": y_pred,
        }

        # Log successful model evaluation
        logging.info("Model evaluated successfully")
        return results

    except Exception as e:
        # Log any unexpected error during evaluation
        logging.exception("Error during model evaluation: %s", e)
        raise