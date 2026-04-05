import logging
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data.
    Returns predictions and evaluation metrics.
    """
    try:
        y_pred = model.predict(X_test)

        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "classification_report": classification_report(y_test, y_pred),
            "y_pred": y_pred,
        }

        logging.info("Model evaluated successfully")
        return results

    except Exception as e:
        logging.exception("Error during model evaluation: %s", e)
        raise