# =========================
# Visualization Module
# =========================
# This file contains functions to visualize model performance,
# including confusion matrix and neural network training loss.

import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(model, X_test, y_test):
    """
    Generate a confusion matrix visualization.

    Parameters:
    model: Trained classification model
    X_test: Test feature set
    y_test: True labels for test data

    Returns:
    matplotlib.figure.Figure: Confusion matrix figure

    Purpose:
    The confusion matrix helps evaluate classification performance
    by showing correct and incorrect predictions for each class.
    """
    try:
        # =========================
        # Create Plot
        # =========================
        fig, ax = plt.subplots(figsize=(5, 4))

        # Generate confusion matrix directly from model
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)

        # Set title and layout
        ax.set_title("Confusion Matrix")
        plt.tight_layout()

        # Log success
        logging.info("Confusion matrix plotted successfully")

        return fig

    except Exception as e:
        # Log any plotting error
        logging.exception("Error plotting confusion matrix: %s", e)
        raise


def plot_loss_curve(model, model_name="Model"):
    """
    Generate a loss curve for a trained neural network model.

    Parameters:
    model: Trained MLPClassifier model
    model_name (str): Name of the model (for display purposes)

    Returns:
    matplotlib.figure.Figure: Loss curve figure

    Purpose:
    The loss curve shows how the model's error decreases during training.
    It helps assess whether the model is learning properly or overfitting.
    """
    try:
        # =========================
        # Create Plot
        # =========================
        fig, ax = plt.subplots(figsize=(6, 4))

        # Plot loss values recorded during training iterations
        ax.plot(model.loss_curve_)

        # Label axes and title
        ax.set_title(f"Loss Curve - {model_name}")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")

        # Add grid for better readability
        ax.grid(True)

        # Adjust layout
        plt.tight_layout()

        # Log success
        logging.info("Loss curve plotted successfully")

        return fig

    except Exception as e:
        # Log plotting error
        logging.exception("Error plotting loss curve: %s", e)
        raise