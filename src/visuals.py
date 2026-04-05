import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(model, X_test, y_test):
    """
    Create a confusion matrix figure.
    """
    try:
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        logging.info("Confusion matrix plotted successfully")
        return fig
    except Exception as e:
        logging.exception("Error plotting confusion matrix: %s", e)
        raise


def plot_loss_curve(model, model_name="Model"):
    """
    Create a loss curve figure for an MLP model.
    """
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(model.loss_curve_)
        ax.set_title(f"Loss Curve - {model_name}")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")
        ax.grid(True)
        plt.tight_layout()
        logging.info("Loss curve plotted successfully")
        return fig
    except Exception as e:
        logging.exception("Error plotting loss curve: %s", e)
        raise