# =========================
# Model Training Module
# =========================
# This file contains functions for training neural network models
# used in the admission prediction project.

import logging
from sklearn.neural_network import MLPClassifier

from src.config import RANDOM_STATE, DEFAULT_MODEL_NAME, TANH_MODEL_NAME


def train_models(X_train, y_train):
    """
    Train multiple MLPClassifier models for comparison.

    Parameters:
    X_train: Training feature set
    y_train: Training target labels

    Returns:
    dict: Dictionary containing trained neural network models

    Purpose:
    This function trains two neural network models with different
    activation settings so their performance can be compared later.
    """
    try:
        logging.info("Training models started")

        # =========================
        # Default Neural Network Model
        # =========================
        # Train a baseline MLPClassifier using default activation settings
        default_model = MLPClassifier(
            hidden_layer_sizes=3,
            batch_size=50,
            max_iter=200,
            random_state=RANDOM_STATE
        )
        default_model.fit(X_train, y_train)

        # =========================
        # Tanh Activation Model
        # =========================
        # Train a second neural network model using tanh activation
        # to compare its behaviour with the default model
        tanh_model = MLPClassifier(
            hidden_layer_sizes=3,
            batch_size=50,
            max_iter=200,
            activation="tanh",
            random_state=RANDOM_STATE
        )
        tanh_model.fit(X_train, y_train)

        # Store both trained models in a dictionary
        models = {
            DEFAULT_MODEL_NAME: default_model,
            TANH_MODEL_NAME: tanh_model,
        }

        logging.info("Training completed successfully")
        return models

    except Exception as e:
        # Log any unexpected training error
        logging.exception("Error during model training: %s", e)
        raise