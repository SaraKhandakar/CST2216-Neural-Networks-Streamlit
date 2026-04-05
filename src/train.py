import logging
from sklearn.neural_network import MLPClassifier

from src.config import RANDOM_STATE, DEFAULT_MODEL_NAME, TANH_MODEL_NAME


def train_models(X_train, y_train):
    """
    Train two MLPClassifier models:
    1. Default model
    2. tanh activation model
    """
    try:
        logging.info("Training models started")

        default_model = MLPClassifier(
            hidden_layer_sizes=3,
            batch_size=50,
            max_iter=200,
            random_state=RANDOM_STATE
        )
        default_model.fit(X_train, y_train)

        tanh_model = MLPClassifier(
            hidden_layer_sizes=3,
            batch_size=50,
            max_iter=200,
            activation="tanh",
            random_state=RANDOM_STATE
        )
        tanh_model.fit(X_train, y_train)

        models = {
            DEFAULT_MODEL_NAME: default_model,
            TANH_MODEL_NAME: tanh_model,
        }

        logging.info("Training completed successfully")
        return models

    except Exception as e:
        logging.exception("Error during model training: %s", e)
        raise