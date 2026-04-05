import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    TARGET_COLUMN,
    TARGET_THRESHOLD,
    DROP_COLUMNS,
    CATEGORICAL_COLUMNS,
    TEST_SIZE,
    RANDOM_STATE,
)


def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the dataset:
    - Create binary target
    - Drop unnecessary columns
    - Convert categorical variables
    - Apply one-hot encoding
    - Split into train/test
    - Scale features
    """

    try:
        logging.info("Starting preprocessing")

        # -----------------------------
        # Step 1: Create binary target
        # -----------------------------
        df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(
            lambda x: 1 if x >= TARGET_THRESHOLD else 0
        )

        # -----------------------------
        # Step 2: Drop columns
        # -----------------------------
        df = df.drop(columns=DROP_COLUMNS)

        # -----------------------------
        # Step 3: Convert to categorical
        # -----------------------------
        for col in CATEGORICAL_COLUMNS:
            df[col] = df[col].astype("category")

        # -----------------------------
        # Step 4: One-hot encoding
        # -----------------------------
        df = pd.get_dummies(
            df,
            columns=CATEGORICAL_COLUMNS,
            dtype=int
        )

        # -----------------------------
        # Step 5: Split features & target
        # -----------------------------
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        # -----------------------------
        # Step 6: Train-test split
        # -----------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_STATE,
        )

        # -----------------------------
        # Step 7: Scaling
        # -----------------------------
        scaler = MinMaxScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        logging.info("Preprocessing completed successfully")

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler

    except Exception as e:
        logging.exception("Error during preprocessing: %s", e)
        raise