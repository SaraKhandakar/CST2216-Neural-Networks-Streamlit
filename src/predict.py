# =========================
# Prediction & Input Processing Module
# =========================
# This file handles preparing user input for prediction
# and generating predictions using the trained neural network model.

import logging
import pandas as pd


def prepare_single_input(
    gre_score,
    toefl_score,
    university_rating,
    sop,
    lor,
    cgpa,
    research,
    training_columns,
):
    """
    Prepare a single user input so it matches the format used during training.

    Parameters:
    gre_score, toefl_score, university_rating, sop, lor, cgpa, research:
        User input features
    training_columns (list):
        List of feature columns used during model training

    Returns:
    pd.DataFrame: Processed and encoded input ready for prediction

    Purpose:
    - Convert user input into a DataFrame
    - Apply the same preprocessing steps used during training
    - Ensure feature alignment with trained model input
    """
    try:
        # =========================
        # Create Input DataFrame
        # =========================
        # Convert user input into a structured tabular format
        input_df = pd.DataFrame(
            [
                {
                    "GRE_Score": gre_score,
                    "TOEFL_Score": toefl_score,
                    "University_Rating": university_rating,
                    "SOP": sop,
                    "LOR": lor,
                    "CGPA": cgpa,
                    "Research": research,
                }
            ]
        )

        # =========================
        # Convert Categorical Features
        # =========================
        # Convert categorical variables to object type for encoding
        input_df["University_Rating"] = input_df["University_Rating"].astype("object")
        input_df["Research"] = input_df["Research"].astype("object")

        # =========================
        # One-Hot Encoding
        # =========================
        # Convert categorical variables into numeric format
        input_encoded = pd.get_dummies(input_df, drop_first=True)

        # =========================
        # Align with Training Columns
        # =========================
        # Ensure input has same columns as training data
        # Missing columns are filled with 0
        input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)

        # Log successful preprocessing
        logging.info("Single input prepared successfully")

        return input_encoded

    except Exception as e:
        # Log any unexpected errors
        logging.exception("Error preparing single input: %s", e)
        raise


def make_prediction(model, scaler, input_df):
    """
    Scale input data and generate prediction using trained model.

    Parameters:
    model: Trained neural network model
    scaler: Fitted scaler used during training
    input_df (DataFrame): Processed input data

    Returns:
    tuple:
        prediction: Predicted class label
        probability: Prediction probabilities for each class

    Purpose:
    - Apply scaling to match training conditions
    - Generate prediction and probability scores
    """
    try:
        # =========================
        # Feature Scaling
        # =========================
        # Apply same scaling used during training
        scaled_input = scaler.transform(input_df)

        # =========================
        # Model Prediction
        # =========================
        # Predict class label
        prediction = model.predict(scaled_input)[0]

        # Predict class probabilities (confidence scores)
        probability = model.predict_proba(scaled_input)[0]

        # Log successful prediction
        logging.info("Prediction completed successfully")

        return prediction, probability

    except Exception as e:
        # Log prediction errors
        logging.exception("Error during prediction: %s", e)
        raise