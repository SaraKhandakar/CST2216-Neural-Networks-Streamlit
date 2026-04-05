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
    Prepare a single user input row so it matches training columns.
    """
    try:
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

        input_df["University_Rating"] = input_df["University_Rating"].astype("object")
        input_df["Research"] = input_df["Research"].astype("object")

        input_encoded = pd.get_dummies(input_df, drop_first=True)

        input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)

        logging.info("Single input prepared successfully")
        return input_encoded

    except Exception as e:
        logging.exception("Error preparing single input: %s", e)
        raise


def make_prediction(model, scaler, input_df):
    """
    Scale input and make prediction.
    """
    try:
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0]

        logging.info("Prediction completed successfully")
        return prediction, probability

    except Exception as e:
        logging.exception("Error during prediction: %s", e)
        raise