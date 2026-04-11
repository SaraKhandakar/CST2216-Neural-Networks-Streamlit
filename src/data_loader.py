# =========================
# Data Loading Module
# =========================
# This file contains the function responsible for loading
# the admission dataset used in the neural network project.

import logging
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the admission dataset from a CSV file.

    Parameters:
    file_path (str): Path to the CSV file containing the dataset.

    Returns:
    pd.DataFrame: Loaded dataset as a pandas DataFrame.

    Purpose:
    This function centralizes dataset loading for the project
    and logs success or failure messages to support debugging.
    """
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Log successful data loading
        logging.info("Data loaded successfully from %s", file_path)

        return df

    except FileNotFoundError:
        # Log error if the specified file does not exist
        logging.error("File not found: %s", file_path)
        raise

    except Exception as e:
        # Log any unexpected error during data loading
        logging.exception("Unexpected error while loading data: %s", e)
        raise