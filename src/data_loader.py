import logging
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the admission dataset from CSV.
    """
    try:
        df = pd.read_csv(file_path)
        logging.info("Data loaded successfully from %s", file_path)
        return df
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        raise
    except Exception as e:
        logging.exception("Unexpected error while loading data: %s", e)
        raise