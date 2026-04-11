# =========================
# Utility Functions Module
# =========================
# This file contains helper functions used across the project,
# including logging configuration and prediction formatting.

import logging
from pathlib import Path


def setup_logging():
    """
    Configure logging for the entire project.

    Purpose:
    - Create a logs directory if it does not exist
    - Store logs in a file for tracking application behavior
    - Also display logs in the console for real-time debugging

    Logging Output:
    - File: logs/app.log
    - Console: Terminal output
    """

    # =========================
    # Create Logs Directory
    # =========================
    # Ensures a folder exists to store log files
    log_folder = Path("logs")
    log_folder.mkdir(exist_ok=True)

    # Define log file path
    log_file = log_folder / "app.log"

    # =========================
    # Configure Logging
    # =========================
    logging.basicConfig(
        level=logging.INFO,  # Log INFO level and above
        format="%(asctime)s - %(levelname)s - %(message)s",  # Standard log format
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),  # Save logs to file
            logging.StreamHandler(),  # Output logs to console
        ],
        force=True,  # Override any previous logging configuration
    )


def format_prediction_label(prediction: int) -> str:
    """
    Convert numerical model prediction into a user-friendly label.

    Parameters:
    prediction (int): Model output (0 or 1)

    Returns:
    str: Human-readable interpretation of prediction

    Purpose:
    Improves user experience by translating model output
    into meaningful text instead of raw numeric values.
    """

    # Convert binary output into readable message
    return "High chance of admission" if prediction == 1 else "Low chance of admission"