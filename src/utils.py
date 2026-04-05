import logging
from pathlib import Path


def setup_logging():
    """
    Configure logging for the project.
    """
    log_folder = Path("logs")
    log_folder.mkdir(exist_ok=True)

    log_file = log_folder / "app.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def format_prediction_label(prediction: int) -> str:
    """
    Convert model output into a readable label.
    """
    return "High chance of admission" if prediction == 1 else "Low chance of admission"