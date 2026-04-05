from pathlib import Path

# Base project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Data path
DATA_PATH = BASE_DIR / "data" / "Admission.csv"

# Model settings
TARGET_COLUMN = "Admit_Chance"
TARGET_THRESHOLD = 0.8
DROP_COLUMNS = ["Serial_No"]
CATEGORICAL_COLUMNS = ["University_Rating", "Research"]

# Split settings
TEST_SIZE = 0.2
RANDOM_STATE = 123

# Model names
DEFAULT_MODEL_NAME = "Default MLPClassifier"
TANH_MODEL_NAME = "MLPClassifier (tanh)"