from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Data directory (customize if needed)
DATA_DIR = BASE_DIR / "data"

# Model directory
MODEL_DIR = BASE_DIR / "models"

# Path to saved model (dummy placeholder)
MODEL_PATH = MODEL_DIR / "best_model.pkl"
