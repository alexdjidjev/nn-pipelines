import pathlib
import torch

CURRENT_FILE_DIR = pathlib.Path(__file__).parent
REPO_DIR = CURRENT_FILE_DIR.parent

HYPERPARAM_DB_PATH = REPO_DIR / "db/tuning_results.db"
HYPERPARAM_DB_URI = "sqlite:///" + str(HYPERPARAM_DB_PATH)

GLOBAL_RANDOM_STATE = 42


