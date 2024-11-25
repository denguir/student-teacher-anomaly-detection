from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    ROOT_DIR = Path(__file__).resolve().parent.parent
    MODEL_PATH = ROOT_DIR / "models"
    DATA_PATH = ROOT_DIR / "data"
