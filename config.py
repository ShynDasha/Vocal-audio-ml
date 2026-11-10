"""Shared local paths and reproducible experiment settings."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / 'data'
ARCHIVE = DATA_DIR / 'archives' / 'VocalSet11.zip'
RAW_DIR = DATA_DIR / 'raw'
MANIFEST = DATA_DIR / 'manifests' / 'vocalset_manifest.csv'
FEATURES = DATA_DIR / 'features' / 'vocal_features.csv'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'
SAMPLE_RATE = 16000
DURATION = 5
SEED = 42
