"""Compatibility entry point: extract local features, then train models."""
from audio_features import main as extract_main
from train import main as train_main

if __name__ == '__main__':
    extract_main()
    train_main()
