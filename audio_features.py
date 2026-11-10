"""Extract and cache audio features; importing this module does not run work."""
import argparse
import hashlib
import json
import librosa
import numpy as np
import pandas as pd
from config import RAW_DIR, MANIFEST, FEATURES, SAMPLE_RATE, DURATION, SEED

def load_vocal_audio(path):
    signal, sr = librosa.load(
        path, sr=SAMPLE_RATE,
        mono=True, duration=DURATION
    )

    if len(signal) == 0:
        raise ValueError("Порожній аудіозапис")

    target = SAMPLE_RATE * DURATION
    signal = librosa.util.fix_length(signal, size=target)
    signal = librosa.util.normalize(signal)

    return signal, sr


def create_mel_spectrogram(signal, sr):
    mel = librosa.feature.melspectrogram(
        y=signal, sr=sr,
        n_fft=1024, hop_length=512,
        n_mels=128
    )
    return librosa.power_to_db(mel, ref=np.max)


def extract_audio_features(signal, sr):
    mfcc = librosa.feature.mfcc(
        y=signal, sr=sr, n_mfcc=20
    )

    centroid = librosa.feature.spectral_centroid(
        y=signal, sr=sr
    )
    bandwidth = librosa.feature.spectral_bandwidth(
        y=signal, sr=sr
    )
    rms = librosa.feature.rms(y=signal)
    zcr = librosa.feature.zero_crossing_rate(signal)

    features = {
        f"mfcc_{i+1}": float(value)
        for i, value in enumerate(np.mean(mfcc, axis=1))
    }

    features.update({
        "spectral_centroid": float(np.mean(centroid)),
        "spectral_bandwidth": float(np.mean(bandwidth)),
        "rms": float(np.mean(rms)),
        "zero_crossing_rate": float(np.mean(zcr))
    })

    return features


def estimate_pitch(signal, sr):
    f0, voiced, probability = librosa.pyin(
        signal, sr=sr,
        fmin=65, fmax=1000
    )

    valid = f0[np.isfinite(f0)]

    if len(valid) == 0:
        return {
            "f0_mean": 0.0,
            "f0_std": 0.0,
            "voiced_ratio": 0.0
        }

    return {
        "f0_mean": float(np.mean(valid)),
        "f0_std": float(np.std(valid)),
        "voiced_ratio": float(np.mean(voiced))
    }


def process_audio(path):
    signal, sr = load_vocal_audio(path)

    features = extract_audio_features(signal, sr)
    features.update(estimate_pitch(signal, sr))

    return features



def select_records(manifest, max_files):
    if max_files is None or max_files >= len(manifest):
        return manifest.copy()
    groups = [group.sample(frac=1, random_state=SEED)
              for _, group in manifest.groupby(['technique', 'singer'], sort=True)]
    if max_files < len(groups):
        raise ValueError(f'Для охоплення всіх пар техніка/виконавець потрібно мінімум {len(groups)} файлів. Збільште --max-files або використайте --all.')
    indices = []
    for position in range(max(map(len, groups))):
        for group in groups:
            if position < len(group):
                indices.append(group.index[position])
                if len(indices) == max_files:
                    return manifest.loc[indices].copy()
    return manifest.loc[indices].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument('--max-files', type=int, default=200)
    choice.add_argument('--all', action='store_true')
    args = parser.parse_args()
    if args.max_files <= 0:
        parser.error('--max-files має бути додатним')
    if not MANIFEST.exists():
        parser.error('Спочатку запустіть prepare_vocalset.py')
    manifest = pd.read_csv(MANIFEST)
    selected = select_records(manifest, None if args.all else args.max_files)
    rows = []
    for number, record in enumerate(selected.to_dict('records'), 1):
        path = (RAW_DIR / record['path']).resolve()
        if not path.is_relative_to(RAW_DIR.resolve()):
            raise ValueError(f'Шлях за межами raw: {path}')
        print(f'[{number}/{len(selected)}] {record["path"]}', flush=True)
        try:
            features = process_audio(path)
        except Exception as error:
            raise RuntimeError(f'Не вдалося обробити {path}; таблицю не оновлено') from error
        if not all(np.isfinite(value) for value in features.values()):
            raise ValueError(f'Некоректні ознаки: {path}')
        rows.append({**record, **features})
    FEATURES.parent.mkdir(parents=True, exist_ok=True)
    temporary = FEATURES.with_suffix('.csv.part')
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(FEATURES)
    metadata = dict(sample_rate=SAMPLE_RATE, duration=DURATION, seed=SEED,
                    feature_version=1, records=len(rows),
                    manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
                    features_sha256=hashlib.sha256(FEATURES.read_bytes()).hexdigest(),
                    librosa_version=librosa.__version__,
                    selection='all' if args.all else 'round-robin technique/singer',
                    max_files=None if args.all else args.max_files)
    FEATURES.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Ознаки збережено:', FEATURES)


if __name__ == '__main__':
    main()
