import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def load_vocal_audio(file_path, sr=16000, duration=5):
    signal, _ = librosa.load(
        file_path,
        sr=sr,
        mono=True,
        duration=duration
    )

    signal = librosa.util.normalize(signal)

    return signal, sr

def create_mel_spectrogram(signal, sr):
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sr,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
        fmin=50,
        fmax=8000
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return mel_db


def extract_audio_features(signal, sr):

    # MFCC — характеристики спектральної форми сигналу
    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=20
    )

    # Спектральний центр
    spectral_centroid = librosa.feature.spectral_centroid(
        y=signal,
        sr=sr
    )

    # Спектральна ширина
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=signal,
        sr=sr
    )

    # Частота переходів сигналу через нуль
    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        signal
    )

    # RMS — енергія сигналу
    rms = librosa.feature.rms(
        y=signal
    )

    return {
        "mfcc": np.mean(mfcc, axis=1),
        "spectral_centroid": np.mean(spectral_centroid),
        "spectral_bandwidth": np.mean(spectral_bandwidth),
        "zero_crossing_rate": np.mean(zero_crossing_rate),
        "rms": np.mean(rms)
    }



def estimate_pitch(signal, sr):

    f0, voiced_flag, voiced_prob = librosa.pyin(
        signal,
        fmin=65,
        fmax=1000,
        sr=sr
    )

    # Беремо тільки ділянки, де визначена основна частота
    valid_f0 = f0[~np.isnan(f0)]

    if len(valid_f0) > 0:
        mean_f0 = np.mean(valid_f0)
        min_f0 = np.min(valid_f0)
        max_f0 = np.max(valid_f0)
    else:
        mean_f0 = min_f0 = max_f0 = 0

    return mean_f0, min_f0, max_f0



def create_feature_vector(features, pitch):

    mean_f0, min_f0, max_f0 = pitch

    feature_vector = np.concatenate([
        features["mfcc"],
        [
            features["spectral_centroid"],
            features["spectral_bandwidth"],
            features["zero_crossing_rate"],
            features["rms"],
            mean_f0,
            min_f0,
            max_f0
        ]
    ])

    return feature_vector



audio_path = "/content/vocal.wav"

signal, sr = load_vocal_audio(audio_path)

mel_db = create_mel_spectrogram(
    signal,
    sr
)

features = extract_audio_features(
    signal,
    sr
)

pitch = estimate_pitch(
    signal,
    sr
)

feature_vector = create_feature_vector(
    features,
    pitch
)


print("=== ПАРАМЕТРИ АУДІО ===")

print(f"Частота дискретизації: {sr} Гц")
print(f"Тривалість: {len(signal) / sr:.2f} с")
print(f"Кількість відліків: {len(signal)}")

print("\n=== СПЕКТРАЛЬНІ ХАРАКТЕРИСТИКИ ===")

print(
    f"Спектральний центр: "
    f"{features['spectral_centroid']:.2f} Гц"
)

print(
    f"Спектральна ширина: "
    f"{features['spectral_bandwidth']:.2f} Гц"
)

print(
    f"Zero Crossing Rate: "
    f"{features['zero_crossing_rate']:.4f}"
)

print(
    f"RMS-енергія: "
    f"{features['rms']:.4f}"
)

print("\n=== ОСНОВНА ЧАСТОТА ГОЛОСУ ===")

print(f"Середня F0: {pitch[0]:.2f} Гц")
print(f"Мінімальна F0: {pitch[1]:.2f} Гц")
print(f"Максимальна F0: {pitch[2]:.2f} Гц")

print("\n=== ML-ПРЕДСТАВЛЕННЯ ===")

print(
    "Кількість ознак у векторі:",
    len(feature_vector)
)

print(
    "Розмір Mel-спектрограми:",
    mel_db.shape
)


plt.figure(figsize=(12, 4))

librosa.display.specshow(
    mel_db,
    sr=sr,
    hop_length=512,
    x_axis="time",
    y_axis="mel",
    fmax=8000
)

plt.colorbar(format="%+2.0f dB")

plt.title(
    "Mel-спектрограма вокального аудіозапису"
)

plt.xlabel("Час, с")
plt.ylabel("Частота, Гц")

plt.tight_layout()
plt.show()