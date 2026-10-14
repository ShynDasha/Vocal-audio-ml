import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

audio_path = "/content/vocal.wav"

signal, sr = librosa.load(
    audio_path,
    sr=16000,
    mono=True,
    duration=5
)

signal = librosa.util.normalize(signal)

print(f"Частота дискретизації: {sr} Гц")
print(f"Тривалість запису: {len(signal) / sr:.2f} с")
print(f"Кількість відліків: {len(signal)}")

mel = librosa.feature.melspectrogram(
    y=signal,
    sr=sr,
    n_fft=1024,
    hop_length=512,
    n_mels=128,
    fmin=50,
    fmax=8000
)

mel_db = librosa.power_to_db(mel, ref=np.max)

print(f"Розмір Mel-спектрограми: {mel_db.shape}")

mfcc = librosa.feature.mfcc(
    y=signal,
    sr=sr,
    n_mfcc=20
)

spectral_centroid = librosa.feature.spectral_centroid(
    y=signal,
    sr=sr
)

rms = librosa.feature.rms(y=signal)

zero_crossing_rate = librosa.feature.zero_crossing_rate(signal)

print("\nХарактеристики вокального запису:")
print(f"Середній спектральний центр: "
      f"{np.mean(spectral_centroid):.2f} Гц")

print(f"Середня RMS-енергія: "
      f"{np.mean(rms):.4f}")

print(f"Zero Crossing Rate: "
      f"{np.mean(zero_crossing_rate):.4f}")

print(f"Кількість MFCC-коефіцієнтів: "
      f"{mfcc.shape[0]}")

plt.figure(figsize=(11, 4))

librosa.display.specshow(
    mel_db,
    sr=sr,
    hop_length=512,
    x_axis="time",
    y_axis="mel",
    fmax=8000
)

plt.colorbar(format="%+2.0f dB")
plt.title("Mel-спектрограма вокального аудіозапису")
plt.xlabel("Час, с")
plt.ylabel("Частота, Гц")

plt.tight_layout()
plt.show()