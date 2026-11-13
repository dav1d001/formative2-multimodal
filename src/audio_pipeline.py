"""Audio feature extraction for one recording (same steps as the Task 3 notebook)."""
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SR = 16000
N_MFCC = 13


def load_audio(path):
    """Load any audio file as a 16 kHz mono signal (uses ffmpeg when installed)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        if shutil.which("ffmpeg"):
            with tempfile.TemporaryDirectory() as folder:
                wav = Path(folder) / "decoded.wav"
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
                                "-ac", "1", "-ar", str(SR), str(wav)], check=True)
                y, _ = librosa.load(wav, sr=SR, mono=True)
        else:
            y, _ = librosa.load(path, sr=SR, mono=True)
    except Exception:
        raise ValueError(f"Could not read {path.name}. Use a .wav file or install ffmpeg.")
    return y


def clean_audio(y, top_db=30, min_burst_s=0.10, max_gap_s=0.6, pad_s=0.05):
    """Keep only the spoken phrase, then normalise the volume."""
    parts = librosa.effects.split(y, top_db=top_db, frame_length=512, hop_length=128)
    parts = [(s, e) for s, e in parts if (e - s) / SR >= min_burst_s]
    if parts:
        groups = [list(parts[0])]
        for s, e in parts[1:]:
            if (s - groups[-1][1]) / SR < max_gap_s:
                groups[-1][1] = e
            else:
                groups.append([s, e])
        s, e = max(groups, key=lambda g: np.sum(y[g[0]:g[1]] ** 2))
        pad = int(pad_s * SR)
        y = y[max(0, s - pad): e + pad]
    peak = np.max(np.abs(y)) if len(y) else 0
    return 0.95 * y / peak if peak > 0 else y


def extract_features(y):
    """MFCCs, delta MFCCs and spectral features, named like audio_features.csv."""
    feats = {"duration_s": len(y) / SR}
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    for i in range(N_MFCC):
        feats[f"mfcc_{i + 1}_mean"] = mfcc[i].mean()
        feats[f"mfcc_{i + 1}_std"] = mfcc[i].std()
    for i in range(N_MFCC):
        feats[f"delta_mfcc_{i + 1}_mean"] = delta[i].mean()
    spectral = {
        "rolloff": librosa.feature.spectral_rolloff(y=y, sr=SR, roll_percent=0.85)[0],
        "rms": librosa.feature.rms(y=y)[0],
        "centroid": librosa.feature.spectral_centroid(y=y, sr=SR)[0],
        "bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=SR)[0],
        "zcr": librosa.feature.zero_crossing_rate(y)[0],
    }
    for name, values in spectral.items():
        feats[f"{name}_mean"] = values.mean()
        feats[f"{name}_std"] = values.std()
    return feats


def audio_file_to_feature_row(path):
    """Return (one-row DataFrame of features, speech length in seconds)."""
    y = clean_audio(load_audio(path))
    seconds = len(y) / SR
    if seconds < 0.3 or not np.any(y):
        raise ValueError(f"No usable speech found in {Path(path).name}. Record it again closer to the microphone.")
    row = pd.DataFrame([extract_features(y)]).round(5)
    return row, seconds
