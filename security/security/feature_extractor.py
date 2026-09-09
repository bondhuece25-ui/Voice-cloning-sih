import librosa
import numpy as np


def extract_features(file_path):
    """
    Extract exactly 84 features.

    40 MFCC means
    40 MFCC standard deviations
    1 spectral centroid mean
    1 spectral bandwidth mean
    1 spectral rolloff mean
    1 zero-crossing-rate mean

    Total = 84 features
    """

    audio, sample_rate = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate
    )

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sample_rate
    )

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sample_rate
    )

    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        y=audio
    )

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),

        [np.mean(spectral_centroid)],
        [np.mean(spectral_bandwidth)],
        [np.mean(spectral_rolloff)],
        [np.mean(zero_crossing_rate)]
    ])

    return features.astype(np.float32)