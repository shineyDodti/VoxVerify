import numpy as np
import librosa
import warnings

# Suppress specific librosa warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")

def extract_features(audio_data, sample_rate):
    """
    Extract acoustic features from audio data that help differentiate 
    between AI-generated and human voices.
    
    Parameters:
    -----------
    audio_data : numpy array
        The audio time series
    sample_rate : int
        The sample rate of the audio
    
    Returns:
    --------
    features : dict
        Dictionary of extracted features
    """
    # Ensure audio is mono
    if len(audio_data.shape) > 1:
        audio_data = librosa.to_mono(audio_data)
    
    # Normalize audio
    audio_data = audio_data / np.max(np.abs(audio_data)) if np.max(np.abs(audio_data)) > 0 else audio_data
    
    # Trim leading and trailing silence
    audio_data, _ = librosa.effects.trim(audio_data, top_db=30)
    
    # Apply a pre-emphasis filter to emphasize higher frequencies
    pre_emphasis = 0.97
    audio_data = np.append(audio_data[0], audio_data[1:] - pre_emphasis * audio_data[:-1])
    
    # Initialize features dictionary
    features = {}
    
    # 1. Pitch Stability
    try:
        f0, _, _ = librosa.pyin(audio_data, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sample_rate)
        f0 = f0[~np.isnan(f0)]
        if len(f0) > 0:
            pitch_std = np.std(f0)
            pitch_mean = np.mean(f0)
            pitch_cv = pitch_std / pitch_mean if pitch_mean > 0 else 0
            features['pitch_stability'] = 1.0 - min(pitch_cv, 1.0)
        else:
            features['pitch_stability'] = 0.5
    except Exception:
        features['pitch_stability'] = 0.5
    
    # 2. Spectral Centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
    features['spectral_centroid'] = np.mean(spectral_centroids)
    
    # 3. Spectral Flatness
    spectral_flatness = librosa.feature.spectral_flatness(y=audio_data)[0]
    features['spectral_flatness'] = np.mean(spectral_flatness)
    
    # 4. Harmonic Ratio
    y_harmonic, y_percussive = librosa.effects.hpss(audio_data)
    harmonic_energy = np.sum(y_harmonic**2)
    percussive_energy = np.sum(y_percussive**2)
    total_energy = harmonic_energy + percussive_energy
    features['harmonic_ratio'] = harmonic_energy / total_energy if total_energy > 0 else 0.5
    
    # 5. Tempo Variability
    onset_env = librosa.onset.onset_strength(y=audio_data, sr=sample_rate)
    if len(onset_env) > 0:
        features['tempo_variability'] = np.std(onset_env) / np.mean(onset_env) if np.mean(onset_env) > 0 else 0
    else:
        features['tempo_variability'] = 0.5
    
    # 6. Formant Clarity
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    mfcc_std = np.std(mfccs, axis=1)
    features['formant_clarity'] = 1.0 - min(np.mean(mfcc_std) / 100.0, 1.0)
    
    # 7. Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
    features['spectral_contrast'] = np.mean(contrast)
    
    # 8. Zero Crossing Rate
    zero_crossings = librosa.feature.zero_crossing_rate(audio_data)[0]
    features['zero_crossing_rate'] = np.mean(zero_crossings)
    
    # Normalize features
    for key in features:
        features[key] = min(max(features[key], 0.0), 1.0)  # Clamp values between 0 and 1
    
    return features
