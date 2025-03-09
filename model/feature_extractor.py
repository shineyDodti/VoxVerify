import numpy as np
import librosa
import librosa.display
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
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Initialize features dictionary
    features = {}
    
    # 1. Extract pitch (F0) and measure stability
    # AI voices often have unnaturally stable pitch
    pitches, magnitudes = librosa.core.piptrack(y=audio_data, sr=sample_rate)
    pitch_indices = np.argmax(magnitudes, axis=0)
    pitches = pitches[pitch_indices, range(magnitudes.shape[1])]
    pitches = pitches[pitches > 0]  # Remove zero pitches
    
    if len(pitches) > 0:
        # Calculate pitch stability (lower variance = more stable = more AI-like)
        pitch_std = np.std(pitches)
        pitch_mean = np.mean(pitches)
        pitch_cv = pitch_std / pitch_mean if pitch_mean > 0 else 0
        features['pitch_stability'] = 1.0 - min(pitch_cv, 1.0)  # Higher value = more stable
    else:
        features['pitch_stability'] = 0.5  # Default if no pitch detected
    
    # 2. Spectral centroid - center of mass of spectrum
    # AI voices might have different spectral distributions
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
    features['spectral_centroid'] = np.mean(spectral_centroids)
    
    # 3. Spectral flatness - how noise-like vs. tone-like the sound is
    # AI voices often have different tonal characteristics
    spectral_flatness = librosa.feature.spectral_flatness(y=audio_data)[0]
    features['spectral_flatness'] = np.mean(spectral_flatness)
    
    # 4. Harmonic-percussive source separation
    # Calculate harmonic to percussive ratio - AI voices may be more harmonic
    y_harmonic, y_percussive = librosa.effects.hpss(audio_data)
    harmonic_energy = np.sum(y_harmonic**2)
    percussive_energy = np.sum(y_percussive**2)
    total_energy = harmonic_energy + percussive_energy
    if total_energy > 0:
        features['harmonic_ratio'] = harmonic_energy / total_energy
    else:
        features['harmonic_ratio'] = 0.5
    
    # 5. Tempo and rhythm analysis - humans have natural variations
    # Extract tempo and onset strength
    onset_env = librosa.onset.onset_strength(y=audio_data, sr=sample_rate)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sample_rate)
    features['tempo'] = tempo
    
    # Calculate tempo variability through onset strength variance
    # Higher variance indicates more natural rhythm (human-like)
    if len(onset_env) > 0:
        features['tempo_variability'] = np.std(onset_env) / np.mean(onset_env) if np.mean(onset_env) > 0 else 0
    else:
        features['tempo_variability'] = 0.5
    
    # 6. Formant analysis - AI voices might have unnaturally clear formants
    # Use MFCC as a proxy for formant analysis
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    
    # Using the variance in MFCCs as an indicator of formant clarity
    features['formant_clarity'] = 1.0 - min(np.mean(mfcc_std) / 100.0, 1.0)  # Higher = clearer formants
    
    # 7. Jitter and shimmer approximation 
    # (variations in pitch and amplitude - humans have more)
    
    # For shimmer (amplitude variation)
    y_frames = librosa.util.frame(audio_data, frame_length=2048, hop_length=512)
    rms_frames = np.sqrt(np.mean(y_frames**2, axis=0))
    if len(rms_frames) > 1:
        shimmer = np.std(rms_frames) / np.mean(rms_frames) if np.mean(rms_frames) > 0 else 0
        features['shimmer'] = min(shimmer, 1.0)  # Higher = more variation = more human-like
    else:
        features['shimmer'] = 0.5
    
    # 8. Spectral contrast - differences between peaks and valleys
    contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate)
    features['spectral_contrast'] = np.mean(contrast)
    
    # 9. Zero crossing rate - related to frequency content
    zero_crossings = librosa.feature.zero_crossing_rate(audio_data)[0]
    features['zero_crossing_rate'] = np.mean(zero_crossings)
    
    return features
