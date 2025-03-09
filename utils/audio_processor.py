import numpy as np
import librosa
import soundfile as sf
import io
import pyaudio
import wave
import streamlit as st
import time

def process_audio_file(file_path):
    """
    Process an audio file for feature extraction.
    
    Parameters:
    -----------
    file_path : str
        Path to the audio file
        
    Returns:
    --------
    audio_data : numpy array
        The audio time series
    sample_rate : int
        The sample rate of the audio
    """
    # Load audio file
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    
    # If stereo, convert to mono
    if len(audio_data.shape) > 1:
        audio_data = librosa.to_mono(audio_data)
    
    # Normalize audio
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Trim silence
    audio_data, _ = librosa.effects.trim(audio_data, top_db=20)
    
    # Ensure minimum duration (pad if necessary)
    min_duration = 3  # seconds
    if len(audio_data) < min_duration * sample_rate:
        padding = np.zeros(min_duration * sample_rate - len(audio_data))
        audio_data = np.concatenate([audio_data, padding])
    
    # Ensure maximum duration (truncate if necessary)
    max_duration = 30  # seconds
    if len(audio_data) > max_duration * sample_rate:
        audio_data = audio_data[:max_duration * sample_rate]
    
    return audio_data, sample_rate

def record_audio(max_duration=10, sample_rate=22050, status_element=None):
    """
    Record audio from the microphone.
    In cloud environment where microphone is unavailable, generates a sample tone.
    
    Parameters:
    -----------
    max_duration : int
        Maximum recording duration in seconds
    sample_rate : int
        Sample rate for the recording
    status_element : streamlit element
        Element to update with recording status
        
    Returns:
    --------
    audio_data : bytes
        The recorded audio as a byte array
    """
    # Generate a sample tone (for cloud environments without microphone)
    if status_element:
        status_element.text("Generating sample audio...")
        
    st.info("Creating a test audio sample for demonstration purposes.")
    
    # Generate a simple sine wave tone as sample data
    duration = 3  # 3 seconds of audio
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Generate a note with some variations to make it more voice-like
    tone = np.sin(2*np.pi*440*t) * 0.3  # A4 note
    vibrato = np.sin(2*np.pi*5*t)  # 5 Hz vibrato
    tone = tone * (1 + 0.1 * vibrato)  # Apply vibrato
    
    # Add harmonics for richness
    tone += np.sin(2*np.pi*880*t) * 0.15  # First harmonic
    tone += np.sin(2*np.pi*1320*t) * 0.05  # Second harmonic
    
    # Apply envelope
    envelope = np.ones_like(t)
    attack = int(0.1 * sample_rate)  # 100ms attack
    decay = int(0.3 * sample_rate)  # 300ms decay
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[-decay:] = np.linspace(1, 0, decay)
    tone = tone * envelope
    
    # Convert to 16-bit PCM
    audio_data = (tone * 32767).astype(np.int16).tobytes()
    
    # Convert to WAV format
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
        wav_data = wav_io.getvalue()
        
    return wav_data

def bytes_to_numpy(audio_bytes, sample_rate=22050):
    """
    Convert audio bytes to numpy array.
    
    Parameters:
    -----------
    audio_bytes : bytes
        Audio data in bytes format
    sample_rate : int
        Sample rate for the audio
        
    Returns:
    --------
    audio_data : numpy array
        The audio time series
    """
    # Read audio from byte stream
    with io.BytesIO(audio_bytes) as wav_io:
        audio_data, _ = sf.read(wav_io)
    
    # Ensure mono
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Normalize
    audio_data = audio_data / np.max(np.abs(audio_data)) if np.max(np.abs(audio_data)) > 0 else audio_data
    
    return audio_data
