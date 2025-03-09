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
    Generates sample audio mimicking either human or AI voice.
    In cloud environment where microphone is unavailable.
    
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
    # Generate sample audio
    if status_element:
        status_element.text("Generating sample audio...")
        
    st.info("Creating an AI voice sample for demonstration purposes.")
    
    # Choose to generate an AI-like voice sample
    duration = 3  # 3 seconds of audio
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Function to generate a word-like segment
    def generate_word(start_time, duration, base_freq):
        word_t = t[int(start_time*sample_rate):int((start_time+duration)*sample_rate)]
        if len(word_t) == 0:
            return np.array([])
            
        # Very stable pitch (characteristic of AI voices)
        tone = np.sin(2*np.pi*base_freq*word_t) * 0.4
        
        # Add very subtle vibrato (AI voices have minimal variations)
        vibrato = np.sin(2*np.pi*2*word_t) * 0.02  # Very slight vibrato
        tone = tone * (1 + vibrato)
        
        # Add harmonics with perfect ratios (AI voices have very clean harmonics)
        tone += np.sin(2*np.pi*base_freq*2*word_t) * 0.2  # First harmonic
        tone += np.sin(2*np.pi*base_freq*3*word_t) * 0.1  # Second harmonic
        tone += np.sin(2*np.pi*base_freq*4*word_t) * 0.05  # Third harmonic
        
        # Apply a perfect envelope (too perfect, unlike human speech)
        env_length = len(word_t)
        envelope = np.ones(env_length)
        attack = int(0.05 * env_length)  # Short attack
        decay = int(0.1 * env_length)  # Short decay
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-decay:] = np.linspace(1, 0, decay)
        
        return tone * envelope
    
    # Generate a sequence of "words" with unnaturally regular timing
    audio = np.zeros_like(t)
    
    # Create consistent spacing between words (too consistent for human speech)
    words = [
        (0.1, 0.4, 380),    # Word 1: time, duration, frequency
        (0.7, 0.3, 420),    # Word 2
        (1.2, 0.5, 400),    # Word 3
        (1.9, 0.35, 440),   # Word 4
        (2.4, 0.45, 390)    # Word 5
    ]
    
    # Add each word to the audio
    for start, dur, freq in words:
        word_segment = generate_word(start, dur, freq)
        if len(word_segment) > 0:
            end_idx = min(int((start+dur)*sample_rate), len(audio))
            start_idx = int(start*sample_rate)
            segment_length = end_idx - start_idx
            if segment_length > 0:
                audio[start_idx:end_idx] = word_segment[:segment_length]
    
    # Add a subtle background hum (common in AI voices)
    background = np.sin(2*np.pi*50*t) * 0.01
    audio = audio + background
    
    # Apply perfect normalization (too perfect for human speech)
    audio = audio / np.max(np.abs(audio)) * 0.9
    
    # Convert to 16-bit PCM
    audio_data = (audio * 32767).astype(np.int16).tobytes()
    
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
