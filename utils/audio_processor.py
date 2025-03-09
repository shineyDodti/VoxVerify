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
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    # Set up the audio stream
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=sample_rate,
        input=True,
        frames_per_buffer=1024
    )
    
    # Start recording
    frames = []
    start_time = time.time()
    
    try:
        while time.time() - start_time < max_duration:
            # Read audio data
            data = stream.read(1024)
            frames.append(data)
            
            # Update status if provided
            if status_element:
                elapsed = time.time() - start_time
                status_element.text(f"Recording: {elapsed:.1f}s / {max_duration}s")
    
    except Exception as e:
        st.error(f"Error during recording: {str(e)}")
    
    finally:
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()
    
    # Combine all frames into a single byte array
    audio_data = b''.join(frames)
    
    # Convert to WAV format in memory
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
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
