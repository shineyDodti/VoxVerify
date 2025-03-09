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
    try:
        # Load audio file with error handling for various formats
        try:
            # First try standard librosa load
            audio_data, sample_rate = librosa.load(file_path, sr=None, res_type='kaiser_fast')
        except Exception as e:
            # If that fails, try loading with scipy (handles more formats)
            import scipy.io.wavfile as wav
            sample_rate, audio_data = wav.read(file_path)
            # Convert to float32 for librosa compatibility
            audio_data = audio_data.astype(np.float32)
            # Normalize to -1 to 1 range
            if audio_data.dtype == np.int16:
                audio_data = audio_data / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data / 2147483648.0
            elif audio_data.dtype == np.uint8:
                audio_data = (audio_data.astype(np.float32) - 128) / 128.0
        
        # If stereo, convert to mono
        if len(audio_data.shape) > 1:
            if audio_data.shape[1] == 2:  # If stereo (2 channels)
                audio_data = librosa.to_mono(audio_data.T)  # Transpose if needed for librosa
            else:
                # Just take the first channel if multi-channel
                audio_data = audio_data[:, 0]
        
        # Ensure audio data is not empty
        if len(audio_data) == 0 or np.max(np.abs(audio_data)) == 0:
            raise ValueError("Audio file contains no data or is silent")
            
        # Normalize audio (with protection against division by zero)
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        
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
        
    except Exception as e:
        # Re-raise with a more informative error message
        raise Exception(f"Failed to process audio file: {str(e)}. Make sure the file is a valid audio format.")

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
            
        # Perfectly stable pitch (characteristic of AI voices)
        tone = np.sin(2*np.pi*base_freq*word_t) * 0.4
        
        # No vibrato at all - completely mechanical sound
        # This is an obvious sign of AI-generated voice
        
        # Add harmonics with mathematically perfect ratios
        tone += np.sin(2*np.pi*base_freq*2*word_t) * 0.2  # First harmonic
        tone += np.sin(2*np.pi*base_freq*3*word_t) * 0.1  # Second harmonic
        tone += np.sin(2*np.pi*base_freq*4*word_t) * 0.05  # Third harmonic
        
        # Apply a perfectly symmetrical envelope (unnatural in real speech)
        env_length = len(word_t)
        envelope = np.ones(env_length)
        attack = int(0.05 * env_length)  # Short attack
        decay = int(0.05 * env_length)  # Identical decay (unnatural symmetry)
        
        # Perfectly linear attack and decay (too perfect for human speech)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[-decay:] = np.linspace(1, 0, decay)
        
        return tone * envelope
    
    # Generate a sequence of "words" with unnaturally regular timing
    audio = np.zeros_like(t)
    
    # Create extremely consistent and robotic spacing between words
    # Make it even more AI-like by having perfectly equal distances and frequencies
    # This is a distinctive characteristic of synthetic speech
    words = [
        (0.1, 0.35, 400),    # Word 1: time, duration, frequency (constant frequency)
        (0.6, 0.35, 400),    # Word 2: perfect timing and same pitch 
        (1.1, 0.35, 400),    # Word 3: perfect timing and same pitch
        (1.6, 0.35, 400),    # Word 4: perfect timing and same pitch
        (2.1, 0.35, 400)     # Word 5: perfect timing and same pitch
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
    try:
        # Read audio from byte stream
        with io.BytesIO(audio_bytes) as wav_io:
            try:
                # First try with soundfile
                audio_data, _ = sf.read(wav_io)
            except Exception as e:
                # If that fails, try with wave + numpy
                wav_io.seek(0)  # Reset position to beginning of file
                with wave.open(wav_io, 'rb') as wf:
                    n_frames = wf.getnframes()
                    frames = wf.readframes(n_frames)
                    width = wf.getsampwidth()
                    
                    # Convert based on sample width
                    if width == 1:  # 8-bit unsigned
                        dtype = np.uint8
                        audio_data = np.frombuffer(frames, dtype=dtype)
                        audio_data = (audio_data.astype(np.float32) - 128) / 128.0
                    elif width == 2:  # 16-bit signed
                        dtype = np.int16
                        audio_data = np.frombuffer(frames, dtype=dtype)
                        audio_data = audio_data.astype(np.float32) / 32768.0
                    elif width == 4:  # 32-bit signed
                        dtype = np.int32
                        audio_data = np.frombuffer(frames, dtype=dtype)
                        audio_data = audio_data.astype(np.float32) / 2147483648.0
                    else:
                        raise ValueError(f"Unsupported sample width: {width}")
        
        # Ensure mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Ensure audio data is not empty
        if len(audio_data) == 0:
            raise ValueError("Audio data is empty")
            
        # Normalize (with protection against division by zero)
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
        
        # Trim silence
        audio_data, _ = librosa.effects.trim(audio_data, top_db=20)
        
        return audio_data
    
    except Exception as e:
        # In case of error, generate a fallback tone
        st.error(f"Error processing audio: {str(e)}. Using fallback audio.")
        
        # Generate a simple sine wave as fallback
        duration = 3  # seconds
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * 440 * t) * 0.5  # 440 Hz tone
        
        return audio_data
