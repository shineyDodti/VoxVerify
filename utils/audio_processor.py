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
    Process an audio file for feature extraction.  Handles more audio formats.

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
        # Try loading with soundfile first (handles many formats)
        audio_data, sample_rate = sf.read(file_path)

        # If stereo, convert to mono
        if len(audio_data.shape) > 1:
            audio_data = librosa.to_mono(audio_data.T)

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
        raise Exception(f"Failed to process audio file: {str(e)}.  Check file format and ensure it's a supported audio type.")


def record_audio(max_duration=10, sample_rate=22050, status_element=None):
    """
    Records actual audio using a microphone.

    Parameters:
    -----------
    max_duration : int
        Maximum recording duration in seconds.
    sample_rate : int
        Sample rate for the recording.
    status_element : streamlit element
        Element to update with recording status.

    Returns:
    --------
    audio_data : bytes
        The recorded audio as a byte array in WAV format.
    """
    import pyaudio
    import wave

    # PyAudio configuration
    chunk = 1024  # Buffer size
    format = pyaudio.paInt16  # 16-bit PCM
    channels = 1  # Mono
    rate = sample_rate  # Sample rate

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open the stream
    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    st.info("Recording audio... Speak now!")
    if status_element:
        status_element.text("Recording in progress...")

    frames = []

    # Record audio for the specified duration
    for _ in range(0, int(rate / chunk * max_duration)):
        data = stream.read(chunk)
        frames.append(data)

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    st.success("Recording complete!")

    # Save the recorded audio to a WAV file in memory
    with io.BytesIO() as wav_io:
        with wave.open(wav_io, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(p.get_sample_size(format))
            wf.setframerate(rate)
            wf.writeframes(b''.join(frames))
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
