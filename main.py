import streamlit as st
import numpy as np
import tempfile
import os
import io
import matplotlib.pyplot as plt
from datetime import datetime
import time

from utils.audio_processor import process_audio_file, record_audio
from model.feature_extractor import extract_features
from model.classifier import predict_voice_type
from utils.visualizer import plot_spectrogram, plot_features, plot_confidence

# Page configuration
st.set_page_config(
    page_title="AI vs Human Voice Detector",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App header
st.title("AI vs Human Voice Detector")
st.markdown("""
    <div style="text-align: center;">
        <p>Differentiate between AI-generated and human voices using machine learning</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Create two columns for input methods
col1, col2 = st.columns(2)

# Initialize session state variables if they don't exist
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'sample_rate' not in st.session_state:
    st.session_state.sample_rate = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False
if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None

# File uploader column
with col1:
    st.subheader("Option 1: Upload Audio File")
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg', 'm4a'])
    
    if uploaded_file is not None:
        # Process the uploaded file
        with st.spinner("Processing audio file..."):
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.'+uploaded_file.name.split('.')[-1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_filename = tmp_file.name
            
            try:
                # Process the audio file
                audio_data, sample_rate = process_audio_file(temp_filename)
                st.session_state.audio_data = audio_data
                st.session_state.sample_rate = sample_rate
                st.success("Audio file processed successfully!")
                
                # Clean up the temporary file
                os.unlink(temp_filename)
            except Exception as e:
                st.error(f"Error processing audio file: {str(e)}")
                # Clean up the temporary file
                os.unlink(temp_filename)

# Audio recording column
with col2:
    st.subheader("Option 2: Record Your Voice")
    
    # Recording controls
    record_col1, record_col2 = st.columns(2)
    
    with record_col1:
        if not st.session_state.is_recording:
            if st.button("Start Recording"):
                st.session_state.is_recording = True
                st.session_state.recorded_audio = None
                st.session_state.audio_data = None
                st.experimental_rerun()
        else:
            if st.button("Stop Recording"):
                st.session_state.is_recording = False
                st.experimental_rerun()
    
    with record_col2:
        if st.session_state.recorded_audio is not None:
            if st.button("Use Recorded Audio"):
                st.session_state.audio_data = st.session_state.recorded_audio
                st.session_state.sample_rate = 22050  # Standard sample rate used in the recording function
                st.success("Using the recorded audio!")
    
    # Recording status and control
    if st.session_state.is_recording:
        st.warning("Recording in progress... (Speak now)")
        
        # Create a placeholder for the recording duration display
        recording_status = st.empty()
        
        # Record audio for a maximum of 10 seconds
        max_duration = 10  # seconds
        start_time = time.time()
        
        audio_data = record_audio(max_duration=max_duration, 
                                   status_element=recording_status)
        
        st.session_state.recorded_audio = audio_data
        st.session_state.is_recording = False
        st.success("Recording completed!")
        st.audio(audio_data, format="audio/wav", sample_rate=22050)

st.markdown("---")

# Analysis section
if st.session_state.audio_data is not None:
    st.subheader("Voice Analysis")
    
    analyze_button = st.button("Analyze Voice")
    
    if analyze_button:
        with st.spinner("Analyzing voice characteristics..."):
            # Extract features
            st.session_state.features = extract_features(st.session_state.audio_data, st.session_state.sample_rate)
            
            # Make prediction
            st.session_state.prediction, st.session_state.confidence = predict_voice_type(st.session_state.features)
            
            # Display results header based on prediction
            if st.session_state.prediction == "human":
                result_color = "green"
                result_text = "Human Voice Detected"
                result_emoji = "👨‍🦰"
            else:
                result_color = "orange"
                result_text = "AI-Generated Voice Detected"
                result_emoji = "🤖"
            
            st.markdown(f"""
            <div style="background-color: {result_color}; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: white;">{result_text} {result_emoji}</h2>
                <h3 style="color: white;">Confidence: {st.session_state.confidence:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Create three columns for visualizations
            viz_col1, viz_col2, viz_col3 = st.columns(3)
            
            with viz_col1:
                st.subheader("Audio Spectrogram")
                fig = plot_spectrogram(st.session_state.audio_data, st.session_state.sample_rate)
                st.pyplot(fig)
            
            with viz_col2:
                st.subheader("Key Features")
                fig = plot_features(st.session_state.features)
                st.pyplot(fig)
            
            with viz_col3:
                st.subheader("Confidence Score")
                fig = plot_confidence(st.session_state.confidence, st.session_state.prediction)
                st.pyplot(fig)
            
            # Feature explanation
            st.subheader("Feature Analysis")
            
            # Create two columns for feature explanations
            feat_col1, feat_col2 = st.columns(2)
            
            with feat_col1:
                st.markdown("**Pitch Stability:**")
                pitch_stability = st.session_state.features['pitch_stability']
                if pitch_stability > 0.7:
                    st.markdown("🔹 Very stable pitch (typical of AI voices)")
                elif pitch_stability > 0.5:
                    st.markdown("🔹 Moderately stable pitch")
                else:
                    st.markdown("🔹 Natural pitch variations (typical of human voices)")
                
                st.markdown("**Spectral Centroid:**")
                spectral_centroid = st.session_state.features['spectral_centroid']
                if spectral_centroid > 3000:
                    st.markdown("🔹 Higher frequency concentration (brighter sound)")
                elif spectral_centroid > 2000:
                    st.markdown("🔹 Moderate frequency concentration")
                else:
                    st.markdown("🔹 Lower frequency concentration (warmer sound)")
                
                st.markdown("**Harmonic Ratio:**")
                harmonic_ratio = st.session_state.features['harmonic_ratio']
                if harmonic_ratio > 0.6:
                    st.markdown("🔹 High harmonic content (cleaner sound, often AI)")
                else:
                    st.markdown("🔹 Lower harmonic content (more noise, often human)")
            
            with feat_col2:
                st.markdown("**Tempo Variability:**")
                tempo_var = st.session_state.features['tempo_variability']
                if tempo_var < 0.1:
                    st.markdown("🔹 Very consistent tempo (typical of AI)")
                elif tempo_var < 0.3:
                    st.markdown("🔹 Moderately variable tempo")
                else:
                    st.markdown("🔹 Natural tempo fluctuations (human-like)")
                
                st.markdown("**Spectral Flatness:**")
                flatness = st.session_state.features['spectral_flatness']
                if flatness > 0.4:
                    st.markdown("🔹 More noise-like spectrum (often human)")
                else:
                    st.markdown("🔹 More tonal spectrum (often AI)")
                
                st.markdown("**Formant Clarity:**")
                formant_clarity = st.session_state.features['formant_clarity']
                if formant_clarity > 0.7:
                    st.markdown("🔹 Very clear formants (often AI-generated)")
                else:
                    st.markdown("🔹 Natural formant patterns (typical of human speech)")

# Sidebar with information
with st.sidebar:
    st.image("assets/app_logo.svg", width=80)
    st.title("About this App")
    st.info("""
    This app analyzes voice recordings to determine if they are:
    - 👨‍🦰 Human voice
    - 🤖 AI-generated voice
    
    **How it works:**
    1. Upload or record an audio file
    2. The app extracts acoustic features
    3. A machine learning model classifies the voice
    4. Results and visualizations are displayed
    """)
    
    st.markdown("---")
    
    st.subheader("Why This Matters")
    st.markdown("""
    With the rise of AI voice technology, detecting synthetic speech has become increasingly important for:
    
    - Media verification
    - Fraud prevention
    - Digital security
    - Content moderation
    """)
    
    st.markdown("---")
    
    st.markdown("### Usage Tips")
    st.markdown("""
    - For best results, use clear recordings without background noise
    - Recordings should be at least 3 seconds long
    - The app works with most common voice types and accents
    - Higher confidence scores (>80%) indicate stronger predictions
    """)
