import streamlit as st
import numpy as np
import tempfile
import os
import io
import matplotlib.pyplot as plt
from datetime import datetime
import time

from utils.audio_processor import process_audio_file, record_audio, bytes_to_numpy
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
if 'audio_source' not in st.session_state:
    st.session_state.audio_source = None
if 'filename' not in st.session_state:
    st.session_state.filename = None
if 'current_page' not in st.session_state:
    st.session_state.current_page = "main"

# File uploader column
with col1:
    st.subheader("Option 1: Upload Audio File")
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg', 'm4a'])
    
    if uploaded_file is not None:
        # Display file information
        st.info(f"Selected file: **{uploaded_file.name}**")
        
        # Add a player for the uploaded audio
        st.audio(uploaded_file, format=f"audio/{uploaded_file.name.split('.')[-1]}")
        
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
                st.session_state.audio_source = "upload"
                st.session_state.filename = uploaded_file.name
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
                # Generate a unique timestamp for the recording
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.session_state.filename = f"recording_{timestamp}.wav"
                st.rerun()
        else:
            if st.button("Stop Recording"):
                st.session_state.is_recording = False
                st.rerun()
    
    with record_col2:
        if st.session_state.recorded_audio is not None:
            if st.button("Use Recorded Audio"):
                st.session_state.audio_data = st.session_state.recorded_audio
                st.session_state.sample_rate = 22050  # Standard sample rate used in the recording function
                st.session_state.audio_source = "recording"
                st.success("Using the recorded audio!")
    
    # Recording status and control
    if st.session_state.is_recording:
        # Create a container for the recording process
        record_container = st.container()
        
        with record_container:
            st.warning("Recording in progress... (Speak now)")
            
            # Create a placeholder for the recording duration display
            recording_status = st.empty()
            
            # Add a stop button inside the recording process
            if st.button("Stop Recording Now"):
                st.session_state.is_recording = False
                st.rerun()
            
            # Record audio for a maximum of 10 seconds
            max_duration = 10  # seconds
            
            # Generate sample audio since we're in a cloud environment
            audio_data = record_audio(max_duration=max_duration, 
                                     status_element=recording_status)
            
            st.session_state.recorded_audio = audio_data
            st.session_state.is_recording = False
            st.success("Sample audio generated successfully!")
            st.audio(audio_data, format="audio/wav", sample_rate=22050)
    
    # Display recorded audio information if available
    if st.session_state.recorded_audio is not None and not st.session_state.is_recording:
        st.info(f"Recorded file: **{st.session_state.filename}**")
        st.audio(st.session_state.recorded_audio, format="audio/wav", sample_rate=22050)

st.markdown("---")

# Page navigation
if st.session_state.audio_data is not None:
    if st.session_state.current_page == "main":
        # Analysis section (Main Page)
        st.subheader("Voice Analysis")
        
        analyze_button = st.button("Analyze Voice")
        
        if analyze_button:
            with st.spinner("Analyzing voice characteristics..."):
                # Check if audio_data is bytes (from recording) or numpy array (from file upload)
                audio_data = st.session_state.audio_data
                sample_rate = st.session_state.sample_rate
                
                # If it's bytes, convert to numpy array
                if isinstance(audio_data, bytes):
                    # Ensure we have a valid sample rate (default to 22050 if None)
                    sr = sample_rate if sample_rate is not None else 22050
                    audio_data = bytes_to_numpy(audio_data, sr)
                
                # Make sure we have a valid sample rate for feature extraction
                sr = sample_rate if sample_rate is not None else 22050
                    
                # Extract features
                st.session_state.features = extract_features(audio_data, sr)
                
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
                
                # Audio source info
                if st.session_state.filename:
                    st.markdown(f"**Analyzing:** {st.session_state.filename}")
                
                # Create three columns for visualizations
                viz_col1, viz_col2, viz_col3 = st.columns(3)
                
                with viz_col1:
                    st.subheader("Audio Spectrogram")
                    # Make sure we're using the converted audio_data for the spectrogram if it was bytes
                    display_audio = audio_data if isinstance(st.session_state.audio_data, bytes) else st.session_state.audio_data
                    
                    # Ensure we have a valid sample rate for visualization
                    display_sr = st.session_state.sample_rate if st.session_state.sample_rate is not None else 22050
                    fig = plot_spectrogram(display_audio, display_sr)
                    st.pyplot(fig)
                
                with viz_col2:
                    st.subheader("Key Features")
                    fig = plot_features(st.session_state.features)
                    st.pyplot(fig)
                
                with viz_col3:
                    st.subheader("Confidence Score")
                    fig = plot_confidence(st.session_state.confidence, st.session_state.prediction)
                    st.pyplot(fig)
                
                # Summary of feature analysis
                st.subheader("Feature Analysis Summary")
                
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
                
                # Button to view detailed feature analysis
                if st.button("View Detailed Feature Analysis"):
                    st.session_state.current_page = "detailed_analysis"
                    st.rerun()
        
    elif st.session_state.current_page == "detailed_analysis":
        # Detailed Feature Analysis Page
        st.title("Detailed Feature Analysis")
        
        # Back button
        if st.button("← Back to Main Page"):
            st.session_state.current_page = "main"
            st.rerun()
        
        # Show audio file information
        if st.session_state.filename:
            st.markdown(f"## Analysis of: {st.session_state.filename}")
            
            # Add audio player
            if st.session_state.audio_source == "upload":
                # We don't have direct access to the original upload, so describe how to re-listen
                st.info("To listen to this audio again, please go back to the main page.")
            elif st.session_state.audio_source == "recording" and st.session_state.recorded_audio is not None:
                st.audio(st.session_state.recorded_audio, format="audio/wav", sample_rate=22050)
        
        # Display result and confidence
        if st.session_state.prediction == "human":
            result_color = "green"
            result_text = "Human Voice Detected"
            result_emoji = "👨‍🦰"
        else:
            result_color = "orange"
            result_text = "AI-Generated Voice Detected"
            result_emoji = "🤖"
        
        st.markdown(f"""
        <div style="background-color: {result_color}; padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
            <h2 style="color: white;">{result_text} {result_emoji}</h2>
            <h3 style="color: white;">Confidence: {st.session_state.confidence:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Audio spectrogram
        st.subheader("Audio Spectrogram Analysis")
        # We need to access the audio data again
        audio_data = st.session_state.audio_data
        sample_rate = st.session_state.sample_rate
        
        # If it's bytes, convert to numpy array
        if isinstance(audio_data, bytes):
            # Ensure we have a valid sample rate (default to 22050 if None)
            sr = sample_rate if sample_rate is not None else 22050
            audio_data = bytes_to_numpy(audio_data, sr)
        
        # Ensure we have a valid sample rate for visualization
        display_sr = st.session_state.sample_rate if st.session_state.sample_rate is not None else 22050
        fig = plot_spectrogram(audio_data, display_sr)
        st.pyplot(fig)
        
        st.markdown("""
        The spectrogram shows the frequency content of the audio over time:
        - **Horizontal axis**: Time (seconds)
        - **Vertical axis**: Frequency (Hz)
        - **Color intensity**: Energy at each time-frequency point
        
        AI-generated voices often have more uniform, regular patterns in their spectrograms, 
        while human voices typically show more natural variations and irregularities.
        """)
        
        # Feature details
        st.subheader("Detailed Feature Breakdown")
        st.markdown("This section provides a detailed analysis of each acoustic feature extracted from the voice sample.")
        
        # Pitch Stability
        st.markdown("### 1. Pitch Stability")
        pitch_stability = st.session_state.features['pitch_stability']
        st.markdown(f"**Value:** {pitch_stability:.4f}")
        
        # Create a gauge-like visualization for pitch stability
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.5)
        ax.axvspan(0, 0.5, alpha=0.3, color='green')
        ax.axvspan(0.5, 0.7, alpha=0.3, color='yellow')
        ax.axvspan(0.7, 1, alpha=0.3, color='red')
        ax.plot([pitch_stability, pitch_stability], [0, 0.5], 'k-', linewidth=3)
        ax.scatter(pitch_stability, 0.25, color='black', s=200, zorder=5)
        ax.text(0.25, 0.4, "Human-like", ha='center')
        ax.text(0.6, 0.4, "Uncertain", ha='center')
        ax.text(0.85, 0.4, "AI-like", ha='center')
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1.0'])
        ax.set_yticks([])
        st.pyplot(fig)
        
        st.markdown("""
        **Interpretation:**
        - Pitch stability measures how consistent the pitch (fundamental frequency) is throughout the voice sample.
        - Higher values (>0.7) indicate unnaturally stable pitch, typical of AI-generated voices.
        - Lower values (<0.5) suggest natural pitch variations found in human speech.
        - Mid-range values (0.5-0.7) could be either human or AI.
        
        **Why it matters:** Humans naturally vary their pitch while speaking, while AI systems often produce more mechanically consistent pitch patterns.
        """)
        
        # Harmonic Ratio
        st.markdown("### 2. Harmonic Ratio")
        harmonic_ratio = st.session_state.features['harmonic_ratio']
        st.markdown(f"**Value:** {harmonic_ratio:.4f}")
        
        # Create a gauge visualization
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.5)
        ax.axvspan(0, 0.4, alpha=0.3, color='green')
        ax.axvspan(0.4, 0.6, alpha=0.3, color='yellow')
        ax.axvspan(0.6, 1, alpha=0.3, color='red')
        ax.plot([harmonic_ratio, harmonic_ratio], [0, 0.5], 'k-', linewidth=3)
        ax.scatter(harmonic_ratio, 0.25, color='black', s=200, zorder=5)
        ax.text(0.2, 0.4, "Human-like", ha='center')
        ax.text(0.5, 0.4, "Uncertain", ha='center')
        ax.text(0.8, 0.4, "AI-like", ha='center')
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.set_yticks([])
        st.pyplot(fig)
        
        st.markdown("""
        **Interpretation:**
        - Harmonic ratio indicates the proportion of harmonic content to noise in the voice.
        - Higher values (>0.6) suggest cleaner, more harmonic sound (often AI-generated).
        - Lower values (<0.4) indicate more natural noise components (typical of human voice).
        - AI voices often have unnaturally clean harmonics with less noise between them.
        
        **Why it matters:** Human voices naturally contain some noise and irregularities. AI-generated voices often have too-perfect harmonic structures.
        """)
        
        # Formant Clarity
        st.markdown("### 3. Formant Clarity")
        formant_clarity = st.session_state.features['formant_clarity']
        st.markdown(f"**Value:** {formant_clarity:.4f}")
        
        # Create a gauge visualization
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 0.5)
        ax.axvspan(0, 0.3, alpha=0.3, color='green')
        ax.axvspan(0.3, 0.5, alpha=0.3, color='yellow')
        ax.axvspan(0.5, 0.7, alpha=0.3, color='orange')
        ax.axvspan(0.7, 1, alpha=0.3, color='red')
        ax.plot([formant_clarity, formant_clarity], [0, 0.5], 'k-', linewidth=3)
        ax.scatter(formant_clarity, 0.25, color='black', s=200, zorder=5)
        ax.text(0.15, 0.4, "Human-like", ha='center')
        ax.text(0.4, 0.4, "Possibly Human", ha='center')
        ax.text(0.6, 0.4, "Possibly AI", ha='center')
        ax.text(0.85, 0.4, "AI-like", ha='center')
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.set_yticks([])
        st.pyplot(fig)
        
        st.markdown("""
        **Interpretation:**
        - Formant clarity measures how distinctly the vocal tract resonances (formants) appear.
        - Higher values (>0.7) indicate unnaturally clear formants, often seen in AI voices.
        - Lower values (<0.3) suggest more natural formant patterns typical of human speech.
        - Mid-range values require consideration of other features for classification.
        
        **Why it matters:** Formants are resonances of the vocal tract that give a voice its characteristic sound. Human formants have natural variation and slight 'fuzziness', while AI-generated formants may be too well-defined.
        """)
        
        # Additional Features
        st.markdown("### Additional Features")
        
        # Create a table for other features
        data = {
            "Feature": ["Spectral Centroid", "Spectral Flatness", "Tempo Variability", "Zero Crossing Rate"],
            "Value": [
                f"{st.session_state.features['spectral_centroid']:.2f} Hz",
                f"{st.session_state.features['spectral_flatness']:.4f}",
                f"{st.session_state.features['tempo_variability']:.4f}",
                f"{st.session_state.features['zero_crossing_rate']:.4f}"
            ],
            "Interpretation": [
                "Higher values indicate brighter sound, midrange (2000-3000 Hz) is typical for natural speech",
                "Higher values (>0.4) suggest more noise-like spectrum (often human)",
                "Higher values (>0.3) indicate natural rhythm variations (human-like)",
                "Extreme values may indicate artificial processing"
            ]
        }
        
        # Convert to markdown table
        table = "| Feature | Value | Interpretation |\n| ------ | ------ | ------ |\n"
        for i in range(len(data["Feature"])):
            table += f"| {data['Feature'][i]} | {data['Value'][i]} | {data['Interpretation'][i]} |\n"
        
        st.markdown(table)
        
        # Overall analysis conclusion
        st.subheader("Conclusion")
        
        conclusion = f"""
        Based on comprehensive acoustic analysis of the voice sample, this voice is most likely **{st.session_state.prediction.upper()}** 
        with a confidence of **{st.session_state.confidence:.2f}%**.
        
        The key indicators supporting this conclusion are:
        
        - **Pitch stability**: {'Unnaturally stable (AI-like)' if pitch_stability > 0.7 else 'Contains natural variations (human-like)'}
        - **Harmonic content**: {'Unusually clean harmonics (AI-like)' if harmonic_ratio > 0.6 else 'Natural harmonic-to-noise ratio (human-like)'}
        - **Formant structure**: {'Unnaturally clear formants (AI-like)' if formant_clarity > 0.7 else 'Natural formant patterns (human-like)'}
        - **Temporal variations**: {'Mechanical rhythm (AI-like)' if st.session_state.features['tempo_variability'] < 0.1 else 'Natural rhythm variations (human-like)'}
        """
        
        st.markdown(conclusion)

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
