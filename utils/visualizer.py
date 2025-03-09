import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import matplotlib.ticker as ticker

def plot_spectrogram(audio_data, sample_rate):
    """
    Create a spectrogram visualization of the audio data.
    
    Parameters:
    -----------
    audio_data : numpy array
        The audio time series
    sample_rate : int
        The sample rate of the audio
        
    Returns:
    --------
    fig : matplotlib figure
        The spectrogram visualization
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Compute spectrogram
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio_data)), 
        ref=np.max
    )
    
    # Plot spectrogram
    img = librosa.display.specshow(
        D, 
        x_axis='time', 
        y_axis='log', 
        sr=sample_rate,
        ax=ax
    )
    
    # Add colorbar
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    
    # Add title and labels
    ax.set_title('Audio Spectrogram', fontsize=14)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    
    # Format frequency axis with Hz labels
    formatter = ticker.FuncFormatter(lambda x, pos: f'{int(x)} Hz')
    ax.yaxis.set_major_formatter(formatter)
    
    # Ensure tight layout
    plt.tight_layout()
    
    return fig

def plot_features(features):
    """
    Create a bar plot of key features.
    
    Parameters:
    -----------
    features : dict
        Dictionary of extracted features
        
    Returns:
    --------
    fig : matplotlib figure
        The features visualization
    """
    # Select key features to display
    key_features = [
        'pitch_stability',
        'harmonic_ratio', 
        'tempo_variability',
        'formant_clarity',
        'spectral_flatness',
        'shimmer'
    ]
    
    # Get feature values, ensuring all keys exist
    feature_values = [features.get(f, 0) for f in key_features]
    
    # Create better feature names for display
    display_names = [
        'Pitch Stability',
        'Harmonic Ratio',
        'Tempo Variability',
        'Formant Clarity',
        'Spectral Flatness',
        'Shimmer'
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Create horizontal bar chart
    bars = ax.barh(display_names, feature_values, color='skyblue')
    
    # Add value labels
    for i, v in enumerate(feature_values):
        ax.text(v + 0.01, i, f'{v:.2f}', va='center')
    
    # Set up chart
    ax.set_title('Voice Characteristics', fontsize=14)
    ax.set_xlabel('Normalized Value', fontsize=12)
    ax.set_xlim(0, 1.1)
    
    # Add vertical line for reference
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
    
    # Add AI/Human indicators
    ax.text(0.05, -0.8, 'Human-like', fontsize=10, ha='left')
    ax.text(0.95, -0.8, 'AI-like', fontsize=10, ha='right')
    
    # Ensure tight layout
    plt.tight_layout()
    
    return fig

def plot_confidence(confidence, prediction):
    """
    Create a gauge-like visualization of the confidence score.
    
    Parameters:
    -----------
    confidence : float
        Confidence score (0-100)
    prediction : str
        'human' or 'ai'
        
    Returns:
    --------
    fig : matplotlib figure
        The confidence visualization
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Define gauge parameters
    theta = np.linspace(0, np.pi, 100)
    r = np.ones_like(theta)
    
    # Define gauge color based on prediction
    if prediction == 'human':
        gauge_color = 'green'
        text = 'Human'
    else:
        gauge_color = 'orange'
        text = 'AI'
    
    # Plot full gauge in light gray
    ax.plot(theta, r, color='lightgray', linewidth=15, alpha=0.3)
    
    # Normalize confidence to 0-1 range
    norm_confidence = confidence / 100
    
    # Plot filled gauge based on confidence
    gauge_theta = np.linspace(0, np.pi * norm_confidence, 50)
    gauge_r = np.ones_like(gauge_theta)
    ax.plot(gauge_theta, gauge_r, color=gauge_color, linewidth=15, alpha=0.7)
    
    # Add confidence text
    ax.text(np.pi/2, 0.5, f"{text}\n{confidence:.1f}%", 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Set limits and remove axes
    ax.set_ylim(0, 1.3)
    ax.set_frame_on(False)
    ax.axes.get_yaxis().set_visible(False)
    
    # Set custom ticks
    tick_angles = np.linspace(0, np.pi, 6)
    tick_labels = ['0%', '20%', '40%', '60%', '80%', '100%']
    ax.set_xticks(tick_angles)
    ax.set_xticklabels(tick_labels)
    
    # Add human/AI labels to gauge
    ax.text(0, 1.1, 'Human', fontsize=12, ha='left', va='center')
    ax.text(np.pi, 1.1, 'AI', fontsize=12, ha='right', va='center')
    
    # Set title
    ax.set_title('Classification Confidence', pad=20, fontsize=14)
    
    return fig
