import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Since we don't have a pre-trained model, we'll implement a rule-based
# approach that simulates a machine learning model based on the features
# we've extracted. This approach uses domain knowledge about the differences
# between AI and human voices.

def predict_voice_type(features):
    """
    Predicts whether a voice is AI-generated or human based on extracted features.
    
    Parameters:
    -----------
    features : dict
        Dictionary of acoustic features
    
    Returns:
    --------
    prediction : str
        'human' or 'ai'
    confidence : float
        Prediction confidence (0-100)
    """
    # Enhanced rule-based classification system with improved weights and thresholds
    # These weights represent the importance of each feature in the decision
    feature_weights = {
        'pitch_stability': 0.35,         # Higher values favor AI prediction (increased importance)
        'spectral_centroid': 0.10,       # Extreme values favor AI (increased importance)
        'spectral_flatness': 0.15,       # Lower values favor AI (increased importance)
        'harmonic_ratio': 0.20,          # Higher values favor AI (increased importance)
        'tempo_variability': 0.25,       # Lower values favor AI (increased importance)
        'formant_clarity': 0.30,         # Higher values favor AI (increased importance)
        'shimmer': 0.15,                 # Lower values favor AI (increased importance)
        'spectral_contrast': 0.10,       # Now used in the model
        'zero_crossing_rate': 0.05       # Now used in the model with small weight
    }
    
    # Initialize score (higher = more likely to be AI)
    ai_score = 0.0
    feature_count = 0
    total_weight = 0.0
    
    # Pitch stability (higher in AI voices) - more pronounced in AI
    if 'pitch_stability' in features:
        pitch_stability = features['pitch_stability']
        # Apply a more aggressive curve to pitch stability
        stability_factor = pitch_stability**2  # Square it to make high values even more indicative of AI
        ai_score += stability_factor * feature_weights['pitch_stability']
        feature_count += 1
        total_weight += feature_weights['pitch_stability']
    
    # Spectral centroid (extreme values favor AI)
    if 'spectral_centroid' in features:
        spectral_centroid = features['spectral_centroid']
        # Refined model for spectral centroid
        human_centroid_range = (1800, 3000)  # Typical range for human speech
        
        # Calculate how far the centroid is from the human range
        if spectral_centroid < human_centroid_range[0]:
            distance = (human_centroid_range[0] - spectral_centroid) / human_centroid_range[0]
        elif spectral_centroid > human_centroid_range[1]:
            distance = (spectral_centroid - human_centroid_range[1]) / human_centroid_range[1]
        else:
            distance = 0  # Within human range
            
        ai_score += min(distance, 1.0) * feature_weights['spectral_centroid']
        feature_count += 1
        total_weight += feature_weights['spectral_centroid']
    
    # Spectral flatness (AI voices tend to have specific patterns)
    if 'spectral_flatness' in features:
        spectral_flatness = features['spectral_flatness']
        # AI voices tend to have either very high or very low flatness, rarely in between
        ai_flatness_score = abs(spectral_flatness - 0.35) / 0.35  # Distance from "typical" value
        ai_score += min(ai_flatness_score, 1.0) * feature_weights['spectral_flatness']
        feature_count += 1
        total_weight += feature_weights['spectral_flatness']
    
    # Harmonic ratio (higher in AI voices)
    if 'harmonic_ratio' in features:
        harmonic_ratio = features['harmonic_ratio']
        # AI voices typically have harmonic ratio above 0.6
        if harmonic_ratio > 0.7:
            harmonic_score = 1.0  # Strong indicator of AI
        elif harmonic_ratio > 0.6:
            harmonic_score = 0.7  # Moderate indicator of AI
        elif harmonic_ratio > 0.5:
            harmonic_score = 0.4  # Weak indicator
        else:
            harmonic_score = 0.0  # Human-like
            
        ai_score += harmonic_score * feature_weights['harmonic_ratio']
        feature_count += 1
        total_weight += feature_weights['harmonic_ratio']
    
    # Tempo variability (lower in AI voices)
    if 'tempo_variability' in features:
        tempo_var = features['tempo_variability']
        # AI voices have unnaturally consistent tempo
        if tempo_var < 0.1:
            tempo_score = 1.0  # Very consistent tempo (AI)
        elif tempo_var < 0.2:
            tempo_score = 0.8  # Somewhat consistent (likely AI)
        elif tempo_var < 0.3:
            tempo_score = 0.5  # Moderate variability (could be either)
        else:
            tempo_score = 0.0  # Highly variable (human)
            
        ai_score += tempo_score * feature_weights['tempo_variability']
        feature_count += 1
        total_weight += feature_weights['tempo_variability']
    
    # Formant clarity (higher in AI voices)
    if 'formant_clarity' in features:
        formant_clarity = features['formant_clarity']
        # AI voices often have unnaturally clear formants
        if formant_clarity > 0.8:
            clarity_score = 1.0  # Very clear formants (AI)
        elif formant_clarity > 0.6:
            clarity_score = 0.8  # Clear formants (likely AI)
        elif formant_clarity > 0.4:
            clarity_score = 0.4  # Moderate clarity (could be either)
        else:
            clarity_score = 0.0  # Natural formants (human)
            
        ai_score += clarity_score * feature_weights['formant_clarity']
        feature_count += 1
        total_weight += feature_weights['formant_clarity']
    
    # Shimmer (lower in AI voices)
    if 'shimmer' in features:
        shimmer = features['shimmer']
        # AI voices have less amplitude variation
        if shimmer < 0.1:
            shimmer_score = 1.0  # Very stable amplitude (AI)
        elif shimmer < 0.2:
            shimmer_score = 0.7  # Stable amplitude (likely AI)
        elif shimmer < 0.3:
            shimmer_score = 0.3  # Some variation (could be either)
        else:
            shimmer_score = 0.0  # Natural variation (human)
            
        ai_score += shimmer_score * feature_weights['shimmer']
        feature_count += 1
        total_weight += feature_weights['shimmer']
    
    # Spectral contrast (now used in the model)
    if 'spectral_contrast' in features and abs(features['spectral_contrast']) > 0.001:
        spectral_contrast = features['spectral_contrast']
        # AI voices may have different spectral contrast patterns
        contrast_normalized = min(abs(spectral_contrast) / 20.0, 1.0)  # Normalize to 0-1 range
        
        # Higher contrast can indicate AI voice (with current synthesis techniques)
        ai_score += contrast_normalized * feature_weights['spectral_contrast']
        feature_count += 1
        total_weight += feature_weights['spectral_contrast']
    
    # Zero crossing rate (now used with small weight)
    if 'zero_crossing_rate' in features:
        zcr = features['zero_crossing_rate']
        # Extreme ZCR values (very high or very low) may indicate AI
        zcr_normalized = abs(zcr - 0.1) / 0.2  # Distance from "typical" human value
        ai_score += min(zcr_normalized, 1.0) * feature_weights['zero_crossing_rate']
        feature_count += 1
        total_weight += feature_weights['zero_crossing_rate']
    
    # Normalize the score based on features that were actually used
    if total_weight > 0:
        normalized_ai_score = ai_score / total_weight
    else:
        normalized_ai_score = 0.5  # Default if no features were used
    
    # Apply a more sensitive sigmoid to get a better probability distribution
    # This makes the model more decisive with clearer distinctions
    ai_probability = 1 / (1 + np.exp(-8 * (normalized_ai_score - 0.45)))
    
    # Make prediction based on threshold
    prediction = 'ai' if ai_probability > 0.5 else 'human'
    
    # Calculate confidence percentage
    confidence = max(ai_probability, 1 - ai_probability) * 100
    
    return prediction, confidence

# This is a placeholder for future model training
def train_model(X, y):
    """
    Train a model to classify AI vs human voices.
    This would be used when we have a labeled dataset.
    
    Parameters:
    -----------
    X : array-like
        Feature matrix
    y : array-like
        Target labels ('human' or 'ai')
    
    Returns:
    --------
    model : trained classifier
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# This is a placeholder for saving the model
def save_model(model, filename='voice_classifier_model.pkl'):
    """
    Save trained model to disk.
    
    Parameters:
    -----------
    model : trained classifier
    filename : str
        Path to save the model
    """
    joblib.dump(model, filename)

# This is a placeholder for loading the model
def load_model(filename='voice_classifier_model.pkl'):
    """
    Load trained model from disk.
    
    Parameters:
    -----------
    filename : str
        Path to the saved model
    
    Returns:
    --------
    model : trained classifier
    """
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        return None
