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
    # Improved rule-based classification system with recalibrated weights and thresholds
    # These weights represent the importance of each feature in the decision - adjusted based on user feedback
    feature_weights = {
        'pitch_stability': 0.40,         # Higher values favor AI prediction
        'spectral_centroid': 0.15,       # Extreme values favor AI
        'spectral_flatness': 0.20,       # Lower values favor AI
        'harmonic_ratio': 0.30,          # Higher values favor AI
        'tempo_variability': 0.35,       # Lower values favor AI
        'formant_clarity': 0.40,         # Higher values favor AI
        'shimmer': 0.25,                 # Lower values favor AI
        'spectral_contrast': 0.15,       # Used in the model
        'zero_crossing_rate': 0.10       # Used with increased weight
    }
    
    # Initialize variables
    ai_score = 0.0
    human_score = 0.0  # Added to measure human characteristics more explicitly
    feature_count = 0
    total_weight = 0.0
    
    # Pitch stability (higher in AI voices)
    if 'pitch_stability' in features:
        pitch_stability = features['pitch_stability']
        
        # More aggressive curve to identify AI voices (recalibrated)
        if pitch_stability > 0.75:  # Very high stability
            ai_contribution = 1.0
            human_contribution = 0.0
        elif pitch_stability > 0.6:  # High stability
            ai_contribution = 0.8
            human_contribution = 0.2
        elif pitch_stability > 0.45:  # Moderate stability
            ai_contribution = 0.5
            human_contribution = 0.5
        elif pitch_stability > 0.3:  # Natural stability
            ai_contribution = 0.2
            human_contribution = 0.8
        else:  # Very natural (low stability)
            ai_contribution = 0.0
            human_contribution = 1.0
            
        ai_score += ai_contribution * feature_weights['pitch_stability']
        human_score += human_contribution * feature_weights['pitch_stability']
        feature_count += 1
        total_weight += feature_weights['pitch_stability']
    
    # Spectral centroid (extreme values favor AI)
    if 'spectral_centroid' in features:
        spectral_centroid = features['spectral_centroid']
        # Refined model for spectral centroid - adjusted for better accuracy
        human_centroid_range = (1800, 3200)  # Expanded range for human speech
        
        # Calculate how far the centroid is from the human range
        if spectral_centroid < human_centroid_range[0]:
            distance = (human_centroid_range[0] - spectral_centroid) / human_centroid_range[0]
            ai_contribution = min(distance * 1.5, 1.0)  # Amplified effect
            human_contribution = 1.0 - ai_contribution
        elif spectral_centroid > human_centroid_range[1]:
            distance = (spectral_centroid - human_centroid_range[1]) / human_centroid_range[1]
            ai_contribution = min(distance * 1.5, 1.0)  # Amplified effect
            human_contribution = 1.0 - ai_contribution
        else:
            # Within human range - strongly favor human prediction
            ai_contribution = 0.1
            human_contribution = 0.9
            
        ai_score += ai_contribution * feature_weights['spectral_centroid']
        human_score += human_contribution * feature_weights['spectral_centroid']
        feature_count += 1
        total_weight += feature_weights['spectral_centroid']
    
    # Spectral flatness (AI voices tend to have specific patterns)
    if 'spectral_flatness' in features:
        spectral_flatness = features['spectral_flatness']
        
        # Recalibrated flatness assessment
        if spectral_flatness > 0.5:  # Very noise-like (often human)
            ai_contribution = 0.1
            human_contribution = 0.9
        elif spectral_flatness > 0.35:  # Moderate noise (human-like)
            ai_contribution = 0.3
            human_contribution = 0.7
        elif spectral_flatness > 0.25:  # Middle ground
            ai_contribution = 0.5
            human_contribution = 0.5
        elif spectral_flatness > 0.15:  # More tonal (AI-leaning)
            ai_contribution = 0.7
            human_contribution = 0.3
        else:  # Very tonal (strongly AI-like)
            ai_contribution = 0.9
            human_contribution = 0.1
        
        ai_score += ai_contribution * feature_weights['spectral_flatness']
        human_score += human_contribution * feature_weights['spectral_flatness']
        feature_count += 1
        total_weight += feature_weights['spectral_flatness']
    
    # Harmonic ratio (higher in AI voices)
    if 'harmonic_ratio' in features:
        harmonic_ratio = features['harmonic_ratio']
        
        # Recalibrated thresholds for better distinction
        if harmonic_ratio > 0.7:  # Extremely harmonic (AI)
            ai_contribution = 1.0
            human_contribution = 0.0
        elif harmonic_ratio > 0.6:  # Very harmonic (likely AI)
            ai_contribution = 0.85
            human_contribution = 0.15
        elif harmonic_ratio > 0.5:  # Moderately harmonic (possibly AI)
            ai_contribution = 0.7
            human_contribution = 0.3
        elif harmonic_ratio > 0.4:  # Natural harmonics (could be either)
            ai_contribution = 0.4
            human_contribution = 0.6
        elif harmonic_ratio > 0.3:  # Natural with noise (likely human)
            ai_contribution = 0.2
            human_contribution = 0.8
        else:  # Very natural (definitely human)
            ai_contribution = 0.05
            human_contribution = 0.95
            
        # Increased weight for this crucial feature
        effective_weight = feature_weights['harmonic_ratio'] * 1.3
        ai_score += ai_contribution * effective_weight
        human_score += human_contribution * effective_weight
        feature_count += 1
        total_weight += effective_weight
    
    # Tempo variability (lower in AI voices)
    if 'tempo_variability' in features:
        tempo_var = features['tempo_variability']
        
        # Recalibrated tempo variability assessment
        if tempo_var < 0.08:  # Extremely consistent (AI)
            ai_contribution = 1.0
            human_contribution = 0.0
        elif tempo_var < 0.15:  # Very consistent (likely AI)
            ai_contribution = 0.85
            human_contribution = 0.15
        elif tempo_var < 0.25:  # Somewhat consistent (possibly AI)
            ai_contribution = 0.6
            human_contribution = 0.4
        elif tempo_var < 0.35:  # Moderate variability (could be either)
            ai_contribution = 0.4
            human_contribution = 0.6
        elif tempo_var < 0.5:  # Natural variability (likely human)
            ai_contribution = 0.15
            human_contribution = 0.85
        else:  # High variability (definitely human)
            ai_contribution = 0.0
            human_contribution = 1.0
            
        ai_score += ai_contribution * feature_weights['tempo_variability']
        human_score += human_contribution * feature_weights['tempo_variability']
        feature_count += 1
        total_weight += feature_weights['tempo_variability']
    
    # Formant clarity (higher in AI voices)
    if 'formant_clarity' in features:
        formant_clarity = features['formant_clarity']
        
        # Recalibrated formant clarity assessment - critical feature
        if formant_clarity > 0.8:  # Extremely clear formants (definitely AI)
            ai_contribution = 1.0
            human_contribution = 0.0
        elif formant_clarity > 0.65:  # Very clear formants (likely AI)
            ai_contribution = 0.9
            human_contribution = 0.1
        elif formant_clarity > 0.5:  # Clear formants (possibly AI)
            ai_contribution = 0.7
            human_contribution = 0.3
        elif formant_clarity > 0.35:  # Moderate clarity (could be either)
            ai_contribution = 0.4
            human_contribution = 0.6
        elif formant_clarity > 0.2:  # Natural formants (likely human)
            ai_contribution = 0.15
            human_contribution = 0.85
        else:  # Very natural formants (definitely human)
            ai_contribution = 0.0
            human_contribution = 1.0
            
        # Significantly increased weight for this critical feature
        effective_weight = feature_weights['formant_clarity'] * 1.75
        ai_score += ai_contribution * effective_weight
        human_score += human_contribution * effective_weight
        feature_count += 1
        total_weight += effective_weight
    
    # Shimmer (lower in AI voices)
    if 'shimmer' in features:
        shimmer = features['shimmer']
        
        # Recalibrated shimmer assessment
        if shimmer < 0.08:  # Extremely stable amplitude (AI)
            ai_contribution = 1.0
            human_contribution = 0.0
        elif shimmer < 0.15:  # Very stable amplitude (likely AI)
            ai_contribution = 0.8
            human_contribution = 0.2
        elif shimmer < 0.25:  # Moderately stable (possibly AI)
            ai_contribution = 0.6
            human_contribution = 0.4
        elif shimmer < 0.35:  # Some variation (could be either)
            ai_contribution = 0.4
            human_contribution = 0.6
        elif shimmer < 0.5:  # Natural variation (likely human)
            ai_contribution = 0.2
            human_contribution = 0.8
        else:  # High variation (definitely human)
            ai_contribution = 0.0
            human_contribution = 1.0
            
        ai_score += ai_contribution * feature_weights['shimmer']
        human_score += human_contribution * feature_weights['shimmer']
        feature_count += 1
        total_weight += feature_weights['shimmer']
    
    # Spectral contrast
    if 'spectral_contrast' in features and abs(features['spectral_contrast']) > 0.001:
        spectral_contrast = features['spectral_contrast']
        
        # Normalize to a more meaningful range and recalibrate
        contrast_normalized = min(abs(spectral_contrast) / 25.0, 1.0)
        
        # Very high or very low contrast may indicate AI
        if contrast_normalized > 0.8:  # Extreme contrast (likely AI)
            ai_contribution = 0.9
            human_contribution = 0.1
        elif contrast_normalized > 0.6:  # High contrast (possibly AI)
            ai_contribution = 0.7
            human_contribution = 0.3
        elif contrast_normalized > 0.4:  # Moderate contrast (unclear)
            ai_contribution = 0.5
            human_contribution = 0.5
        elif contrast_normalized > 0.2:  # Low contrast (possibly human)
            ai_contribution = 0.3
            human_contribution = 0.7
        else:  # Very low contrast (likely human)
            ai_contribution = 0.1
            human_contribution = 0.9
            
        ai_score += ai_contribution * feature_weights['spectral_contrast']
        human_score += human_contribution * feature_weights['spectral_contrast']
        feature_count += 1
        total_weight += feature_weights['spectral_contrast']
    
    # Zero crossing rate
    if 'zero_crossing_rate' in features:
        zcr = features['zero_crossing_rate']
        
        # Recalibrated ZCR assessment
        zcr_normalized = abs(zcr - 0.12) / 0.12  # Distance from typical human value
        
        if zcr_normalized > 0.8:  # Very far from typical (likely AI)
            ai_contribution = 0.9
            human_contribution = 0.1
        elif zcr_normalized > 0.5:  # Moderately far (possibly AI)
            ai_contribution = 0.7
            human_contribution = 0.3
        elif zcr_normalized > 0.3:  # Somewhat atypical (unclear)
            ai_contribution = 0.5
            human_contribution = 0.5
        else:  # Within typical range (likely human)
            ai_contribution = 0.2
            human_contribution = 0.8
            
        ai_score += ai_contribution * feature_weights['zero_crossing_rate']
        human_score += human_contribution * feature_weights['zero_crossing_rate']
        feature_count += 1
        total_weight += feature_weights['zero_crossing_rate']
    
    # Calculate final scores based on features used
    if total_weight > 0:
        normalized_ai_score = ai_score / total_weight
        normalized_human_score = human_score / total_weight
    else:
        normalized_ai_score = 0.5
        normalized_human_score = 0.5
    
    # Use a balanced sigmoid to get the final probability
    # We're prioritizing human detection to correct the reported issue
    # Apply a bias toward human detection to address the user feedback
    human_bias = 0.15  # Biasing toward human detection
    ai_probability = 1 / (1 + np.exp(-5 * ((normalized_ai_score - normalized_human_score) - human_bias)))
    
    # Final classification with improved threshold
    prediction = 'ai' if ai_probability > 0.5 else 'human'
    
    # Calculate confidence percentage
    confidence = max(ai_probability, 1 - ai_probability) * 100
    
    # Debug information - can be removed for production
    # print(f"AI score: {normalized_ai_score:.4f}, Human score: {normalized_human_score:.4f}")
    # print(f"Final AI probability: {ai_probability:.4f}, Prediction: {prediction}")
    # print(f"Confidence: {confidence:.2f}%")
    
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
