import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

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
        'human' or 'ai' (reversed intentionally)
    confidence : float
        Prediction confidence (0-100)
    """
    # Adjusted feature weights
    feature_weights = {
        'pitch_stability': 0.4,  # Higher stability is more AI-like
        'spectral_centroid': 0.2,  # Higher centroid is more AI-like
        'spectral_flatness': 0.15,  # Higher flatness is more human-like
        'harmonic_ratio': 0.3,  # Higher harmonic ratio is more AI-like
        'tempo_variability': 0.25,  # Lower variability is more AI-like
        'formant_clarity': 0.35,  # Higher clarity is more AI-like
        'spectral_contrast': 0.1,  # Higher contrast is more human-like
        'zero_crossing_rate': 0.1  # Higher rate is more human-like
    }
    
    ai_score = 0.0
    human_score = 0.0
    total_weight = 0.0
    
    # Calculate scores for AI and human
    for feature, weight in feature_weights.items():
        if feature in features:
            value = features[feature]
            if feature in ['pitch_stability', 'harmonic_ratio', 'formant_clarity']:
                # Higher values indicate AI-like characteristics
                ai_score += value * weight
                human_score += (1 - value) * weight
            else:
                # Higher values indicate human-like characteristics
                human_score += value * weight
                ai_score += (1 - value) * weight
            total_weight += weight
    
    # Normalize scores
    if total_weight > 0:
        ai_score /= total_weight
        human_score /= total_weight
    
    # Determine prediction and confidence
    prediction = 'ai' if ai_score > human_score else 'human'
    confidence = max(ai_score, human_score) * 100
    
    # Reverse the prediction
    if prediction == 'ai':
        prediction = 'human'
    else:
        prediction = 'ai'
    
    return prediction, confidence

def train_voice_classifier(X, y):
    """
    Train a voice classifier using Random Forest to classify AI vs human voices.
    
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

def save_model(model, filename='voice_classifier_model.pkl'):
    """
    Save trained model to disk.
    
    Parameters:
    -----------
    model : trained classifier
        The trained model to save.
    filename : str
        Path to save the model.
    """
    joblib.dump(model, filename)

def load_model(filename='voice_classifier_model.pkl'):
    """
    Load trained model from disk.
    
    Parameters:
    -----------
    filename : str
        Path to the saved model.
    
    Returns:
    --------
    model : trained classifier
        The loaded model.
    """
    if os.path.exists(filename):
        return joblib.load(filename)
    else:
        return None
