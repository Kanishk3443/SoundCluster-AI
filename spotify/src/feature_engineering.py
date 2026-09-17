import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
import joblib
from src.config import SCALER_PATH, NUMERICAL_AUDIO_FEATURES

logger = logging.getLogger(__name__)

def engineer_features(df):
    """
    Feature engineering pipeline.
    """
    logger.info("Starting feature engineering...")
    
    # Duration conversion (ms to minutes)
    if 'duration_ms' in df.columns:
        df['duration_minutes'] = df['duration_ms'] / 60000.0
    
    return df

def scale_features(df):
    """
    Standardize numerical audio features.
    Saves the scaler to disk.
    """
    logger.info("Scaling features...")
    
    scaler = StandardScaler()
    
    features_to_scale = [f for f in NUMERICAL_AUDIO_FEATURES if f in df.columns]
    # Use duration_minutes instead of duration_ms if available
    if 'duration_minutes' in df.columns and 'duration_ms' in features_to_scale:
        features_to_scale.remove('duration_ms')
        features_to_scale.append('duration_minutes')
        
    scaled_data = scaler.fit_transform(df[features_to_scale])
    
    # Save the scaler
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    logger.info(f"Scaler saved to {SCALER_PATH}")
    
    df_scaled = pd.DataFrame(scaled_data, columns=features_to_scale, index=df.index)
    return df_scaled, features_to_scale
