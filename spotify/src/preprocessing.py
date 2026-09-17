import pandas as pd
import numpy as np
import logging
from src.config import CATEGORICAL_FEATURES, EXCLUDED_FROM_CLUSTERING

logger = logging.getLogger(__name__)

def clean_data(df):
    """
    Cleans the dataset:
    - Handles missing values
    - Handles duplicates
    - Parses dates safely
    """
    logger.info("Starting data cleaning...")
    
    # 1. Handle Duplicates
    initial_rows = df.shape[0]
    df = df.drop_duplicates(keep='first')
    logger.info(f"Dropped {initial_rows - df.shape[0]} exact duplicate rows.")
    
    # We do NOT drop duplicated track_ids because songs can belong to multiple playlists
    # and playlist_genre / playlist_subgenre are useful.
    
    # 2. Handle Missing Values
    missing_before = df.isnull().sum()
    
    # Text metadata
    text_cols = ['track_name', 'track_artist', 'track_album_name']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
            
    # Remove rows where crucial audio features are missing (if any)
    audio_features = [
        "danceability", "energy", "key", "loudness", "mode", 
        "speechiness", "acousticness", "instrumentalness", 
        "liveness", "valence", "tempo", "duration_ms"
    ]
    df = df.dropna(subset=audio_features)
    logger.info(f"Dropped rows with missing audio features. Current shape: {df.shape}")
    
    # 3. Date Processing
    if 'track_album_release_date' in df.columns:
        # Some dates are just 'YYYY', some are 'YYYY-MM', some 'YYYY-MM-DD'
        df['release_date_parsed'] = pd.to_datetime(df['track_album_release_date'], format='mixed', errors='coerce')
        df['release_year'] = df['release_date_parsed'].dt.year
        df['release_month'] = df['release_date_parsed'].dt.month
        # For missing years, impute with median
        df['release_year'] = df['release_year'].fillna(df['release_year'].median())
        df['release_month'] = df['release_month'].fillna(1) # Default to January if unknown
    
    # 4. Handle categorical anomalies
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
            df[col] = df[col].astype(str).str.strip().str.lower()
    
    logger.info("Data cleaning completed.")
    return df
