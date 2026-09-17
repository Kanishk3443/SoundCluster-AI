import pytest
import pandas as pd
import numpy as np
from src.config import CATEGORICAL_FEATURES, NUMERICAL_AUDIO_FEATURES
from src.preprocessing import clean_data
from src.feature_engineering import scale_features

@pytest.fixture
def sample_data():
    data = {
        'track_id': ['1', '2', '2', '3'],
        'track_name': ['Song A', 'Song B', 'Song B', None],
        'track_artist': ['Artist A', 'Artist B', 'Artist B', 'Artist C'],
        'track_popularity': [80, 40, 40, 50],
        'playlist_genre': ['pop', 'rock', 'rock', None],
        'playlist_subgenre': ['dance pop', 'classic rock', 'classic rock', 'indie'],
        'danceability': [0.8, 0.5, 0.5, 0.6],
        'energy': [0.9, 0.6, 0.6, 0.7],
        'key': [1, 2, 2, 3],
        'loudness': [-5.0, -8.0, -8.0, -6.0],
        'mode': [1, 0, 0, 1],
        'speechiness': [0.05, 0.04, 0.04, 0.05],
        'acousticness': [0.1, 0.8, 0.8, 0.5],
        'instrumentalness': [0.0, 0.1, 0.1, 0.0],
        'liveness': [0.1, 0.2, 0.2, 0.15],
        'valence': [0.8, 0.4, 0.4, 0.6],
        'tempo': [120.0, 90.0, 90.0, 110.0],
        'duration_ms': [200000, 180000, 180000, 190000]
    }
    return pd.DataFrame(data)

def test_clean_data(sample_data):
    df_clean = clean_data(sample_data)
    
    # Check duplicate removal
    assert len(df_clean) == 3
    
    # Check null imputation
    assert df_clean['track_name'].isnull().sum() == 0
    assert df_clean['playlist_genre'].isnull().sum() == 0
    
    # Check text standardization
    assert 'unknown' in df_clean['playlist_genre'].values

def test_scale_features(sample_data):
    df_clean = clean_data(sample_data)
    df_scaled, features = scale_features(df_clean)
    
    # Check output shape
    assert df_scaled.shape[0] == 3
    assert len(features) == len(NUMERICAL_AUDIO_FEATURES)
    
    # Mean should be close to 0
    assert np.isclose(df_scaled['danceability'].mean(), 0, atol=1e-7)
    
    # Std should be close to 1 (for N=2, std with ddof=0 is 1.0)
    assert np.isclose(df_scaled['danceability'].std(ddof=0), 1.0, atol=1e-7)
