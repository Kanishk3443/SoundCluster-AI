import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Dataset
DATASET_FILENAME = "spotify_dataset.csv"
DATASET_PATH = RAW_DATA_DIR / DATASET_FILENAME
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "spotify_processed.csv"

# Model paths
SCALER_PATH = MODELS_DIR / "scaler.joblib"
KMEANS_PATH = MODELS_DIR / "kmeans.joblib"
RECOMMENDATION_MATRIX_PATH = MODELS_DIR / "recommendation_matrix.joblib"
METADATA_PATH = MODELS_DIR / "metadata.joblib"

# Random Seed for Reproducibility
RANDOM_STATE = 42

# Clustering configuration
K_MIN = 2
K_MAX = 12

# Recommendation configuration
TOP_K_RECOMMENDATIONS = 10
REC_WEIGHT_AUDIO_SIMILARITY = 0.70
REC_WEIGHT_POPULARITY = 0.15
REC_WEIGHT_GENRE_ALIGN = 0.10
REC_WEIGHT_CLUSTER_ALIGN = 0.05

# Feature Lists
NUMERICAL_AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode", 
    "speechiness", "acousticness", "instrumentalness", 
    "liveness", "valence", "tempo", "duration_ms"
]

CATEGORICAL_FEATURES = [
    "playlist_genre", "playlist_subgenre"
]

EXCLUDED_FROM_CLUSTERING = [
    "track_id", "track_name", "track_artist", "track_album_id", 
    "track_album_name", "track_album_release_date", "playlist_name", 
    "playlist_id", "playlist_genre", "playlist_subgenre", "track_popularity"
]
