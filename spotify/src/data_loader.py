import pandas as pd
import logging
from pathlib import Path
from src.config import DATASET_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(filepath=None):
    """
    Loads the dataset from the specified path or config default.
    """
    path = Path(filepath) if filepath else DATASET_PATH
    if not path.exists():
        logger.error(f"Dataset not found at {path}")
        raise FileNotFoundError(f"Dataset not found at {path}")
    
    logger.info(f"Loading dataset from {path}")
    try:
        df = pd.read_csv(path, encoding="utf-8")
        logger.info(f"Dataset loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

def validate_schema(df):
    """
    Validates that the required columns are present.
    """
    required_columns = [
        "track_id", "track_name", "track_artist", "track_popularity",
        "track_album_id", "track_album_name", "track_album_release_date",
        "playlist_name", "playlist_id", "playlist_genre", "playlist_subgenre",
        "danceability", "energy", "key", "loudness", "mode", "speechiness",
        "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms"
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing columns in dataset: {missing_columns}")
        raise ValueError(f"Dataset schema invalid. Missing columns: {missing_columns}")
    
    logger.info("Dataset schema validation passed.")
    return True

def generate_data_profile(df, output_path=None):
    """
    Generates a basic data profile summary and prints it.
    """
    print("="*50)
    print("DATASET PROFILE SUMMARY")
    print("="*50)
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
    
    print("\nMissing Values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")
    print(f"Duplicated Track IDs: {df.duplicated(subset=['track_id']).sum()}")
    
    print(f"\nUnique Genres: {df['playlist_genre'].nunique()}")
    print(f"Unique Subgenres: {df['playlist_subgenre'].nunique()}")
    print(f"Unique Artists: {df['track_artist'].nunique()}")
    print(f"Unique Playlists: {df['playlist_id'].nunique()}")
    print("="*50)
