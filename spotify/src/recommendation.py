import pandas as pd
import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from src.config import RECOMMENDATION_MATRIX_PATH, METADATA_PATH, TOP_K_RECOMMENDATIONS

logger = logging.getLogger(__name__)

def build_recommendation_system(df_full, df_scaled, cluster_labels):
    """
    Builds and saves the recommendation matrices/metadata.
    For performance on ~32k tracks, we'll save the scaled feature matrix
    and metadata, then compute similarities on the fly or save a condensed version.
    """
    logger.info("Building recommendation system...")
    
    # Store essential metadata
    metadata = df_full[['track_id', 'track_name', 'track_artist', 'playlist_genre', 'playlist_subgenre', 'track_popularity']].copy()
    metadata['cluster'] = cluster_labels
    
    # Ensure no duplicates in track_id to make lookup simple
    # We will just keep the first instance of each track for recommendation pool
    _, unique_indices = np.unique(metadata['track_id'], return_index=True)
    metadata_unique = metadata.iloc[unique_indices].reset_index(drop=True)
    df_scaled_unique = df_scaled.iloc[unique_indices].reset_index(drop=True)
    
    # Save the matrices
    RECOMMENDATION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(df_scaled_unique, RECOMMENDATION_MATRIX_PATH)
    joblib.dump(metadata_unique, METADATA_PATH)
    
    logger.info(f"Recommendation data saved. Pool size: {len(metadata_unique)} unique tracks.")
    return df_scaled_unique, metadata_unique

def get_baseline_recommendations(track_idx, df_scaled_unique, metadata_unique, top_k=TOP_K_RECOMMENDATIONS):
    """
    Baseline recommender using pure cosine similarity.
    """
    target_vector = df_scaled_unique.iloc[track_idx].values.reshape(1, -1)
    
    # Compute similarity against all tracks
    similarities = cosine_similarity(target_vector, df_scaled_unique)[0]
    
    # Get top K indices (ignoring the track itself)
    # argsort sorts ascending, so take from the end, skip the very last (which is the track itself if sim=1.0)
    # More safely: set self-similarity to -1
    similarities[track_idx] = -1.0
    
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    recs = metadata_unique.iloc[top_indices].copy()
    recs['similarity_score'] = similarities[top_indices]
    
    return recs

def get_advanced_recommendations(track_idx, df_scaled_unique, metadata_unique, top_k=TOP_K_RECOMMENDATIONS):
    """
    Advanced recommender using similarity, popularity, cluster, and genre alignment.
    """
    target_vector = df_scaled_unique.iloc[track_idx].values.reshape(1, -1)
    target_meta = metadata_unique.iloc[track_idx]
    
    similarities = cosine_similarity(target_vector, df_scaled_unique)[0]
    similarities[track_idx] = -1.0
    
    # Advanced scoring
    # Normalize popularity to 0-1
    pop = metadata_unique['track_popularity'].fillna(0)
    pop_norm = pop / 100.0 if pop.max() <= 100 else pop / pop.max()
    
    # Cluster alignment
    cluster_align = (metadata_unique['cluster'] == target_meta['cluster']).astype(float)
    
    # Genre alignment
    genre_align = (metadata_unique['playlist_genre'] == target_meta['playlist_genre']).astype(float)
    
    from src.config import REC_WEIGHT_AUDIO_SIMILARITY, REC_WEIGHT_POPULARITY, REC_WEIGHT_GENRE_ALIGN, REC_WEIGHT_CLUSTER_ALIGN
    
    final_scores = (
        REC_WEIGHT_AUDIO_SIMILARITY * similarities +
        REC_WEIGHT_POPULARITY * pop_norm +
        REC_WEIGHT_GENRE_ALIGN * genre_align +
        REC_WEIGHT_CLUSTER_ALIGN * cluster_align
    )
    final_scores[track_idx] = -1.0 # Ensure not recommending itself
    
    # Filter for artist diversity: max 2 from same artist
    sorted_indices = np.argsort(final_scores)[::-1]
    
    final_recs_indices = []
    artist_counts = {}
    
    for idx in sorted_indices:
        artist = metadata_unique.iloc[idx]['track_artist']
        if artist_counts.get(artist, 0) < 2:
            final_recs_indices.append(idx)
            artist_counts[artist] = artist_counts.get(artist, 0) + 1
        
        if len(final_recs_indices) == top_k:
            break
            
    recs = metadata_unique.iloc[final_recs_indices].copy()
    recs['final_score'] = final_scores[final_recs_indices]
    recs['audio_similarity'] = similarities[final_recs_indices]
    
    return recs
