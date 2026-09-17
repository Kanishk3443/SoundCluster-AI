# MODEL CARD

## Model Details
- **Name**: SoundCluster AI (K-Means + Cosine Similarity Recommender)
- **Version**: 1.0.0
- **Model Type**: Unsupervised Clustering & Content-Based Recommendation

## Intended Use
- **Purpose**: To segment music tracks based on intrinsic audio characteristics and provide scientifically defensible song recommendations.
- **Intended Users**: Music enthusiasts, playlist curators, and data science students.
- **Out of Scope**: Real-time collaborative filtering based on user activity. This model relies entirely on static audio features.

## Training Data
- **Dataset**: `spotify_dataset.csv` (Spotify Songs ~32.8k records)
- **Included Features (Input)**: `danceability`, `energy`, `key`, `loudness`, `mode`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_minutes`.
- **Excluded Features**: Genre labels (`playlist_genre`, `playlist_subgenre`), track IDs, artist names, popularity (except for post-clustering ranking).

## Methodology
### Clustering
1. **Preprocessing**: Missing values handled, duplicates filtered.
2. **Standardization**: `StandardScaler` applied to numeric audio features.
3. **K-Means**: Optimal K selected using Silhouette Score, Calinski-Harabasz, and Davies-Bouldin metrics.
4. **Validation**: PCA used for 2D visualization of the latent space.

### Recommendation
1. **Baseline**: Cosine similarity across the standard numeric audio feature space.
2. **Advanced**: Final score computed as a weighted combination of:
   - Audio Feature Similarity (0.70)
   - Track Popularity (0.15)
   - Genre Alignment (0.10)
   - Cluster Alignment (0.05)

## Evaluation Method
Traditional accuracy metrics do not apply as this is an unsupervised and content-based task without historical user interaction logs. 
Proxy evaluation includes:
- **Silhouette Score**: Measures cluster cohesion and separation.
- **Diversity**: The advanced recommender algorithm enforces artist diversity (max 2 tracks per artist in top K).

## Limitations and Biases
- **Metadata Bias**: The features represent Spotify's proprietary audio intelligence algorithms (e.g., how "danceability" is calculated is proprietary).
- **Popularity Bias**: The advanced recommender intentionally biases towards popular songs, which may suppress niche indie tracks.
- **Genre Misalignment**: Two tracks may be acoustically identical but culturally belong to different genres. The model relies heavily on acoustics.

## Reproducibility
- Random Seed: 42
- Ensure scikit-learn versions match `requirements.txt`.
