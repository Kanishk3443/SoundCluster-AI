# PROJECT REPORT

## 1. Abstract
This project presents an unsupervised machine learning pipeline to segment Spotify tracks based purely on their acoustic features and recommend similar tracks using a content-based recommendation approach.

## 2. Introduction
Music recommendation is traditionally done via collaborative filtering (using user listening histories). However, the "cold start" problem exists for new songs. This project solves this by recommending songs based entirely on the intrinsic mathematical structure of the audio itself.

## 3. Problem Statement
Given a large dataset of tracks with audio features (like danceability, energy, acousticness) and metadata, how can we mathematically group songs together and build an intelligent recommendation engine without relying on subjective genre labels?

## 4. Objectives
- Perform extensive EDA on Spotify audio features.
- Segment tracks using K-Means clustering to discover latent acoustic groups.
- Build a recommendation engine combining mathematical similarity, popularity, and cluster alignment.

## 5. Dataset Description
The dataset contains approximately 32,800 tracks with detailed metadata (track name, artist, genre, subgenre, popularity) and 12 numeric audio features.

## 6. Feature Description
Key numeric features:
- **Danceability**: Suitability for dancing based on tempo and rhythm.
- **Energy**: Perceptual measure of intensity and activity.
- **Acousticness**: Confidence measure of whether the track is acoustic.
- **Valence**: Musical positiveness conveyed by a track.

## 7. Data Preprocessing
Duplicates were managed, missing string metadata imputed with 'Unknown', and durations converted to minutes for better scale interpretability.

## 8. Exploratory Data Analysis
EDA highlighted that certain genres strongly align with specific numeric features (e.g., EDM with high Energy). Popularity distributions showed heavy skewness toward 0 (many unpopular tracks) and a normal distribution around 50 for the rest.

## 9. Correlation Analysis
A Pearson correlation matrix revealed strong positive correlations between Loudness and Energy, and strong negative correlations between Energy and Acousticness. 

## 10. Feature Engineering
String metadata and popularity were explicitly excluded from the clustering input to prevent the model from simply memorizing labels. All numeric inputs were standardized using `StandardScaler`.

## 11. Machine Learning Methodology
An unsupervised approach was taken. We used the K-Means algorithm to partition the latent space.

## 12. K-Means Clustering
The model trained on the normalized, 12-dimensional acoustic feature space.

## 13. Optimal Cluster Selection
We evaluated K from 2 to 12. The Silhouette Score guided the final selection, balancing mathematical cohesion with interpretability. (The exact optimal K is dynamic and determined at runtime).

## 14. Cluster Analysis
Post-clustering, we profiled the clusters. We cross-referenced the discovered mathematical clusters with the human-assigned Spotify genre labels to see how well acoustic groupings align with subjective definitions.

## 15. Recommendation System
A content-based recommendation engine was developed.

## 16. Recommendation Methodology
**Baseline Recommender**: Pure cosine similarity.
**Advanced Recommender**:
- Similarity (70%)
- Popularity (15%)
- Genre Alignment (10%)
- Discovered Cluster Alignment (5%)
- Hard constraint: Maximum 2 tracks per artist.

## 17. Evaluation
Proxy metrics were used. A good cluster-aware recommendation algorithm will naturally maintain high average similarity while ensuring artist diversity.

## 18. Results
The Streamlit dashboard successfully serves dynamic recommendations, proving that unsupervised feature analysis is highly effective at discovering songs that truly sound alike.

## 19. Limitations
Spotify's audio features are proprietary black boxes. If Spotify's algorithm for "danceability" is flawed, this model inherits that flaw.

## 20. Ethical Considerations
The recommendation algorithm explicitly boosts popular tracks, which could contribute to the suppression of niche/indie artists. The artist diversity penalty helps mitigate this.

## 21. Future Improvements
- Integrating Spotify's Web API.
- Deep learning (Neural Collaborative Filtering).

## 22. Conclusion
Acoustic feature analysis provides an objective, robust foundation for genre segmentation and track recommendation.
