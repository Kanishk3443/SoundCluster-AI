import pandas as pd
import numpy as np
import logging
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from src.config import KMEANS_PATH, FIGURES_DIR, K_MIN, K_MAX, RANDOM_STATE

logger = logging.getLogger(__name__)

def evaluate_k(df_scaled):
    """
    Evaluate multiple values of K and save metrics and plots.
    Returns the optimal K based on silhouette score.
    """
    logger.info("Evaluating optimal K...")
    
    k_values = list(range(K_MIN, K_MAX + 1))
    inertia_scores = []
    silhouette_scores = []
    db_scores = []
    ch_scores = []
    
    # Sample data if too large to speed up evaluation
    eval_data = df_scaled.sample(n=min(10000, len(df_scaled)), random_state=RANDOM_STATE) if len(df_scaled) > 10000 else df_scaled
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto')
        labels = kmeans.fit_predict(eval_data)
        
        inertia_scores.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(eval_data, labels))
        db_scores.append(davies_bouldin_score(eval_data, labels))
        ch_scores.append(calinski_harabasz_score(eval_data, labels))
        
        logger.info(f"K={k} - Silhouette: {silhouette_scores[-1]:.4f}")

    # Plot metrics
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(k_values, inertia_scores, 'bo-')
    plt.title('Elbow Curve (Inertia)')
    plt.xlabel('Number of Clusters (K)')
    
    plt.subplot(2, 2, 2)
    plt.plot(k_values, silhouette_scores, 'ro-')
    plt.title('Silhouette Score')
    plt.xlabel('Number of Clusters (K)')
    
    plt.subplot(2, 2, 3)
    plt.plot(k_values, ch_scores, 'go-')
    plt.title('Calinski-Harabasz Score')
    plt.xlabel('Number of Clusters (K)')
    
    plt.subplot(2, 2, 4)
    plt.plot(k_values, db_scores, 'ko-')
    plt.title('Davies-Bouldin Score')
    plt.xlabel('Number of Clusters (K)')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'clustering_metrics.png', dpi=300)
    plt.close()
    
    # Choose optimal K based on silhouette score
    best_k = k_values[np.argmax(silhouette_scores)]
    logger.info(f"Optimal K selected: {best_k}")
    return best_k

def train_kmeans(df_scaled, best_k):
    """
    Train final K-Means model on full scaled dataset.
    """
    logger.info(f"Training K-Means with K={best_k}...")
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init='auto')
    labels = kmeans.fit_predict(df_scaled)
    
    # Save the model
    KMEANS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans, KMEANS_PATH)
    logger.info(f"K-Means model saved to {KMEANS_PATH}")
    
    return labels, kmeans

def visualize_clusters_pca(df_scaled, labels, df_full=None):
    """
    Visualize clusters using PCA.
    """
    logger.info("Performing PCA for visualization...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_result = pca.fit_transform(df_scaled)
    
    explained_var = pca.explained_variance_ratio_
    logger.info(f"PCA Explained Variance: PC1={explained_var[0]:.4f}, PC2={explained_var[1]:.4f}")
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=pca_result[:, 0], y=pca_result[:, 1],
        hue=labels,
        palette='tab10',
        alpha=0.6,
        s=20
    )
    plt.title(f'PCA Cluster Visualization (PC1: {explained_var[0]:.1%}, PC2: {explained_var[1]:.1%})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'pca_clusters.png', dpi=300)
    plt.close()
    
    if df_full is not None and 'playlist_genre' in df_full.columns:
        plt.figure(figsize=(12, 10))
        sns.scatterplot(
            x=pca_result[:, 0], y=pca_result[:, 1],
            hue=df_full['playlist_genre'],
            palette='Set1',
            alpha=0.6,
            s=20
        )
        plt.title('PCA Visualization by Playlist Genre')
        plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'pca_genres.png', dpi=300)
        plt.close()

def profile_clusters(df, labels):
    """
    Profile clusters based on original features and labels.
    """
    df_profile = df.copy()
    df_profile['cluster'] = labels
    
    # Numeric profile
    numeric_cols = df_profile.select_dtypes(include=[np.number]).columns.drop('cluster', errors='ignore')
    cluster_profiles = df_profile.groupby('cluster')[numeric_cols].mean()
    
    logger.info(f"Cluster Profiles (Numerical):\n{cluster_profiles}")
    
    # Genre profile
    if 'playlist_genre' in df.columns:
        genre_dist = pd.crosstab(df_profile['cluster'], df_profile['playlist_genre'], normalize='index') * 100
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(genre_dist, annot=True, cmap='YlGnBu', fmt=".1f")
        plt.title('Genre Distribution per Cluster (%)')
        plt.ylabel('Cluster')
        plt.xlabel('Playlist Genre')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'cluster_genre_heatmap.png', dpi=300)
        plt.close()
        
    return cluster_profiles
