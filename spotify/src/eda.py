import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from src.config import FIGURES_DIR, NUMERICAL_AUDIO_FEATURES

logger = logging.getLogger(__name__)

def generate_eda(df):
    """
    Generates Exploratory Data Analysis figures.
    """
    logger.info("Starting EDA...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    _plot_genre_distribution(df)
    _plot_audio_feature_distributions(df)
    _plot_correlation_matrix(df)
    _plot_popularity_analysis(df)
    
    logger.info("EDA completed.")

def _plot_genre_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(y='playlist_genre', data=df, order=df['playlist_genre'].value_counts().index, palette='viridis')
    plt.title('Distribution of Playlist Genres')
    plt.xlabel('Count')
    plt.ylabel('Genre')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'genre_distribution.png', dpi=300)
    plt.close()

def _plot_audio_feature_distributions(df):
    # Plotting histograms for all numerical audio features
    num_features = [f for f in NUMERICAL_AUDIO_FEATURES if f in df.columns]
    
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 16))
    axes = axes.flatten()
    
    for i, feature in enumerate(num_features):
        if i < len(axes):
            sns.histplot(df[feature], bins=30, kde=True, ax=axes[i], color='royalblue')
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')
            
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'audio_features_distribution.png', dpi=300)
    plt.close()

def _plot_correlation_matrix(df):
    num_features = [f for f in NUMERICAL_AUDIO_FEATURES if f in df.columns]
    if 'track_popularity' in df.columns:
        num_features.append('track_popularity')
        
    corr_matrix = df[num_features].corr(method='pearson')
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'feature_correlation_heatmap.png', dpi=300)
    plt.close()

    # Log strong correlations
    corr_unstacked = corr_matrix.unstack()
    strong_corrs = corr_unstacked[(abs(corr_unstacked) > 0.6) & (corr_unstacked != 1.0)]
    strong_corrs = strong_corrs.drop_duplicates()
    logger.info(f"Top correlations:\n{strong_corrs}")

def _plot_popularity_analysis(df):
    if 'track_popularity' not in df.columns or 'playlist_genre' not in df.columns:
        return
        
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='playlist_genre', y='track_popularity', data=df, palette='Set2')
    plt.title('Track Popularity by Playlist Genre')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'popularity_by_genre.png', dpi=300)
    plt.close()
