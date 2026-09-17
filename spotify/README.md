# SoundCluster AI: Spotify Music Intelligence & Recommendation System

## Overview
SoundCluster AI is an end-to-end unsupervised machine learning portfolio project designed to perform genre segmentation, track clustering, and intelligent music recommendations using Spotify audio features.

This project was built from scratch and includes comprehensive data cleaning, Exploratory Data Analysis (EDA), K-Means clustering, PCA visualization, and a content-based recommendation engine. The entire pipeline is bundled with a polished Streamlit dashboard.

## Internship Skill Mapping
* **Python** → complete project implementation, modular architecture
* **Pandas** → data cleaning, validation, and feature engineering
* **NumPy** → numerical computation and similarity matrix operations
* **Matplotlib** → static and exploratory visualizations
* **Seaborn** → statistical correlation heatmaps and distribution plots
* **Scikit-learn** → standardization, PCA, K-Means clustering, metrics, cosine similarity
* **Statistics** → EDA, outlier handling, and correlation analysis
* **Machine Learning** → determining optimal K, cluster profiling
* **Unsupervised Learning** → feature-based track segmentation
* **AI Recommendation** → content-based, cluster-aware engine

## Architecture
```text
RAW CSV
   ↓
Data Validation & Cleaning
   ↓
Feature Engineering & Selection
   ↓
StandardScaler
   ↓
K-Means Clustering
   ↓
Cluster Profiling & PCA
   ↓
Similarity Engine
   ↓
Recommendation Ranking
   ↓
Streamlit Dashboard
```

## Setup & Installation

1. Create a virtual environment (optional but recommended)
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place `spotify_dataset.csv` in `data/raw/`

## Usage

### 1. Run the Backend Pipeline
You can run the full machine learning pipeline (cleaning, EDA, training, evaluation, saving models):
```bash
python run_project.py --all
```
*Note: Depending on dataset size, PCA and K-Means training may take a minute.*

### 2. Launch the Streamlit Dashboard
```bash
streamlit run app/app.py
```

## Features
- **Data Explorer**: Filter tracks by genre and popularity.
- **Cluster Profiling**: Track segmentation based on pure audio features (danceability, energy, tempo, etc.) without cheating with genre labels.
- **Recommendation Engine**: Find mathematically similar tracks based on audio fingerprint, weighted by popularity and cluster alignment.
- **Explanations**: The recommendation system explains why a track was recommended.

## Future Scope
- Spotify API Integration for real-time recommendations.
- Collaborative filtering using user listening history.
- Deep learning embeddings.
