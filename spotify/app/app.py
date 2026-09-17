import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import (
    PROCESSED_DATASET_PATH, SCALER_PATH, KMEANS_PATH, 
    RECOMMENDATION_MATRIX_PATH, METADATA_PATH, NUMERICAL_AUDIO_FEATURES
)
from src.recommendation import get_advanced_recommendations, get_baseline_recommendations

st.set_page_config(page_title="SoundCluster AI", page_icon="🎵", layout="wide")

@st.cache_data
def load_data():
    if not PROCESSED_DATASET_PATH.exists():
        st.error("Processed dataset not found! Please run the backend pipeline first.")
        st.stop()
    return pd.read_csv(PROCESSED_DATASET_PATH)

@st.cache_resource
def load_models():
    try:
        scaler = joblib.load(SCALER_PATH)
        kmeans = joblib.load(KMEANS_PATH)
        df_scaled_unique = joblib.load(RECOMMENDATION_MATRIX_PATH)
        metadata_unique = joblib.load(METADATA_PATH)
        return scaler, kmeans, df_scaled_unique, metadata_unique
    except Exception as e:
        st.error(f"Failed to load models. Did you run the training pipeline? Error: {e}")
        st.stop()

def dashboard_page(df):
    st.title("🎵 SoundCluster AI Dashboard")
    st.markdown("### Spotify Music Intelligence & Recommendation System")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Songs", f"{len(df):,}")
    col2.metric("Unique Artists", f"{df['track_artist'].nunique():,}")
    col3.metric("Genres", f"{df['playlist_genre'].nunique()}")
    col4.metric("Avg Popularity", f"{df['track_popularity'].mean():.1f}")
    
    st.markdown("---")
    
    colA, colB = st.columns(2)
    with colA:
        genre_counts = df['playlist_genre'].value_counts().reset_index()
        genre_counts.columns = ['Genre', 'Count']
        fig = px.bar(genre_counts, x='Genre', y='Count', title="Songs by Genre", color='Genre')
        st.plotly_chart(fig, use_container_width=True)
        
    with colB:
        fig = px.histogram(df, x='track_popularity', title="Popularity Distribution", nbins=30)
        st.plotly_chart(fig, use_container_width=True)

def data_explorer_page(df):
    st.title("🔍 Data Explorer")
    
    st.sidebar.header("Filters")
    selected_genre = st.sidebar.multiselect("Genre", options=df['playlist_genre'].unique())
    min_pop = st.sidebar.slider("Minimum Popularity", 0, 100, 0)
    
    filtered_df = df.copy()
    if selected_genre:
        filtered_df = filtered_df[filtered_df['playlist_genre'].isin(selected_genre)]
    filtered_df = filtered_df[filtered_df['track_popularity'] >= min_pop]
    
    st.write(f"Showing {len(filtered_df):,} tracks")
    st.dataframe(filtered_df[['track_name', 'track_artist', 'playlist_genre', 'track_popularity', 'danceability', 'energy', 'tempo']])

def recommendation_page(df, df_scaled_unique, metadata_unique):
    st.title("🎧 Recommendation Engine")
    
    # Search for a song
    search_query = st.text_input("Search for a song or artist:")
    
    if search_query:
        mask = metadata_unique['track_name'].str.contains(search_query, case=False, na=False) | \
               metadata_unique['track_artist'].str.contains(search_query, case=False, na=False)
        results = metadata_unique[mask].head(10)
        
        if len(results) == 0:
            st.warning("No songs found. Try another query.")
            return
            
        selected_track = st.selectbox("Select a track:", options=results.index, 
                                     format_func=lambda x: f"{results.loc[x, 'track_name']} by {results.loc[x, 'track_artist']} ({results.loc[x, 'playlist_genre']})")
        
        if selected_track is not None:
            num_recs = st.slider("Number of recommendations", 5, 20, 10)
            rec_type = st.radio("Recommendation Type", ["Advanced (Cluster+Pop+Sim)", "Baseline (Pure Sim)"])
            
            if st.button("Generate Recommendations"):
                with st.spinner("Finding perfect matches..."):
                    if rec_type.startswith("Advanced"):
                        recs = get_advanced_recommendations(selected_track, df_scaled_unique, metadata_unique, top_k=num_recs)
                    else:
                        recs = get_baseline_recommendations(selected_track, df_scaled_unique, metadata_unique, top_k=num_recs)
                
                st.subheader("Selected Track")
                target = metadata_unique.loc[selected_track]
                st.info(f"**{target['track_name']}** by **{target['track_artist']}** | Genre: {target['playlist_genre']} | Cluster: {target['cluster']}")
                
                st.subheader("Recommended Tracks")
                for _, row in recs.iterrows():
                    score_col = 'final_score' if 'final_score' in row else 'similarity_score'
                    with st.expander(f"🎵 {row['track_name']} by {row['track_artist']} (Score: {row[score_col]:.2f})"):
                        st.write(f"**Genre:** {row['playlist_genre']} | **Subgenre:** {row['playlist_subgenre']}")
                        st.write(f"**Cluster:** {row['cluster']} | **Popularity:** {row['track_popularity']}")
                        
                        st.markdown("**Explanation:**")
                        st.write(f"- Discovered Cluster Match: {'Yes' if row['cluster'] == target['cluster'] else 'No'}")
                        if 'audio_similarity' in row:
                            st.write(f"- Audio Feature Similarity: {row['audio_similarity']:.2f}")

def main():
    df = load_data()
    scaler, kmeans, df_scaled_unique, metadata_unique = load_models()
    
    st.sidebar.title("SoundCluster AI")
    page = st.sidebar.radio("Navigate", ["Dashboard", "Data Explorer", "Recommendation Engine"])
    
    if page == "Dashboard":
        dashboard_page(df)
    elif page == "Data Explorer":
        data_explorer_page(df)
    elif page == "Recommendation Engine":
        recommendation_page(df, df_scaled_unique, metadata_unique)

if __name__ == "__main__":
    main()
