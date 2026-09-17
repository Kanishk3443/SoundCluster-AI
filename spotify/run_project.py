import argparse
import logging
import warnings
import pandas as pd
from src.data_loader import load_data, validate_schema, generate_data_profile
from src.preprocessing import clean_data
from src.eda import generate_eda
from src.feature_engineering import engineer_features, scale_features
from src.clustering import evaluate_k, train_kmeans, visualize_clusters_pca, profile_clusters
from src.recommendation import build_recommendation_system
from src.config import PROCESSED_DATASET_PATH

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="SoundCluster AI Pipeline")
    parser.add_argument("--profile", action="store_true", help="Load and profile raw data")
    parser.add_argument("--preprocess", action="store_true", help="Clean and preprocess data")
    parser.add_argument("--eda", action="store_true", help="Generate EDA figures")
    parser.add_argument("--train", action="store_true", help="Train clustering and build recommendation models")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        parser.print_help()
        return

    # 1. Profile / Validate
    if args.profile or args.all:
        logger.info("--- Phase 1: Data Profiling ---")
        df_raw = load_data()
        validate_schema(df_raw)
        generate_data_profile(df_raw)
    
    # 2. Preprocess
    if args.preprocess or args.eda or args.train or args.all:
        logger.info("--- Phase 2: Preprocessing ---")
        df_raw = load_data()
        df_clean = clean_data(df_raw)
        df_engineered = engineer_features(df_clean)
        
        PROCESSED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_engineered.to_csv(PROCESSED_DATASET_PATH, index=False)
        logger.info(f"Processed dataset saved to {PROCESSED_DATASET_PATH}")
    
    # 3. EDA
    if args.eda or args.all:
        logger.info("--- Phase 3: Exploratory Data Analysis ---")
        # Load processed if available
        df_proc = pd.read_csv(PROCESSED_DATASET_PATH) if PROCESSED_DATASET_PATH.exists() else df_engineered
        generate_eda(df_proc)
        
    # 4. Train Models
    if args.train or args.all:
        logger.info("--- Phase 4: Model Training ---")
        df_proc = pd.read_csv(PROCESSED_DATASET_PATH)
        
        df_scaled, scaled_features = scale_features(df_proc)
        
        best_k = evaluate_k(df_scaled)
        cluster_labels, kmeans_model = train_kmeans(df_scaled, best_k)
        
        visualize_clusters_pca(df_scaled, cluster_labels, df_proc)
        profile_clusters(df_proc, cluster_labels)
        
        logger.info("--- Phase 5: Recommendation Engine ---")
        build_recommendation_system(df_proc, df_scaled, cluster_labels)
        
        logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
