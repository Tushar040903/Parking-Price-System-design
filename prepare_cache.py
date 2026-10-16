"""
Precompute and save pipeline artifacts for instant Streamlit startup.
"""

import os
import sys
import time
import joblib
import pandas as pd
import numpy as np

# Import ParkWise modules
from src.data_preprocessing import full_preprocessing_pipeline, get_ml_features
from src.eda import run_statistical_tests
from src.demand_forecasting import compare_forecasting_models
from src.pricing_model import prepare_pricing_data, train_all_models
from src.clustering import prepare_lot_profiles, run_kmeans, profile_clusters
from src.geospatial import build_distance_matrix, compute_competitor_pressure
from src.elasticity import elasticity_by_lot, estimate_demand_curve
from src.ab_testing import (
    static_pricing,
    linear_pricing,
    demand_based_pricing,
    competitive_pricing,
    simulate_strategy,
    simulate_ml_strategy,
    compare_strategies,
    run_all_significance_tests,
)
from src.explainability import get_feature_importance

def build_cache():
    print("=" * 60)
    print("  Building ParkWise Pipeline Cache for Streamlit...")
    print("=" * 60)
    t_start = time.time()
    
    os.makedirs("outputs", exist_ok=True)
    cache_path = os.path.join("outputs", "pipeline_cache.joblib")
    
    # 1. Preprocessing
    print("\n[1/8] Preprocessing dataset...")
    df = full_preprocessing_pipeline("dataset.csv")
    
    # 2. Geospatial competitor pressure
    print("[2/8] Geospatial distance matrix & competitor pressure...")
    dist_matrix, lot_locations = build_distance_matrix(df)
    df = compute_competitor_pressure(df, dist_matrix)
    
    # 3. Statistical tests
    print("[3/8] Statistical hypothesis testing...")
    stat_tests = run_statistical_tests(df)
    
    # 4. Pricing models
    print("[4/8] Training ML pricing models...")
    X_train, X_test, y_train, y_test, scaler, feat_cols = prepare_pricing_data(df)
    pricing_results = train_all_models(X_train, X_test, y_train, y_test)
    best_pricing_model = pricing_results["XGBoost"]["model"]
    
    # 5. Clustering
    print("[5/8] Unsupervised clustering...")
    profiles = prepare_lot_profiles(df)
    cluster_features = ["occ_mean", "occ_max", "queue_mean", "traffic_mean"]
    clustered_df, kmeans_model, scaler_k = run_kmeans(profiles, feature_cols=cluster_features, n_clusters=3)
    c_profiles = profile_clusters(clustered_df, feature_cols=cluster_features)
    
    # 6. Elasticity
    print("[6/8] Econometric elasticity...")
    elast_lots = elasticity_by_lot(df)
    d_curve = estimate_demand_curve(df)
    
    # 7. A/B Testing
    print("[7/8] Simulating pricing strategies...")
    all_results = {}
    all_results["Static"] = simulate_strategy(df, static_pricing, "Static")
    all_results["Linear"] = simulate_strategy(df, linear_pricing, "Linear")
    all_results["Demand-Based"] = simulate_strategy(df, demand_based_pricing, "Demand-Based")
    all_results["Competitive"] = simulate_strategy(df, competitive_pricing, "Competitive")
    all_results["ML-Based"] = simulate_ml_strategy(df, best_pricing_model, scaler, feat_cols, "ML-Based")
    comp_df = compare_strategies(all_results)
    sig_df = run_all_significance_tests(all_results)
    
    # 8. Feature Importance
    print("[8/8] Feature importance & time-series forecast...")
    imp_df = get_feature_importance(best_pricing_model, feat_cols)
    top_lot = df["SystemCodeNumber"].value_counts().index[0]
    forecast_results = compare_forecasting_models(df, lot_id=top_lot, test_size=48)
    
    payload = {
        "df": df,
        "dist_matrix": dist_matrix,
        "lot_locations": lot_locations,
        "stat_tests": stat_tests,
        "pricing_results": pricing_results,
        "best_pricing_model": best_pricing_model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feat_cols": feat_cols,
        "profiles": profiles,
        "clustered_df": clustered_df,
        "kmeans_model": kmeans_model,
        "c_profiles": c_profiles,
        "cluster_features": cluster_features,
        "elast_lots": elast_lots,
        "d_curve": d_curve,
        "all_results": all_results,
        "comp_df": comp_df,
        "sig_df": sig_df,
        "imp_df": imp_df,
        "forecast_results": forecast_results,
        "top_lot": top_lot,
        "created_at": time.time(),
    }
    
    joblib.dump(payload, cache_path, compress=3)
    print(f"\n Cache successfully saved to: {cache_path}")
    print(f"   Size: {os.path.getsize(cache_path) / (1024*1024):.2f} MB")
    print(f"   Total generation time: {time.time() - t_start:.2f} seconds")

if __name__ == "__main__":
    build_cache()
