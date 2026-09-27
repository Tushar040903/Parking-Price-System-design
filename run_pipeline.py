"""
ParkWise: End-to-End Pipeline Execution Script
================================================
Executes all 8 ML/DS upgrades sequentially on dataset.csv
and saves artifacts to the outputs/ directory.
"""

import os
import sys

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt

# Import ParkWise modules
from src.data_preprocessing import full_preprocessing_pipeline, get_ml_features
from src.eda import (
    plot_occupancy_distribution,
    plot_correlation_heatmap,
    run_statistical_tests,
)
from src.demand_forecasting import (
    compare_forecasting_models,
    plot_forecast_comparison,
    plot_metrics_comparison,
)
from src.pricing_model import (
    prepare_pricing_data,
    train_all_models,
    plot_model_comparison,
    plot_actual_vs_predicted,
)
from src.clustering import (
    prepare_lot_profiles,
    run_kmeans,
    plot_cluster_scatter,
    profile_clusters,
    plot_cluster_profiles_radar,
)
from src.geospatial import (
    build_distance_matrix,
    compute_competitor_pressure,
    create_folium_map,
)
from src.elasticity import (
    elasticity_by_lot,
    plot_elasticity_by_lot,
    estimate_demand_curve,
    plot_demand_curve,
)
from src.ab_testing import (
    static_pricing,
    linear_pricing,
    demand_based_pricing,
    competitive_pricing,
    simulate_strategy,
    simulate_ml_strategy,
    compare_strategies,
    plot_strategy_comparison,
    run_all_significance_tests,
)
from src.explainability import (
    get_feature_importance,
    plot_feature_importance,
    compute_shap_values,
    plot_shap_summary,
)


def main():
    print("=" * 75)
    print("  PARKWISE: ML-DRIVEN DYNAMIC PRICING ENGINE - PIPELINE RUNNER")
    print("=" * 75)

    dataset_path = "dataset.csv"
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 1: Preprocessing & Feature Engineering
    # -------------------------------------------------------------------------
    print("\n[1/8] Preprocessing Data & Feature Engineering...")
    df = full_preprocessing_pipeline(dataset_path)
    print(f"      Total records loaded: {len(df):,}")
    print(f"      Total features: {df.shape[1]}")
    print(f"      Parking lots: {df['SystemCodeNumber'].nunique()}")

    # -------------------------------------------------------------------------
    # Step 2: Exploratory Data Analysis & Statistical Tests
    # -------------------------------------------------------------------------
    print("\n[2/8] Running Deep EDA & Statistical Hypothesis Testing...")
    plot_occupancy_distribution(df, save_path=os.path.join(output_dir, "01_eda_occupancy_distribution.png"))
    plot_correlation_heatmap(df, save_path=os.path.join(output_dir, "01_eda_correlation_matrix.png"))

    stat_tests = run_statistical_tests(df)
    t_test = stat_tests["t_test_special_day"]
    anova = stat_tests["anova_across_lots"]
    print(f"      Special Day vs Normal Day t-test p-value: {t_test['p_value']:.4e} ({'Significant' if t_test['significant'] else 'Not Significant'})")
    print(f"      ANOVA across lots p-value: {anova['p_value']:.4e} ({'Significant' if anova['significant'] else 'Not Significant'})")

    # -------------------------------------------------------------------------
    # Step 3: Time-Series Demand Forecasting
    # -------------------------------------------------------------------------
    print("\n[3/8] Training Time-Series Demand Forecasting Models...")
    top_lot = df["SystemCodeNumber"].value_counts().index[0]
    forecast_results = compare_forecasting_models(df, lot_id=top_lot, test_size=48)

    plot_forecast_comparison(forecast_results, lot_id=top_lot,
                             save_path=os.path.join(output_dir, "02_demand_forecasting_comparison.png"))
    plot_metrics_comparison(forecast_results,
                            save_path=os.path.join(output_dir, "02_demand_forecasting_metrics.png"))

    # -------------------------------------------------------------------------
    # Step 4: ML-Based Dynamic Pricing Model
    # -------------------------------------------------------------------------
    print("\n[4/8] Benchmarking Supervised ML Pricing Models...")
    X_train, X_test, y_train, y_test, scaler, feat_cols = prepare_pricing_data(df)
    pricing_results = train_all_models(X_train, X_test, y_train, y_test)
    metrics_summary = pd.DataFrame({
        name: res["metrics"] for name, res in pricing_results.items()
    }).T
    print(metrics_summary.to_string())
    plot_model_comparison(pricing_results, save_path=os.path.join(output_dir, "03_pricing_models_benchmark.png"))
    plot_actual_vs_predicted(y_test, pricing_results, save_path=os.path.join(output_dir, "03_pricing_actual_vs_predicted.png"))
    best_pricing_model = pricing_results["XGBoost"]["model"]

    # -------------------------------------------------------------------------
    # Step 5: Demand Segmentation (Clustering)
    # -------------------------------------------------------------------------
    print("\n[5/8] Unsupervised Demand Segmentation & Lot Clustering...")
    profiles = prepare_lot_profiles(df)
    cluster_features = ["occ_mean", "occ_max", "queue_mean", "traffic_mean"]
    clustered_df, kmeans_model, scaler_k = run_kmeans(profiles, feature_cols=cluster_features, n_clusters=3)
    plot_cluster_scatter(clustered_df, x_col="occ_mean", y_col="queue_mean",
                        cluster_col="Cluster", label_col="SystemCodeNumber",
                        save_path=os.path.join(output_dir, "04_demand_clusters.png"))

    c_profiles = profile_clusters(clustered_df, feature_cols=cluster_features)
    plot_cluster_profiles_radar(c_profiles, feature_cols=cluster_features,
                               save_path=os.path.join(output_dir, "04_cluster_radar.png"))
    print(f"      Clustered {len(clustered_df)} lots into 3 behavioral demand tiers.")

    # -------------------------------------------------------------------------
    # Step 6: Geospatial Competitor Analysis
    # -------------------------------------------------------------------------
    print("\n[6/8] Geospatial Competitor Pressure & Interactive Mapping...")
    dist_matrix, lot_locations = build_distance_matrix(df)
    df = compute_competitor_pressure(df, dist_matrix)
    map_path = os.path.join(output_dir, "05_geospatial_parking_map.html")
    m = create_folium_map(df, lot_locations, save_path=map_path)
    print(f"      Saved interactive Folium map to: {map_path}")

    # -------------------------------------------------------------------------
    # Step 7: Price Elasticity Estimation
    # -------------------------------------------------------------------------
    print("\n[7/8] Econometric Price Elasticity Estimation...")
    elast_lots = elasticity_by_lot(df)
    print(f"      Average elasticity across lots: {elast_lots['elasticity'].mean():.3f} (Inelastic demand)")
    plot_elasticity_by_lot(elast_lots, save_path=os.path.join(output_dir, "06_price_elasticity.png"))

    d_curve = estimate_demand_curve(df)
    plot_demand_curve(d_curve, save_path=os.path.join(output_dir, "06_demand_curve.png"))

    # -------------------------------------------------------------------------
    # Step 8: A/B Testing & Simulation
    # -------------------------------------------------------------------------
    print("\n[8/8] Simulating 5 Dynamic Pricing Strategies (A/B Testing)...")
    all_results = {}
    all_results["Static"] = simulate_strategy(df, static_pricing, "Static")
    all_results["Linear"] = simulate_strategy(df, linear_pricing, "Linear")
    all_results["Demand-Based"] = simulate_strategy(df, demand_based_pricing, "Demand-Based")
    all_results["Competitive"] = simulate_strategy(df, competitive_pricing, "Competitive")
    all_results["ML-Based"] = simulate_ml_strategy(df, best_pricing_model, scaler, feat_cols, "ML-Based")

    comp_df = compare_strategies(all_results)
    print(comp_df[["avg_price", "total_revenue", "avg_utilization", "price_cv"]].to_string())
    plot_strategy_comparison(all_results, save_path=os.path.join(output_dir, "07_ab_test_simulation.png"))

    sig_df = run_all_significance_tests(all_results)
    static_vs_ml = sig_df[sig_df["comparison"] == "Static vs ML-Based"].iloc[0]
    print(f"      Static vs ML-Based t-test p-value: {static_vs_ml['p_value']:.4e} ({'Statistically Significant' if static_vs_ml['significant'] else 'Not Significant'})")
    print(f"      ML Revenue Lift over Static: +{static_vs_ml['revenue_diff_%']:.1f}%")

    # -------------------------------------------------------------------------
    # Step 9: Feature Importance & Explainability
    # -------------------------------------------------------------------------
    print("\n[Bonus] Model Interpretability & Explainability...")
    imp_df = get_feature_importance(best_pricing_model, feat_cols)
    plot_feature_importance(imp_df, top_n=10,
                            save_path=os.path.join(output_dir, "08_feature_importance.png"))

    print("\nComputing SHAP values for sample...")
    shap_vals, explainer, X_explain = compute_shap_values(
        best_pricing_model, X_train, X_test, max_samples=200
    )
    plot_shap_summary(shap_vals, X_explain, save_path=os.path.join(output_dir, "08_shap_summary.png"))

    print("\n" + "=" * 75)
    print("  PIPELINE COMPLETE - All 8 Upgrades Executed Successfully!")
    print(f"  Artifacts saved to: {os.path.abspath(output_dir)}")
    print("=" * 75)


if __name__ == "__main__":
    main()
