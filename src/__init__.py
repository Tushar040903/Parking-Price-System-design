"""
ParkWise: ML-Driven Dynamic Pricing for Urban Parking
=====================================================

Modular source package containing all ML/DS components:
- data_preprocessing: Data loading, cleaning, and feature engineering
- eda: Exploratory data analysis and statistical testing
- demand_forecasting: Time-series forecasting (ARIMA, XGBoost)
- pricing_model: ML-based pricing (Linear, Ridge, RF, XGBoost)
- clustering: Demand segmentation (K-Means, DBSCAN)
- geospatial: Spatial competitor analysis (Haversine, Folium)
- elasticity: Price elasticity estimation
- ab_testing: A/B testing and simulation framework
- explainability: SHAP values, feature importance, PDP
"""

import sys

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

__version__ = "2.0.0"
__author__ = "Tushar"
