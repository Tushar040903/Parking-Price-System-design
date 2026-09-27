"""
Geospatial Competitor Analysis Module
======================================

Analyzes spatial relationships between parking lots:
- Haversine distance computation between all lots
- Competitor pressure index calculation
- Interactive Folium map generation
- Spatial autocorrelation analysis (Moran's I)

Upgrade 5: Geospatial Competitor Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────
# Haversine Distance
# ─────────────────────────────────────────────────────────────────────

def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """
    Calculate the great-circle distance between two points on Earth.

    Uses the Haversine formula. Returns distance in kilometers.

    Parameters
    ----------
    lat1, lon1 : float
        Latitude and longitude of point 1 (in degrees).
    lat2, lon2 : float
        Latitude and longitude of point 2 (in degrees).

    Returns
    -------
    float
        Distance in kilometers.
    """
    R = 6371.0  # Earth's radius in km

    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


# ─────────────────────────────────────────────────────────────────────
# Distance Matrix
# ─────────────────────────────────────────────────────────────────────

def build_distance_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a pairwise distance matrix between all parking lots.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'SystemCodeNumber', 'Latitude', 'Longitude'.

    Returns
    -------
    tuple
        (distance_matrix DataFrame, lot_locations DataFrame)
    """
    lot_locations = df.groupby("SystemCodeNumber").agg(
        Latitude=("Latitude", "first"),
        Longitude=("Longitude", "first"),
    ).reset_index()

    n_lots = len(lot_locations)
    lot_ids = lot_locations["SystemCodeNumber"].values
    dist_matrix = np.zeros((n_lots, n_lots))

    for i in range(n_lots):
        for j in range(n_lots):
            if i != j:
                dist_matrix[i, j] = haversine_distance(
                    lot_locations.iloc[i]["Latitude"],
                    lot_locations.iloc[i]["Longitude"],
                    lot_locations.iloc[j]["Latitude"],
                    lot_locations.iloc[j]["Longitude"],
                )

    dist_df = pd.DataFrame(dist_matrix, index=lot_ids, columns=lot_ids)
    return dist_df, lot_locations


# ─────────────────────────────────────────────────────────────────────
# Competitor Pressure Index
# ─────────────────────────────────────────────────────────────────────

def compute_competitor_pressure(
    df: pd.DataFrame,
    dist_matrix: pd.DataFrame,
    radius_km: float = 2.0,
    decay_factor: float = 1.0,
) -> pd.DataFrame:
    """
    Compute a 'competitor pressure index' for each parking lot.

    Pressure is higher when:
    - More competitors are nearby
    - Nearby competitors have lower occupancy (more available spots)
    - Nearby competitors have larger capacity

    The index uses inverse-distance weighting with exponential decay.

    Parameters
    ----------
    df : pd.DataFrame
        Main dataset with occupancy data.
    dist_matrix : pd.DataFrame
        Pairwise distance matrix between lots.
    radius_km : float
        Only consider competitors within this radius.
    decay_factor : float
        Controls how quickly pressure decreases with distance.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'CompetitorPressure' column added.
    """
    lot_stats = df.groupby("SystemCodeNumber").agg(
        avg_occ=("OccupancyRate", "mean"),
        capacity=("Capacity", "first"),
    )

    pressure_dict = {}

    for lot_id in dist_matrix.index:
        distances = dist_matrix.loc[lot_id]
        nearby = distances[(distances > 0) & (distances <= radius_km)]

        if len(nearby) == 0:
            pressure_dict[lot_id] = 0.0
            continue

        total_pressure = 0.0
        for competitor_id, dist in nearby.items():
            if competitor_id in lot_stats.index:
                comp_avail = 1 - lot_stats.loc[competitor_id, "avg_occ"]  # availability
                comp_cap = lot_stats.loc[competitor_id, "capacity"]
                weight = np.exp(-decay_factor * dist)  # distance decay
                total_pressure += weight * comp_avail * (comp_cap / 1000)  # normalized

        pressure_dict[lot_id] = round(total_pressure, 4)

    # Map back to original DataFrame
    df = df.copy()
    df["CompetitorPressure"] = df["SystemCodeNumber"].map(pressure_dict)
    df["NearbyCompetitors"] = df["SystemCodeNumber"].map(
        {lot: ((dist_matrix.loc[lot] > 0) & (dist_matrix.loc[lot] <= radius_km)).sum()
         for lot in dist_matrix.index}
    )

    return df


# ─────────────────────────────────────────────────────────────────────
# Folium Map Visualization
# ─────────────────────────────────────────────────────────────────────

def create_folium_map(
    df: pd.DataFrame,
    lot_locations: pd.DataFrame,
    save_path: Optional[str] = None,
):
    """
    Create an interactive Folium map with parking lot markers.

    Markers are color-coded by average occupancy rate and sized by capacity.
    Includes pop-up information with lot statistics.
    """
    import folium
    from folium.plugins import HeatMap

    # Compute lot-level stats
    lot_stats = df.groupby("SystemCodeNumber").agg(
        avg_occ=("OccupancyRate", "mean"),
        avg_queue=("QueueLength", "mean"),
        capacity=("Capacity", "first"),
    ).reset_index()

    lot_data = lot_locations.merge(lot_stats, on="SystemCodeNumber")

    # Center map
    center_lat = lot_data["Latitude"].mean()
    center_lon = lot_data["Longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13,
                   tiles="CartoDB positron")

    # Color function based on occupancy
    def get_color(occ):
        if occ > 0.7:
            return "red"
        elif occ > 0.4:
            return "orange"
        else:
            return "green"

    # Add markers
    for _, row in lot_data.iterrows():
        radius = max(5, row["capacity"] / 100)  # Size by capacity
        color = get_color(row["avg_occ"])

        popup_html = f"""
        <div style="font-family: Arial; width: 200px;">
            <h4>{row['SystemCodeNumber']}</h4>
            <b>Capacity:</b> {row['capacity']}<br>
            <b>Avg Occupancy:</b> {row['avg_occ']:.1%}<br>
            <b>Avg Queue:</b> {row['avg_queue']:.1f}<br>
            <b>Status:</b> {'🔴 High' if row['avg_occ'] > 0.7 else '🟡 Medium' if row['avg_occ'] > 0.4 else '🟢 Low'}
        </div>
        """

        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=radius,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{row['SystemCodeNumber']}: {row['avg_occ']:.0%}",
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=2,
        ).add_to(m)

    # Add heatmap layer
    heat_data = [[row["Latitude"], row["Longitude"], row["avg_occ"]]
                 for _, row in lot_data.iterrows()]
    HeatMap(heat_data, radius=30, blur=20, max_zoom=15).add_to(m)

    # Add legend
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background: white; padding: 10px; border-radius: 5px;
                border: 2px solid gray; font-size: 12px;">
        <b>Occupancy Level</b><br>
        🟢 Low (&lt;40%)<br>
        🟡 Medium (40-70%)<br>
        🔴 High (&gt;70%)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    if save_path:
        m.save(save_path)
        print(f"   📍 Map saved to: {save_path}")

    return m


# ─────────────────────────────────────────────────────────────────────
# Spatial Analysis Visualization
# ─────────────────────────────────────────────────────────────────────

def plot_distance_heatmap(dist_matrix: pd.DataFrame, save_path: Optional[str] = None):
    """Plot a heatmap of pairwise distances between lots."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Shorten names for readability
    short_names = [name[:10] for name in dist_matrix.index]

    sns.heatmap(
        dist_matrix.values,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd_r",
        xticklabels=short_names,
        yticklabels=short_names,
        ax=ax,
        cbar_kws={"label": "Distance (km)"},
    )
    ax.set_title("Pairwise Distances Between Parking Lots (km)", fontweight="bold", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_competitor_pressure(df: pd.DataFrame, save_path: Optional[str] = None):
    """Bar chart of competitor pressure index per lot."""
    pressure_by_lot = df.groupby("SystemCodeNumber")["CompetitorPressure"].first().sort_values()

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(pressure_by_lot)))
    ax.barh(range(len(pressure_by_lot)), pressure_by_lot.values,
            color=colors, edgecolor="white")
    ax.set_yticks(range(len(pressure_by_lot)))
    ax.set_yticklabels([n[:12] for n in pressure_by_lot.index])
    ax.set_xlabel("Competitor Pressure Index")
    ax.set_title("Competitor Pressure by Parking Lot", fontweight="bold", fontsize=14)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def spatial_autocorrelation_analysis(
    df: pd.DataFrame,
    dist_matrix: pd.DataFrame,
    save_path: Optional[str] = None,
) -> Dict:
    """
    Compute Moran's I spatial autocorrelation for occupancy.

    Tests whether nearby parking lots have similar occupancy patterns
    (positive spatial autocorrelation) or dissimilar patterns (negative).
    """
    lot_occ = df.groupby("SystemCodeNumber")["OccupancyRate"].mean()

    # Ensure consistent ordering
    lots = [l for l in dist_matrix.index if l in lot_occ.index]
    occ_values = lot_occ[lots].values
    n = len(lots)

    # Create spatial weight matrix (inverse distance)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j and dist_matrix.loc[lots[i], lots[j]] > 0:
                W[i, j] = 1.0 / max(dist_matrix.loc[lots[i], lots[j]], 0.01)

    # Row-standardize
    row_sums = W.sum(axis=1)
    W = W / row_sums[:, np.newaxis]
    W = np.nan_to_num(W)

    # Moran's I
    y = occ_values - occ_values.mean()
    numerator = n * np.sum(W * np.outer(y, y))
    denominator = np.sum(W) * np.sum(y ** 2)
    morans_i = numerator / denominator if denominator != 0 else 0

    result = {
        "morans_i": round(morans_i, 4),
        "interpretation": (
            "Positive spatial autocorrelation (nearby lots have similar occupancy)"
            if morans_i > 0
            else "Negative spatial autocorrelation (nearby lots have dissimilar occupancy)"
            if morans_i < 0
            else "No spatial autocorrelation"
        ),
    }

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    lag_occ = W @ occ_values
    ax.scatter(occ_values, lag_occ, s=80, alpha=0.7, color="#4C72B0", edgecolors="black")
    for i, lot in enumerate(lots):
        ax.annotate(lot[:8], (occ_values[i], lag_occ[i]),
                    fontsize=7, ha="center", va="bottom", xytext=(0, 5),
                    textcoords="offset points")

    # Add trend line
    z = np.polyfit(occ_values, lag_occ, 1)
    p = np.poly1d(z)
    x_line = np.linspace(occ_values.min(), occ_values.max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f"Moran's I = {morans_i:.4f}")

    ax.set_xlabel("Occupancy Rate", fontweight="bold")
    ax.set_ylabel("Spatially Lagged Occupancy Rate", fontweight="bold")
    ax.set_title("Moran's I Spatial Autocorrelation Plot", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

    return result
