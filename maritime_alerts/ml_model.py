"""
ml_model.py
===========
Space-to-Sea Hackathon — Maritime Alerts: ML Anomaly Layer

What the ML adds on top of rule-based alerts
─────────────────────────────────────────────
Rules catch KNOWN violations (polygon cross, proximity breach).
This model catches BEHAVIOURAL ANOMALIES that rules miss:
  • Loitering near the zone boundary without crossing it
  • Repeated passes at decreasing distances (casing behaviour)
  • Unusual approach angle or deceleration pattern
  • Erratic heading changes near the asset

Approach
────────
1. Generate a large synthetic training corpus (1000+ tracks)
   with four labelled behaviour classes:
     0 = NORMAL_TRANSIT     — vessel passes through, steady course
     1 = LOITERING          — slow circles near but outside zone
     2 = REPEATED_APPROACH  — multiple passes, converging on asset
     3 = ERRATIC_APPROACH   — sharp turns, decelerations near asset

2. Extract per-track features (no raw lat/lon — model-agnostic geometry)
     • time_in_zone_frac     — fraction of track points inside watch zone
     • min_dist_to_asset_km  — closest approach to asset
     • mean_speed_proxy      — avg step distance per timestep
     • speed_variance        — how variable is the speed (loitering = high)
     • heading_change_mean   — avg absolute bearing change between steps
     • heading_change_max    — sharpest single turn
     • n_zone_entries        — how many times vessel enters/exits zone
     • closest_approach_rate — is the vessel getting closer over time?
     • time_near_asset_frac  — fraction of time within 5 km of asset

3. Train a RandomForestClassifier (CPU, <60 seconds locally)
   → outputs class label + per-class probability score
   → suspicious_score = P(LOITERING) + P(REPEATED_APPROACH) + P(ERRATIC_APPROACH)

4. Any track with suspicious_score > 0.5 emits a BEHAVIOURAL_ANOMALY alert
   → this is the alert type rules could never produce

Training: local laptop CPU, ~1-2 minutes
No GPU needed. No Kaggle needed for this model.
"""

import json, math, uuid, random, os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    from shapely.geometry import Point, Polygon
    SHAPELY_OK = True
except ImportError:
    SHAPELY_OK = False

try:
    from geopy.distance import geodesic
    GEOPY_OK = True
except ImportError:
    GEOPY_OK = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import LabelEncoder
    import joblib
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[ERROR] scikit-learn not found. Run: uv add scikit-learn joblib")
    raise SystemExit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# ZONE / ASSET (keep in sync with maritime_alerts.py)
# ═══════════════════════════════════════════════════════════════════════════════

WATCH_AREA_RING = [   # (lon, lat)
    [78.80, 8.80], [79.30, 8.80], [79.30, 9.20],
    [78.80, 9.20], [78.80, 8.80],
]
ASSET_LAT, ASSET_LON = 9.00, 79.05
PROXIMITY_KM = 2.0

if SHAPELY_OK:
    WATCH_POLY = Polygon([(c[0], c[1]) for c in WATCH_AREA_RING])

# ═══════════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def dist_km(lat1, lon1, lat2, lon2):
    if GEOPY_OK:
        return geodesic((lat1, lon1), (lat2, lon2)).km
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def bearing(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 → point 2, degrees [0, 360)."""
    dlon = math.radians(lon2 - lon1)
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r)*math.sin(lat2r) - math.sin(lat1r)*math.cos(lat2r)*math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def in_zone(lat, lon):
    if SHAPELY_OK:
        return WATCH_POLY.contains(Point(lon, lat))
    # ray-casting fallback
    ring = WATCH_AREA_RING
    x, y, inside = lon, lat, False
    n = len(ring) - 1
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]; xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and (x < (xj-xi)*(y-yi)/(yj-yi)+xi):
            inside = not inside
        j = i
    return inside

# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC TRACK GENERATOR — 4 behaviour classes
# ═══════════════════════════════════════════════════════════════════════════════

rng = random.Random(42)

# Zone bounding box
LON_MIN, LON_MAX = 78.60, 79.60
LAT_MIN, LAT_MAX = 8.50, 9.50
# Zone centre
ZONE_CX, ZONE_CY = (78.80 + 79.30)/2, (8.80 + 9.20)/2  # 79.05, 9.00

def _jitter(val, scale=0.01):
    return val + rng.gauss(0, scale)

def gen_normal_transit():
    """Straight-ish transit well clear of the zone — no alerts expected."""
    # pick a random entry/exit pair outside zone
    choices = [
        # west → east, south of zone
        {"start": (rng.uniform(8.50, 8.75), 78.60),
         "end":   (rng.uniform(8.50, 8.75), 79.55)},
        # west → east, north of zone
        {"start": (rng.uniform(9.25, 9.50), 78.60),
         "end":   (rng.uniform(9.25, 9.50), 79.55)},
        # south → north, east of zone
        {"start": (8.55, rng.uniform(79.35, 79.55)),
         "end":   (9.45, rng.uniform(79.35, 79.55))},
    ]
    c = rng.choice(choices)
    slat, slon = c["start"]
    elat, elon = c["end"]
    n = rng.randint(10, 18)
    pts = []
    for i in range(n):
        t = i / (n-1)
        pts.append((_jitter(slat*(1-t)+elat*t, 0.005),
                    _jitter(slon*(1-t)+elon*t, 0.005)))
    return pts

def gen_loitering():
    """Slow circles near zone boundary without crossing — the classic shadow."""
    # anchor point just outside zone
    anchors = [
        (8.75, 79.05),   # just south of S edge
        (9.05, 78.72),   # just west of W edge
        (9.25, 79.05),   # just north of N edge
        (9.05, 79.38),   # just east of E edge
    ]
    ax, ay = rng.choice(anchors)
    n = rng.randint(14, 24)
    radius = rng.uniform(0.03, 0.07)   # ~3-7 km circle
    pts = []
    for i in range(n):
        angle = 2 * math.pi * i / n + rng.gauss(0, 0.1)
        lat = ax + radius * math.sin(angle) + rng.gauss(0, 0.005)
        lon = ay + radius * math.cos(angle) + rng.gauss(0, 0.005)
        pts.append((lat, lon))
    return pts

def gen_repeated_approach():
    """Multiple passes, each getting closer to the asset — 'casing' pattern."""
    pts = []
    n_passes = rng.randint(2, 4)
    for p in range(n_passes):
        shrink = 1.0 - p * 0.25   # each pass gets closer
        # approach from random bearing
        bearing_angle = rng.uniform(0, 2*math.pi)
        d_start = rng.uniform(0.25, 0.40) * shrink
        d_end   = rng.uniform(0.03, 0.10) * shrink
        n = rng.randint(6, 10)
        for i in range(n):
            t = i / (n-1)
            d = d_start*(1-t) + d_end*t
            lat = ASSET_LAT + d * math.sin(bearing_angle) + rng.gauss(0, 0.004)
            lon = ASSET_LON + d * math.cos(bearing_angle) + rng.gauss(0, 0.004)
            pts.append((lat, lon))
        # retreat slightly before next pass
        for i in range(4):
            lat = ASSET_LAT + (d_end + 0.05*i) * math.sin(bearing_angle + 0.5)
            lon = ASSET_LON + (d_end + 0.05*i) * math.cos(bearing_angle + 0.5)
            pts.append((_jitter(lat), _jitter(lon)))
    return pts

def gen_erratic_approach():
    """Sharp turns and speed changes while closing in on the asset."""
    pts = []
    lat = rng.uniform(8.70, 9.30)
    lon = rng.uniform(78.70, 79.30)
    target_lat, target_lon = ASSET_LAT, ASSET_LON
    n = rng.randint(16, 26)
    for i in range(n):
        # mostly head toward asset with random sharp deviations
        progress = i / n
        base_dlat = (target_lat - lat) * 0.15
        base_dlon = (target_lon - lon) * 0.15
        # inject erratic jitter — high variance heading changes
        if rng.random() < 0.3:   # 30% chance of a sharp deviation
            deflect = rng.uniform(-0.08, 0.08)
            base_dlat, base_dlon = base_dlat + deflect, base_dlon - deflect
        # slow down as it nears (deceleration = shrinking step)
        scale = max(0.3, 1.0 - progress * 0.7)
        lat = lat + base_dlat*scale + rng.gauss(0, 0.003)
        lon = lon + base_dlon*scale + rng.gauss(0, 0.003)
        pts.append((lat, lon))
    return pts

GENERATORS = {
    "NORMAL_TRANSIT":    gen_normal_transit,
    "LOITERING":         gen_loitering,
    "REPEATED_APPROACH": gen_repeated_approach,
    "ERRATIC_APPROACH":  gen_erratic_approach,
}

def build_training_corpus(n_per_class=300):
    """Generate n_per_class synthetic tracks per behaviour label."""
    rows = []
    for label, gen_fn in GENERATORS.items():
        for _ in range(n_per_class):
            pts = gen_fn()
            rows.append({"label": label, "points": pts})
    rng.shuffle(rows)
    print(f"[DATA] Generated {len(rows)} synthetic tracks "
          f"({n_per_class} per class × {len(GENERATORS)} classes)")
    return rows

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — per track
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features(points):
    """
    Convert a list of (lat, lon) waypoints into a fixed-length feature vector.
    All features are geometry-based — no raw coordinates fed to the model.
    """
    n = len(points)
    if n < 2:
        return None

    dists_to_asset = [dist_km(lat, lon, ASSET_LAT, ASSET_LON) for lat, lon in points]
    in_zone_flags  = [1 if in_zone(lat, lon) else 0 for lat, lon in points]

    # step distances (speed proxy — no actual timestamps needed for shape features)
    step_dists = [
        dist_km(points[i][0], points[i][1], points[i-1][0], points[i-1][1])
        for i in range(1, n)
    ]

    # bearing changes
    bearings = [
        bearing(points[i-1][0], points[i-1][1], points[i][0], points[i][1])
        for i in range(1, n)
    ]
    bearing_deltas = [
        abs((bearings[i] - bearings[i-1] + 180) % 360 - 180)
        for i in range(1, len(bearings))
    ]

    # zone entry count (transitions 0→1)
    zone_entries = sum(
        1 for i in range(1, n) if in_zone_flags[i] == 1 and in_zone_flags[i-1] == 0
    )

    # closest approach rate: slope of dist_to_asset over time
    # negative = getting closer on average
    x = np.arange(len(dists_to_asset))
    slope = float(np.polyfit(x, dists_to_asset, 1)[0]) if len(x) > 1 else 0.0

    return {
        "time_in_zone_frac":     sum(in_zone_flags) / n,
        "min_dist_to_asset_km":  min(dists_to_asset),
        "mean_dist_to_asset_km": float(np.mean(dists_to_asset)),
        "time_near_asset_frac":  sum(1 for d in dists_to_asset if d < 5.0) / n,
        "mean_speed_proxy":      float(np.mean(step_dists)),
        "speed_variance":        float(np.var(step_dists)),
        "heading_change_mean":   float(np.mean(bearing_deltas)) if bearing_deltas else 0.0,
        "heading_change_max":    float(np.max(bearing_deltas))  if bearing_deltas else 0.0,
        "n_zone_entries":        zone_entries,
        "closest_approach_rate": slope,   # negative = converging on asset
        "track_length_km":       sum(step_dists),
        "n_points":              n,
    }

FEATURE_COLS = [
    "time_in_zone_frac", "min_dist_to_asset_km", "mean_dist_to_asset_km",
    "time_near_asset_frac", "mean_speed_proxy", "speed_variance",
    "heading_change_mean", "heading_change_max", "n_zone_entries",
    "closest_approach_rate", "track_length_km", "n_points",
]

# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN
# ═══════════════════════════════════════════════════════════════════════════════

def train(n_per_class=300, model_path="output/maritime_rf_model.joblib"):
    corpus = build_training_corpus(n_per_class)

    records = []
    for item in corpus:
        feats = extract_features(item["points"])
        if feats:
            feats["label"] = item["label"]
            records.append(feats)

    df = pd.DataFrame(records)
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = np.array(df["label"].tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("[TRAIN] Fitting RandomForestClassifier ...")
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,      # use all CPU cores
    )
    clf.fit(X_train, y_train)

    print("\n[EVAL] Classification report on held-out 20%:")
    print(classification_report(y_test, clf.predict(X_test), digits=3))

    # Feature importance
    fi = pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("[FEAT] Top feature importances:")
    print(fi.to_string())

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"model": clf, "feature_cols": FEATURE_COLS}, model_path)
    print(f"\n[OK]  Model saved → {model_path}")
    return clf

# ═══════════════════════════════════════════════════════════════════════════════
# SCORE / PREDICT — apply to new tracks
# ═══════════════════════════════════════════════════════════════════════════════

SUSPICIOUS_CLASSES = {"LOITERING", "REPEATED_APPROACH", "ERRATIC_APPROACH"}

def score_tracks(tracks_data, model_path="output/maritime_rf_model.joblib"):
    """
    tracks_data: list of dicts with keys track_id, lat, lon, timestamp
    Returns: list of alert dicts for BEHAVIOURAL_ANOMALY detections
    """
    bundle = joblib.load(model_path)
    clf    = bundle["model"]
    feat_cols = bundle["feature_cols"]

    # group by track_id
    df = pd.DataFrame(tracks_data).sort_values(["track_id", "timestamp"])
    alerts = []

    for track_id, grp in df.groupby("track_id"):
        points = list(zip(grp["lat"], grp["lon"]))
        feats  = extract_features(points)
        if feats is None:
            continue

        X = np.array([[feats[c] for c in feat_cols]])
        proba  = clf.predict_proba(X)[0]
        classes = clf.classes_
        prob_map = dict(zip(classes, proba))

        pred_class = classes[np.argmax(proba)]
        suspicious_score = sum(prob_map.get(c, 0) for c in SUSPICIOUS_CLASSES)

        print(f"  {track_id}: predicted={pred_class}  "
              f"suspicious_score={suspicious_score:.2f}  "
              f"probs={prob_map}")

        if suspicious_score > 0.5:
            # find the point where vessel was closest to the asset for lat/lon
            dists = [dist_km(lat, lon, ASSET_LAT, ASSET_LON)
                     for lat, lon in points]
            closest_idx = int(np.argmin(dists))
            closest_pt  = points[closest_idx]

            alerts.append({
                "alert_id":          f"ALT-{uuid.uuid4().hex[:8].upper()}",
                "track_id":          str(track_id),
                "alert_type":        "BEHAVIOURAL_ANOMALY",
                "predicted_class":   str(pred_class),
                "suspicious_score":  round(float(suspicious_score), 3),
                "class_proba":       {str(k): round(float(v), 3) for k, v in prob_map.items()},
                "lat":               float(closest_pt[0]),
                "lon":               float(closest_pt[1]),
                "timestamp":         str(grp["timestamp"].iloc[closest_idx]),
                "zone_id":           "ZONE-TN-GULF-01",
                "asset_id":          "ASSET-TN-SUB-01",
                "data_source":       "SIMULATED",
            })

    return alerts

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — train then score the demo tracks
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from maritime_alerts import build_simulated_tracks, write_alerts_json

    print("\n━━━ Space-to-Sea | ML Anomaly Layer ━━━")
    print("    Model   : RandomForestClassifier (scikit-learn)")
    print("    Training: Synthetic corpus — SIMULATED\n")

    # 1. Train
    clf = train(n_per_class=300, model_path="output/maritime_rf_model.joblib")

    # 2. Score the 3 demo tracks
    print("\n[SCORE] Scoring the 3 demo tracks ...\n")
    demo_tracks = build_simulated_tracks()
    ml_alerts   = score_tracks(demo_tracks, model_path="output/maritime_rf_model.joblib")

    # 3. Merge with rule-based alerts and save
    rule_alerts_path = "output/alerts.json"
    if os.path.exists(rule_alerts_path):
        with open(rule_alerts_path) as f:
            existing = json.load(f)["alerts"]
    else:
        existing = []

    all_alerts = existing + ml_alerts
    write_alerts_json(all_alerts, path="output/alerts.json")

    print(f"\n[DONE] {len(ml_alerts)} BEHAVIOURAL_ANOMALY alerts added.")
    print("       All alerts → output/alerts.json\n")

if __name__ == "__main__":
    main()
