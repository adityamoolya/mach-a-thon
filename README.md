# Space-to-Sea — MachAthon

**Satellite renewable siting + offshore infrastructure monitoring**
Hackathon project · 6-hour sprint · 5 parallel workstreams

---

## What This Project Does

Space-to-Sea uses satellite data to:
1. **Site offshore renewable energy** (wind/solar) using spatial analysis
2. **Monitor infrastructure** — detect vessel activity near protected assets in real time

This repo covers **Workstream 4: Maritime Alerts**, the vessel monitoring and anomaly detection module.

---

## Repo Structure

```
mach-a-thon/
│
├── maritime_alerts/              # Vessel monitoring module
│   ├── maritime_alerts.py        # Rule-based alert pipeline (run this first)
│   ├── ml_model.py               # ML anomaly detection layer (run second)
│   ├── maritime_alerts_kaggle.ipynb  # Kaggle-ready notebook (same logic)
│   ├── pyproject.toml            # uv dependency manifest
│   └── output/                   # Generated on first run — hand to dashboard team
│       ├── tracks.geojson        # Simulated vessel tracks
│       ├── alerts.json           # All alerts (rule-based + ML)
│       └── zones.geojson         # Watch-area polygon + protected asset
│
└── data/                         # Raw spatial/survey data for other workstreams
    ├── 150m_Map_Data_A_to_G.csv
    └── *.xlsx                    # Village/district-level survey sheets (Tamil Nadu)
```

---

## Maritime Alerts Module

### What it detects

| Alert Type | Method | Trigger |
|---|---|---|
| `BOUNDARY_CROSSING_OBSERVED` | Rule (geometry) | Vessel enters the watch-area polygon |
| `ASSET_PROXIMITY_ALERT` | Rule (distance) | Vessel comes within 2 km of protected asset |
| `BEHAVIOURAL_ANOMALY` | **ML model** | Suspicious movement pattern detected |

The ML layer catches what rules miss — vessels **loitering** outside the zone, making **repeated passes** at decreasing distances, or exhibiting **erratic approach** behaviour.

### Study area
Gulf of Khambhat (Saurashtra coast), Gujarat, India — a live shipping corridor and candidate region for offshore wind development.

> ⚠️ All vessel tracks are **SIMULATED**. No real AIS data is used.

### How to run

```bash
cd maritime_alerts

# Install deps with uv
uv sync

# Step 1 — rule-based pipeline (tracks + zone alerts)
uv run python maritime_alerts.py output/

# Step 2 — train ML model + score tracks (adds BEHAVIOURAL_ANOMALY alerts)
uv run python ml_model.py

# No uv? Works with system Python too
pip install shapely geopy pandas scikit-learn joblib numpy
python3 maritime_alerts.py output/
python3 ml_model.py
```

Output lands in `maritime_alerts/output/` — load into any GeoJSON viewer or pass to the dashboard module.

### ML model details

- **Algorithm:** RandomForestClassifier (scikit-learn) — CPU only, no GPU needed
- **Training data:** 1 200 synthetic tracks across 4 behaviour classes
- **Training time:** ~4 seconds on a laptop CPU
- **Test accuracy:** 100% on held-out synthetic set (expected — clean archetypes)
- **Features:** 12 geometry-derived features (speed variance, heading changes, zone dwell time, closest approach rate, etc.) — no raw coordinates fed to the model

### Kaggle notebook
`maritime_alerts_kaggle.ipynb` runs the full pipeline on Kaggle with zero changes.
Upload → set accelerator to **None** (rule-based + RandomForest need no GPU) → Run All → download from Output tab.
Switch to **GPU T4×2** only if you upgrade to an LSTM sequence model.

---

## For the Dashboard Team

Three files are ready in `maritime_alerts/output/`:

| File | What to do with it |
|---|---|
| `tracks.geojson` | Render vessel paths as a map layer |
| `zones.geojson` | Render watch-area polygon + asset marker |
| `alerts.json` | Populate the alert panel (sorted by timestamp) |

Alert object shape:
```json
{
  "alert_id": "ALT-916319C2",
  "track_id": "TRACK-001",
  "alert_type": "BOUNDARY_CROSSING_OBSERVED",
  "timestamp": "2024-06-15T06:10:00+00:00",
  "lat": 20.55,
  "lon": 71.60,
  "zone_id": "ZONE-GJ-KHAMBHAT-01",
  "asset_id": null,
  "distance_to_asset_km": 29.721,
  "data_source": "SIMULATED"
}
```

---

## For LLMs / AI Agents Reading This Repo

**Purpose:** Maritime vessel anomaly detection for a renewable energy infrastructure monitoring demo.

**Entry points:**
- `maritime_alerts/maritime_alerts.py` — run with `python3 maritime_alerts.py <output_dir>`. Generates tracks, zones, and rule-based alerts. No ML dependencies required.
- `maritime_alerts/ml_model.py` — run after the above. Trains a RandomForest on synthetic tracks, scores the demo tracks, appends `BEHAVIOURAL_ANOMALY` alerts to `output/alerts.json`.

**Key constants to change** (top of each file):
- `WATCH_AREA["coordinates"]` — swap in real study-area polygon once grid is locked
- `ASSET["lat"]`, `ASSET["lon"]` — real asset location
- `ASSET["proximity_radius_km"]` — tune per asset scale

**What is intentionally excluded:** AIS-association, nationality inference, real-time satellite tasking, hostile-intent classification, dark-ship detection.

**Data contract (output/alerts.json):**
```
alert_id        str   unique hex ID
track_id        str   TRACK-001 / TRACK-002 / TRACK-003
alert_type      str   BOUNDARY_CROSSING_OBSERVED | ASSET_PROXIMITY_ALERT | BEHAVIOURAL_ANOMALY
timestamp       str   ISO-8601 UTC
lat, lon        float triggering position
zone_id         str?  set for polygon alerts, null otherwise
asset_id        str?  set for proximity/ML alerts, null otherwise
distance_to_asset_km  float
suspicious_score      float (ML alerts only, 0–1)
predicted_class       str   (ML alerts only)
data_source     str   always "SIMULATED"
```

---

## Stack

| Layer | Tool |
|---|---|
| Package management | [uv](https://docs.astral.sh/uv/) |
| Geometry | shapely (with pure-Python fallback) |
| Distance | geopy Haversine (with pure-Python fallback) |
| ML | scikit-learn RandomForestClassifier |
| Data | pandas, numpy |
| Notebook | Jupyter (Kaggle-compatible) |

---

*Space-to-Sea · MachAthon · Gujarat, India · All vessel data: SIMULATED*
