"""
maritime_alerts.py
==================
Space-to-Sea Hackathon — Maritime Alerts Module
Workstream: Vessel Monitoring & Infrastructure Protection

Study area: Palk Strait / Gulf of Mannar coast, Tamil Nadu, India
            (between Rameswaram and Tuticorin — active fishing + shipping corridor)

!!! ALL VESSEL TRACKS ARE SIMULATED — NOT REAL AIS DATA !!!

Outputs:
  tracks.geojson   — simulated track points as GeoJSON FeatureCollection
  alerts.json      — alert objects (BOUNDARY_CROSSING_OBSERVED / ASSET_PROXIMITY_ALERT)
  zones.geojson    — watch-area polygon + asset point

Dependencies: shapely, geopy, pandas  (all CPU-only, pip-installable)
"""

import json
import uuid
import math
from datetime import datetime, timedelta, timezone

# ── optional imports with graceful fallback ──────────────────────────────────
try:
    from shapely.geometry import Point, Polygon, mapping
    SHAPELY_OK = True
except ImportError:
    SHAPELY_OK = False
    print("[WARN] shapely not found — falling back to ray-casting PiP.")

try:
    from geopy.distance import geodesic
    GEOPY_OK = True
except ImportError:
    GEOPY_OK = False
    print("[WARN] geopy not found — falling back to Haversine distance.")

# ═══════════════════════════════════════════════════════════════════════════
# 1. ZONE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

# Watch-area polygon: a rectangular corridor in the Palk Strait /
# Gulf of Mannar, covering the approaches to a notional offshore wind
# survey zone.  Approximate box: 8.8°N–9.2°N, 78.8°E–79.3°E.
WATCH_AREA = {
    "zone_id": "ZONE-TN-GULF-01",
    "name": "Gulf of Mannar Offshore Survey Corridor",
    "description": "Watch-area for the notional offshore wind survey zone, Tamil Nadu coast.",
    "coordinates": [           # (lon, lat) — GeoJSON convention
        [78.80, 8.80],
        [79.30, 8.80],
        [79.30, 9.20],
        [78.80, 9.20],
        [78.80, 8.80],         # close the ring
    ],
}

# Protected asset: notional offshore substation / metocean buoy
ASSET = {
    "asset_id": "ASSET-TN-SUB-01",
    "name": "Gulf of Mannar Offshore Substation (SIMULATED)",
    "lat": 9.00,
    "lon": 79.05,
    "proximity_radius_km": 2.0,   # configurable — swap to suit asset scale
}

# ═══════════════════════════════════════════════════════════════════════════
# 2. SIMULATED VESSEL TRACKS
# ═══════════════════════════════════════════════════════════════════════════

def _make_track(track_id, points_latlon, t0: datetime, interval_minutes=10):
    """Zip a list of (lat, lon) waypoints with timestamps."""
    rows = []
    for i, (lat, lon) in enumerate(points_latlon):
        ts = t0 + timedelta(minutes=i * interval_minutes)
        rows.append({
            "track_id": track_id,
            "timestamp": ts.isoformat(),
            "lat": lat,
            "lon": lon,
            "data_source": "SIMULATED",
        })
    return rows


def build_simulated_tracks():
    """
    Three synthetic vessel tracks over the Tamil Nadu study area.

    TRACK-001  Fishing vessel — transits THROUGH the watch-area polygon
               and passes close to the asset.           → both alert types
    TRACK-002  Cargo vessel   — skirts the polygon edge but stays outside.
                                                        → no alerts
    TRACK-003  Survey vessel  — enters the watch area but stays far from
               the asset.                               → BOUNDARY only
    """
    t0 = datetime(2024, 6, 15, 6, 0, 0, tzinfo=timezone.utc)   # 06:00 UTC

    # TRACK-001: enters at SW corner, crosses through, nears asset
    track_001 = _make_track("TRACK-001", [
        (8.70, 78.70),   # outside — south-west of polygon
        (8.82, 78.85),   # enters polygon (SW quadrant)
        (8.92, 78.95),   # inside polygon, moving NE
        (8.99, 79.03),   # ~2.5 km from asset  → BOUNDARY_CROSSING_OBSERVED
        (9.00, 79.06),   # ~1.1 km from asset  → ASSET_PROXIMITY_ALERT
        (9.05, 79.15),   # still inside polygon
        (9.18, 79.25),   # near NE exit of polygon
        (9.30, 79.40),   # outside — exits NE
    ], t0)

    # TRACK-002: cargo vessel — stays entirely outside, passes south of polygon
    track_002 = _make_track("TRACK-002", [
        (8.50, 78.50),   # well outside SW
        (8.60, 78.70),   # heading east, below polygon
        (8.65, 78.90),   # parallel to S edge but south of 8.80 boundary
        (8.72, 79.10),   # continues east, still outside
        (8.70, 79.40),   # well outside SE
        (8.65, 79.60),   # exits east
    ], t0, interval_minutes=15)

    # TRACK-003: survey vessel — enters polygon but loops away from asset
    track_003 = _make_track("TRACK-003", [
        (8.78, 79.28),   # just outside E edge
        (8.85, 79.22),   # enters polygon (NE area)
        (8.95, 79.18),   # inside, but ~6 km from asset
        (9.10, 79.20),   # inside, NE quadrant
        (9.18, 79.28),   # exits N edge
        (9.25, 79.35),   # outside
    ], t0, interval_minutes=12)

    return track_001 + track_002 + track_003


# ═══════════════════════════════════════════════════════════════════════════
# 3. GEOMETRY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _haversine_km(lat1, lon1, lat2, lon2):
    """Pure-Python Haversine — fallback when geopy unavailable."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def distance_km(lat1, lon1, lat2, lon2):
    if GEOPY_OK:
        return geodesic((lat1, lon1), (lat2, lon2)).km
    return _haversine_km(lat1, lon1, lat2, lon2)


def _ray_cast_pip(lat, lon, ring):
    """
    Ray-casting point-in-polygon — fallback when shapely unavailable.
    ring: list of [lon, lat] pairs (GeoJSON convention).
    """
    x, y = lon, lat
    inside = False
    n = len(ring) - 1  # last == first (closed ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def point_in_watch_area(lat, lon):
    ring = WATCH_AREA["coordinates"]
    if SHAPELY_OK:
        poly = Polygon([(c[0], c[1]) for c in ring])
        return poly.contains(Point(lon, lat))
    return _ray_cast_pip(lat, lon, ring)


# ═══════════════════════════════════════════════════════════════════════════
# 4. ALERT LOGIC
# ═══════════════════════════════════════════════════════════════════════════

def run_alert_logic(tracks):
    alerts = []

    asset_lat = ASSET["lat"]
    asset_lon = ASSET["lon"]
    radius_km = ASSET["proximity_radius_km"]

    for point in tracks:
        lat = point["lat"]
        lon = point["lon"]
        track_id = point["track_id"]
        ts = point["timestamp"]

        # ── Rule 1: Point-in-polygon ──────────────────────────────────────
        if point_in_watch_area(lat, lon):
            alerts.append({
                "alert_id": f"ALT-{uuid.uuid4().hex[:8].upper()}",
                "track_id": track_id,
                "alert_type": "BOUNDARY_CROSSING_OBSERVED",
                "timestamp": ts,
                "lat": lat,
                "lon": lon,
                "zone_id": WATCH_AREA["zone_id"],
                "asset_id": None,
                "distance_to_asset_km": round(
                    distance_km(lat, lon, asset_lat, asset_lon), 3
                ),
                "data_source": "SIMULATED",
            })

        # ── Rule 2: Proximity to protected asset ──────────────────────────
        dist = distance_km(lat, lon, asset_lat, asset_lon)
        if dist <= radius_km:
            alerts.append({
                "alert_id": f"ALT-{uuid.uuid4().hex[:8].upper()}",
                "track_id": track_id,
                "alert_type": "ASSET_PROXIMITY_ALERT",
                "timestamp": ts,
                "lat": lat,
                "lon": lon,
                "zone_id": None,
                "asset_id": ASSET["asset_id"],
                "distance_to_asset_km": round(dist, 3),
                "proximity_radius_km": radius_km,
                "data_source": "SIMULATED",
            })

    return alerts


# ═══════════════════════════════════════════════════════════════════════════
# 5. OUTPUT FILE WRITERS
# ═══════════════════════════════════════════════════════════════════════════

def write_tracks_geojson(tracks, path="tracks.geojson"):
    features = []
    for p in tracks:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [p["lon"], p["lat"]]},
            "properties": {
                "track_id":    p["track_id"],
                "timestamp":   p["timestamp"],
                "data_source": "SIMULATED",
            },
        })
    fc = {"type": "FeatureCollection", "features": features,
          "metadata": {"data_source": "SIMULATED", "generated_by": "maritime_alerts.py"}}
    with open(path, "w") as f:
        json.dump(fc, f, indent=2)
    print(f"[OK] {path}  ({len(features)} points across {len({p['track_id'] for p in tracks})} tracks)")


def write_alerts_json(alerts, path="alerts.json"):
    out = {
        "metadata": {
            "data_source": "SIMULATED",
            "generated_by": "maritime_alerts.py",
            "total_alerts": len(alerts),
        },
        "alerts": alerts,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[OK] {path}  ({len(alerts)} alerts)")


def write_zones_geojson(path="zones.geojson"):
    features = [
        # Watch-area polygon
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [WATCH_AREA["coordinates"]],
            },
            "properties": {
                "zone_id":     WATCH_AREA["zone_id"],
                "name":        WATCH_AREA["name"],
                "description": WATCH_AREA["description"],
                "type":        "WATCH_AREA",
                "data_source": "SIMULATED",
            },
        },
        # Protected asset point
        {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [ASSET["lon"], ASSET["lat"]],
            },
            "properties": {
                "asset_id":              ASSET["asset_id"],
                "name":                  ASSET["name"],
                "proximity_radius_km":   ASSET["proximity_radius_km"],
                "type":                  "PROTECTED_ASSET",
                "data_source":           "SIMULATED",
            },
        },
    ]
    fc = {"type": "FeatureCollection", "features": features,
          "metadata": {"data_source": "SIMULATED", "generated_by": "maritime_alerts.py"}}
    with open(path, "w") as f:
        json.dump(fc, f, indent=2)
    print(f"[OK] {path}  (1 watch-area polygon + 1 asset point)")


# ═══════════════════════════════════════════════════════════════════════════
# 6. MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main(output_dir="."):
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("\n━━━ Space-to-Sea | Maritime Alerts Module ━━━")
    print("    Study area : Gulf of Mannar, Tamil Nadu, India")
    print("    Data status: ALL DATA IS SIMULATED\n")

    # Stage 1 — tracks
    print("[STAGE 1] Generating simulated vessel tracks...")
    tracks = build_simulated_tracks()
    track_ids = sorted({p["track_id"] for p in tracks})
    print(f"          {len(track_ids)} tracks, {len(tracks)} total points: {track_ids}")

    # Stage 2 — zones (no dependency on tracks)
    print("[STAGE 2] Zone definitions ready (see WATCH_AREA / ASSET constants).")

    # Stage 3 — alert logic
    print("[STAGE 3] Running alert logic...")
    alerts = run_alert_logic(tracks)
    by_type = {}
    for a in alerts:
        by_type.setdefault(a["alert_type"], 0)
        by_type[a["alert_type"]] += 1
    for atype, count in by_type.items():
        print(f"          {atype}: {count}")

    # Stage 4 — export
    print("[STAGE 4] Writing output files...")
    write_tracks_geojson(tracks,  os.path.join(output_dir, "tracks.geojson"))
    write_alerts_json(alerts,     os.path.join(output_dir, "alerts.json"))
    write_zones_geojson(          os.path.join(output_dir, "zones.geojson"))

    print("\n━━━ Pipeline complete. Hand-off files ready in:", output_dir, "━━━\n")
    return tracks, alerts


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    main(output_dir=out)
