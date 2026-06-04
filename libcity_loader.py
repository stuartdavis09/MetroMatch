"""
LibCity BEIJING_SUBWAY loader  (measured crowding, beta factor)
===============================================================

Computes crowding factors from the *measured* LibCity Beijing Subway dataset
instead of hand-picked estimates.

Dataset: LibCity `BEIJING_SUBWAY` — Beijing subway entry/exit flow, 05:00-23:00,
five weeks (29 Feb - 3 Apr 2016), 17 lines / 276 stations, 10-minute interval.
Download (not bundled; ~tens of MB): LibCity provides it via BaiduDisk
(code 1231) or Google Drive. Place the atomic files at:
    raw_data/BEIJING_SUBWAY/BEIJING_SUBWAY.geo
    raw_data/BEIJING_SUBWAY/BEIJING_SUBWAY.dyna

Atomic-file schema (confirmed from LibCity's conversion script + docs):
    .geo  : geo_id, type, coordinates([lng,lat] GeoJSON), <name?>
    .dyna : dyna_id, type, time(ISO-8601), entity_id(=geo_id), inflow, outflow

What this computes (all normalised so the network mean = 1.0, i.e. directly
usable as multipliers):
  * line_load[line]      : mean station flow on that line / network mean flow
  * tod_factor(weekday,h): mean network flow in that hour / daily mean flow
                           (separate weekday and weekend profiles)

Stations are mapped to lines using the Amap network (by Chinese name, with a
nearest-coordinate fallback). Lines/stations the 2016 dataset doesn't cover
simply have no measured factor, and the caller falls back to the model.

Uses only the standard library (csv) and streams the .dyna file, so it handles
the ~1M-row file without pandas and without loading it all into memory.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field
from datetime import datetime


# ---------------------------------------------------------------------------
# Result object consumed by the recommender's crowd model
# ---------------------------------------------------------------------------
@dataclass
class MeasuredCrowd:
    line_load: dict[str, float] = field(default_factory=dict)
    # profiles[is_weekday][hour] -> multiplier (network mean over the day = 1.0)
    weekday_profile: dict[int, float] = field(default_factory=dict)
    weekend_profile: dict[int, float] = field(default_factory=dict)
    coverage: dict = field(default_factory=dict)

    def tod_factor(self, now: datetime) -> float | None:
        prof = self.weekday_profile if now.weekday() < 5 else self.weekend_profile
        if not prof:
            return None
        return prof.get(now.hour, prof.get(min(prof, key=lambda h: abs(h - now.hour))))

    def summary(self) -> str:
        c = self.coverage
        loads = ", ".join(f"{ln}={v:.2f}" for ln, v in
                          sorted(self.line_load.items(), key=lambda kv: -kv[1]))
        return (f"LibCity BEIJING_SUBWAY measured crowding\n"
                f"  date range : {c.get('date_min')} .. {c.get('date_max')}\n"
                f"  stations   : {c.get('matched_stations')}/{c.get('total_stations')} "
                f"mapped to lines\n"
                f"  line loads (network mean = 1.0): {loads}")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _parse_coords(raw: str):
    """'[116.35, 39.90]' -> (116.35, 39.90)  (GeoJSON: lng first)."""
    raw = raw.strip().strip("[]")
    parts = [p for p in raw.replace(" ", "").split(",") if p]
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def _parse_time(raw: str) -> datetime | None:
    raw = raw.strip().replace("Z", "").replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _norm(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "")


def _read_geo(path: str):
    """Return {geo_id: {'lng','lat','name'}}."""
    stations = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        # detect a name column (anything that isn't id/type/coordinates)
        name_cols = [c for c in cols
                     if c.lower() not in ("geo_id", "type", "coordinates")]
        for row in reader:
            gid = row.get("geo_id")
            coord = _parse_coords(row.get("coordinates", ""))
            name = ""
            for nc in name_cols:
                v = (row.get(nc) or "").strip()
                if v and not v.replace(".", "").replace("-", "").isdigit():
                    name = v
                    break
            if gid is None:
                continue
            stations[gid] = {
                "lng": coord[0] if coord else None,
                "lat": coord[1] if coord else None,
                "name": name,
            }
    return stations


# ---------------------------------------------------------------------------
# Station -> line mapping (via the Amap network)
# ---------------------------------------------------------------------------
def _map_stations_to_lines(geo_stations, amap_net):
    """Return {geo_id: set(lines)} using name match, then nearest coordinate."""
    by_name = {}
    for st in amap_net.stations.values():
        if st.name_cn:
            by_name.setdefault(_norm(st.name_cn), set()).update(st.lines)
        if st.name_en:
            by_name.setdefault(_norm(st.name_en), set()).update(st.lines)

    amap_pts = [(s.lat, s.lng, s.lines) for s in amap_net.stations.values()]
    mapping = {}
    for gid, g in geo_stations.items():
        lines = by_name.get(_norm(g["name"]))
        if not lines and g["lat"] is not None:
            best, bestd = None, 0.03  # ~3 km tolerance (WGS/GCJ offset safe-ish)
            for lat, lng, ls in amap_pts:
                d = math.hypot(lat - g["lat"], lng - g["lng"])
                if d < bestd:
                    best, bestd = ls, d
            lines = best
        if lines:
            mapping[gid] = set(lines)
    return mapping


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def load_libcity(data_dir: str, amap_net, dataset: str = "BEIJING_SUBWAY"
                 ) -> MeasuredCrowd | None:
    geo_path = os.path.join(data_dir, f"{dataset}.geo")
    dyna_path = os.path.join(data_dir, f"{dataset}.dyna")
    if not (os.path.exists(geo_path) and os.path.exists(dyna_path)):
        return None

    geo = _read_geo(geo_path)
    st2lines = _map_stations_to_lines(geo, amap_net)

    # Streaming aggregation over the (possibly huge) .dyna file.
    station_sum: dict[str, float] = {}
    station_cnt: dict[str, int] = {}
    # hour profiles: (is_weekday, hour) -> [sum, count]
    prof: dict[tuple, list] = {}
    date_min = date_max = None

    with open(dyna_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = [c.lower() for c in (reader.fieldnames or [])]
        flow_cols = [c for c in (reader.fieldnames or [])
                     if c.lower() in ("inflow", "outflow", "flow",
                                      "in_flow", "out_flow")]
        for row in reader:
            eid = row.get("entity_id")
            if eid is None:
                continue
            try:
                flow = sum(float(row.get(c, 0) or 0) for c in flow_cols)
            except ValueError:
                continue
            station_sum[eid] = station_sum.get(eid, 0.0) + flow
            station_cnt[eid] = station_cnt.get(eid, 0) + 1

            t = _parse_time(row.get("time", ""))
            if t is not None:
                d = t.date()
                date_min = d if date_min is None or d < date_min else date_min
                date_max = d if date_max is None or d > date_max else date_max
                key = (t.weekday() < 5, t.hour)
                acc = prof.setdefault(key, [0.0, 0])
                acc[0] += flow
                acc[1] += 1

    if not station_sum:
        return None

    # Mean flow per station (per 10-min record).
    station_mean = {s: station_sum[s] / station_cnt[s] for s in station_sum}

    # Per-line mean station flow, normalised to the matched-network mean.
    line_acc: dict[str, list] = {}
    matched = 0
    for sid, mean in station_mean.items():
        lines = st2lines.get(sid)
        if not lines:
            continue
        matched += 1
        for ln in lines:
            a = line_acc.setdefault(ln, [0.0, 0])
            a[0] += mean
            a[1] += 1
    line_mean = {ln: a[0] / a[1] for ln, a in line_acc.items() if a[1]}
    net_mean = (sum(line_mean.values()) / len(line_mean)) if line_mean else 1.0
    line_load = {ln: round(v / net_mean, 3) for ln, v in line_mean.items()}

    # Time-of-day profiles, normalised so each profile's daily mean = 1.0.
    def _profile(is_weekday: bool):
        rows = {h: s / c for (wd, h), (s, c) in prof.items()
                if wd is is_weekday and c}
        if not rows:
            return {}
        m = sum(rows.values()) / len(rows)
        return {h: round(v / m, 3) for h, v in rows.items()} if m else {}

    return MeasuredCrowd(
        line_load=line_load,
        weekday_profile=_profile(True),
        weekend_profile=_profile(False),
        coverage={
            "date_min": str(date_min), "date_max": str(date_max),
            "total_stations": len(geo), "matched_stations": matched,
            "lines": len(line_load),
        },
    )


if __name__ == "__main__":
    import sys
    import amap_parser as ap
    d = sys.argv[1] if len(sys.argv) > 1 else "raw_data/BEIJING_SUBWAY"
    net = ap.build_network(allow_network=False)
    mc = load_libcity(d, net)
    print(mc.summary() if mc else f"No LibCity data found in {d}")
