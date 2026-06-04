"""
Script 1 — Amap (Gaode) Beijing Subway parser
==============================================

PRIMARY TOPOLOGY SOURCE (as specified in the project):
    http://map.amap.com/service/subway?_=1&srhdata=1100_drw_beijing.json

This single file describes the whole network: every line, its stations in
order, GCJ-02 coordinates, and transfer flags. We parse it into a weighted
graph used by the recommender:

    vertices  = (station_id, line)        # a platform of one line at a station
    ride edge = adjacent stations on a line; travel time derived from the
                real GCJ-02 coordinates (haversine distance / operating speed)
    transfer  = two platforms that share the same station_id (an interchange)

Relevant Amap fields
--------------------
  top level : l   -> list of line objects
  line obj  : ln/kn = name, cl = colour, lo = "1" if loop line,
              li = line id(s), st = ordered list of stations
  station   : n  = Chinese name, sp = romanised name (for English UI),
              sl = "lng,lat" in GCJ-02, t = "1" if interchange,
              si/sid = station id, r = ids of lines serving the station

Cross-checking English labels / codes against the Wikipedia "List of Beijing
Subway stations" page is supported by keeping both `n` (Chinese) and `sp`
(romanised) on every Station; swap in a names file if desired.

Network access in this environment may be disabled, so loading falls back to
a bundled real-data sample (beijing_amap_sample.json). With internet access
the full live network is fetched and cached.
"""

from __future__ import annotations

import csv
import json
import math
import os
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field

AMAP_URL = "http://map.amap.com/service/subway?_=1&srhdata=1100_drw_beijing.json"
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(_HERE, "beijing_amap_cache.json")
SAMPLE_FILE = os.path.join(_HERE, "beijing_amap_sample.json")
DISTANCES_FILE = os.path.join(_HERE, "bjsubway_distances.csv")

# --- travel-time model (alpha factor input) -------------------------------
# Travel time uses bjsubway.com OFFICIAL inter-station distances (metres) when
# available (see bjsubway_distances.csv), falling back to the GCJ-02
# coordinate (haversine) estimate for any segment not in that table.
# Distance is converted to minutes with a moving speed + per-stop dwell;
# Line 1 (~31 km / ~30 stops in ~85 min) calibrates this to ~27 km/h all-in.
MOVE_SPEED_KMH = 32.0
DWELL_MIN = 0.5
MIN_HOP_MIN = 1.0

# --- default transfer cost (gamma factor input) ---------------------------
# Amap does not carry in-station walking distance; use a sensible default and
# refine per-station from bjsubway.com / 本地宝 tables when available.
DEFAULT_TRANSFER_WALK_MIN = 4.0
DEFAULT_TRANSFER_WALK_M = 250.0


@dataclass
class Station:
    sid: str
    name_cn: str
    name_en: str
    lat: float
    lng: float
    is_transfer: bool
    lines: set = field(default_factory=set)

    @property
    def label(self) -> str:
        return f"{self.name_en} ({self.name_cn})" if self.name_cn else self.name_en


Node = tuple  # (sid, line_name)


@dataclass
class Edge:
    target: Node
    kind: str          # "ride" | "transfer"
    line: str
    dist_km: float
    time_min: float    # in-vehicle minutes (ride); 0 for transfer
    walk_min: float    # walking minutes (transfer); 0 for ride
    walk_m: float
    accessible: bool   # step-free? ride edges are always True
    source: str = "coord"  # ride distance origin: "bjsubway" | "coord"


def _norm_cn(s: str) -> str:
    return (s or "").strip().replace(" ", "")


def load_bjsubway_distances(path: str = DISTANCES_FILE) -> dict:
    """Official bjsubway.com inter-station distances.

    Reads a CSV (line, from, to, meters) and returns a symmetric lookup
    {(from, to): metres} keyed by Chinese station names. Missing file -> {}.
    """
    out: dict = {}
    if not os.path.exists(path):
        return out
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                a, b = _norm_cn(row.get("from", "")), _norm_cn(row.get("to", ""))
                try:
                    m = float(row.get("meters", ""))
                except (TypeError, ValueError):
                    continue
                if a and b:
                    out[(a, b)] = m
                    out[(b, a)] = m
    except OSError:
        pass
    return out


def haversine_km(a: Station, b: Station) -> float:
    """Great-circle distance in km. GCJ-02 offset cancels over short hops,
    so this is accurate for adjacent-station distances."""
    r = 6371.0
    p1, p2 = math.radians(a.lat), math.radians(b.lat)
    dp = math.radians(b.lat - a.lat)
    dl = math.radians(b.lng - a.lng)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def hop_time_min(dist_km: float) -> float:
    return max(MIN_HOP_MIN, dist_km / MOVE_SPEED_KMH * 60.0 + DWELL_MIN)


def _clean_en(sp: str) -> str:
    sp = (sp or "").strip()
    if not sp:
        return ""
    return " ".join(w[:1].upper() + w[1:] for w in sp.split())


# ---------------------------------------------------------------------------
# Loading the raw Amap JSON
# ---------------------------------------------------------------------------
def _fetch_live(timeout: float = 15.0) -> dict:
    req = urllib.request.Request(AMAP_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_raw(source: str | None = None, allow_network: bool = True,
             cache: str = DEFAULT_CACHE) -> tuple[dict, str]:
    """Return (raw_json, provenance_string).

    Priority: explicit source -> cache file -> live network (then cached)
    -> bundled real-data sample.
    """
    if source and os.path.exists(source):
        with open(source, encoding="utf-8") as f:
            return json.load(f), f"file: {os.path.basename(source)}"

    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return json.load(f), f"cache: {os.path.basename(cache)}"

    if allow_network:
        try:
            raw = _fetch_live()
            try:
                with open(cache, "w", encoding="utf-8") as f:
                    json.dump(raw, f, ensure_ascii=False)
            except OSError:
                pass
            return raw, "live Amap network (full Beijing network)"
        except Exception as exc:  # network blocked / offline
            print(f"  [parser] live fetch failed ({exc.__class__.__name__}); "
                  f"using bundled sample.")

    with open(SAMPLE_FILE, encoding="utf-8") as f:
        return json.load(f), "bundled real-data sample (subset)"


# ---------------------------------------------------------------------------
# Building the weighted graph
# ---------------------------------------------------------------------------
class Network:
    def __init__(self) -> None:
        self.stations: dict[str, Station] = {}
        self.adj: dict[Node, list[Edge]] = defaultdict(list)
        self.line_meta: dict[str, dict] = {}
        self.line_stops: dict[str, list[str]] = {}
        self._name_index: dict[str, str] = {}
        self.provenance: str = ""
        self.dist_official: int = 0
        self.dist_coord: int = 0

    # -- lookups ----------------------------------------------------------
    def _norm(self, s: str) -> str:
        return s.strip().lower().replace(" ", "")

    def resolve(self, raw: str) -> str | None:
        key = self._norm(raw)
        if key in self._name_index:
            return self._name_index[key]
        hits = {sid for k, sid in self._name_index.items() if key and key in k}
        if len(hits) == 1:
            return next(iter(hits))
        return None

    def suggestions(self, raw: str, limit: int = 8) -> list[str]:
        key = self._norm(raw)
        out = []
        for sid, st in self.stations.items():
            if key and key in self._norm(st.name_en + st.name_cn):
                out.append(st.label)
        return sorted(set(out))[:limit]

    def platforms_at(self, sid: str) -> list[Node]:
        return [n for n in self.adj if n[0] == sid]

    def all_station_labels(self) -> list[str]:
        return sorted({st.label for st in self.stations.values()})


def build_network(source: str | None = None, allow_network: bool = True,
                  cache: str = DEFAULT_CACHE,
                  transfer_walk_min: float = DEFAULT_TRANSFER_WALK_MIN,
                  transfer_walk_m: float = DEFAULT_TRANSFER_WALK_M,
                  accessibility: dict[str, bool] | None = None,
                  distances: dict | None = None) -> Network:
    """Parse the Amap network into a Network of vertices + weighted edges.

    accessibility: optional {station_id: step_free_bool} from 本地宝 / MTR
    tables. Missing entries default to accessible (with a coverage caveat).
    distances: optional {(name_cn, name_cn): metres} of official bjsubway
    inter-station distances. Defaults to auto-loading bjsubway_distances.csv;
    segments not covered fall back to the coordinate estimate.
    """
    raw, provenance = load_raw(source, allow_network, cache)
    if distances is None:
        distances = load_bjsubway_distances()
    net = Network()
    net.provenance = provenance
    net.dist_official = 0
    net.dist_coord = 0

    # First pass: stations + ordered stop lists per line.
    for L in raw.get("l", []):
        line = L.get("ln") or L.get("kn") or L.get("li", "?")
        net.line_meta[line] = {
            "name": line,
            "color": L.get("cl", ""),
            "loop": str(L.get("lo", "0")) == "1",
        }
        stops: list[str] = []
        for s in L.get("st", []):
            sid = s.get("si") or s.get("sid")
            if not sid or "sl" not in s:
                continue
            try:
                lng, lat = (float(x) for x in s["sl"].split(","))
            except ValueError:
                continue
            st = net.stations.get(sid)
            if st is None:
                st = Station(sid, s.get("n", ""), _clean_en(s.get("sp", "")),
                             lat, lng, str(s.get("t", "0")) == "1", set())
                net.stations[sid] = st
            st.lines.add(line)
            stops.append(sid)
        net.line_stops[line] = stops

    # Name index for lookups (romanised + Chinese -> sid).
    for sid, st in net.stations.items():
        if st.name_en:
            net._name_index[net._norm(st.name_en)] = sid
        if st.name_cn:
            net._name_index[net._norm(st.name_cn)] = sid

    # Ride edges along each line (close the loop where flagged).
    for line, stops in net.line_stops.items():
        pairs = list(zip(stops, stops[1:]))
        if net.line_meta[line]["loop"] and len(stops) > 2:
            pairs.append((stops[-1], stops[0]))
        for a, b in pairs:
            if a == b:
                continue
            key = (_norm_cn(net.stations[a].name_cn), _norm_cn(net.stations[b].name_cn))
            if key in distances:
                d = distances[key] / 1000.0       # official metres -> km
                src = "bjsubway"
                net.dist_official += 1
            else:
                d = haversine_km(net.stations[a], net.stations[b])
                src = "coord"
                net.dist_coord += 1
            t = hop_time_min(d)
            na, nb = (a, line), (b, line)
            net.adj[na].append(Edge(nb, "ride", line, d, t, 0.0, 0.0, True, src))
            net.adj[nb].append(Edge(na, "ride", line, d, t, 0.0, 0.0, True, src))

    # Transfer edges: any station served by >1 line (shared station id).
    platforms: dict[str, set] = defaultdict(set)
    for (sid, line) in list(net.adj.keys()):
        platforms[sid].add(line)
    for sid, lines in platforms.items():
        ls = sorted(lines)
        if len(ls) < 2:
            continue
        acc = True if accessibility is None else accessibility.get(sid, True)
        for i in range(len(ls)):
            for j in range(i + 1, len(ls)):
                a, b = (sid, ls[i]), (sid, ls[j])
                net.adj[a].append(
                    Edge(b, "transfer", "transfer", 0.0, 0.0,
                         transfer_walk_min, transfer_walk_m, acc))
                net.adj[b].append(
                    Edge(a, "transfer", "transfer", 0.0, 0.0,
                         transfer_walk_min, transfer_walk_m, acc))

    return net


def apply_accessibility(net: Network, acc: dict[str, bool]) -> int:
    """Update transfer-edge step-free flags in place from {station_id: bool}.

    A transfer edge joins two platforms of the same station, so its
    accessibility is the station's accessibility. Stations absent from `acc`
    keep the default (accessible). Returns the number of interchange stations
    marked NOT step-free.
    """
    not_free = set()
    for node, edges in net.adj.items():
        for e in edges:
            if e.kind == "transfer":
                step_free = acc.get(node[0], True)
                e.accessible = step_free
                if not step_free:
                    not_free.add(node[0])
    return len(not_free)


def network_summary(net: Network) -> str:
    n_stations = len(net.stations)
    n_lines = len(net.line_meta)
    n_transfers = sum(1 for st in net.stations.values() if len(st.lines) > 1)
    n_ride = sum(1 for edges in net.adj.values() for e in edges if e.kind == "ride") // 2
    seg = net.dist_official + net.dist_coord
    pct = (100 * net.dist_official / seg) if seg else 0
    return (f"source: {net.provenance}\n"
            f"  lines={n_lines}  stations={n_stations}  "
            f"interchanges={n_transfers}  ride-edges={n_ride}\n"
            f"  travel time: bjsubway official distance for {net.dist_official}/{seg} "
            f"segments ({pct:.0f}%), coordinate estimate for the rest")


if __name__ == "__main__":
    net = build_network()
    print(network_summary(net))
    print("  lines:", ", ".join(sorted(net.line_meta)))
