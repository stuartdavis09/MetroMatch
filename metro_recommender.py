"""
Beijing Metro — Preference-Based Path Recommendation System
===========================================================

Backend that recommends a metro route by a weighted cost rather than pure
shortest-time, per the project brief:

    Cost = alpha*(time) + beta*(crowd) + gamma*(walking) + delta*(accessibility)

DATA SOURCES (all from the project's specified databases)
  topology + coordinates : Amap subway JSON, parsed by amap_parser.py
                           (script 1). Travel time (alpha) is derived from the
                           real GCJ-02 coordinates in that file.
  crowd (beta)           : current Beijing time -> bimodal weekday/weekend
                           ridership pattern from published Beijing Subway flow
                           studies (AM peak ~6:30-10:00, PM peak ~17:00-20:00),
                           scaled by per-line load (Lines 4/5/10 are the most
                           overloaded in the AM peak per city data). A hook is
                           provided to plug in the LibCity BEIJING_SUBWAY flow
                           dataset for measured crowding where it has coverage.
  walking (gamma)        : in-station transfer walk (default; refine per-station
                           from bjsubway.com / 本地宝 tables).
  accessibility (delta)  : optional step-free table from 本地宝 / Beijing MTR;
                           when required, stair-only transfers are excluded.

Algorithm: Dijkstra with a priority queue over (station, line) platform nodes,
using the custom cost function as edge weight.

User input: a 1-5 priority for journey time, crowding, and transfer walking,
plus YES/NO for an accessibility requirement (simple console input()).
"""

from __future__ import annotations

import heapq
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import amap_parser as ap

# Current Beijing time (real Asia/Shanghai; fall back to fixed UTC+8).
try:
    from zoneinfo import ZoneInfo
    BEIJING_TZ = ZoneInfo("Asia/Shanghai")
except Exception:  # pragma: no cover
    BEIJING_TZ = timezone(timedelta(hours=8))


# ===========================================================================
# CROWD MODEL  (beta factor)
# ===========================================================================
# PRIMARY source: measured per-line load + time-of-day profiles computed from
# the LibCity BEIJING_SUBWAY flow dataset (see libcity_loader.py). When that
# data is loaded it is used directly; per line/hour it isn't covered, we fall
# back to the rough estimates below.
#
# _MEASURED is a MeasuredCrowd (or None). Set it once at startup via
# set_measured(); crowd_level() then prefers measured values automatically.
_MEASURED = None


def set_measured(measured) -> None:
    global _MEASURED
    _MEASURED = measured


# --- FALLBACK estimates (used only when measured data is absent) -----------
# These are hand-set rough guesses, NOT measurements. They keep the system
# usable offline / for lines & hours the 2016 dataset doesn't cover. The
# relative ordering (Lines 4/5/10 busiest) follows Beijing's published peak
# load-rate reports; the magnitudes are estimates.
FALLBACK_LINE_LOAD_RULES = [
    ("10号线", 1.35), ("4号线", 1.30), ("5号线", 1.28), ("13号线", 1.25),
    ("1号线", 1.25), ("2号线", 1.20), ("6号线", 1.12), ("8号线", 1.10),
    ("S1", 0.80), ("机场", 0.70), ("S", 0.85),
]


def fallback_line_load(line: str) -> float:
    for key, val in FALLBACK_LINE_LOAD_RULES:
        if key in line:
            return val
    return 1.10


def _bump(x: float, center: float, width: float) -> float:
    return max(0.0, 1.0 - abs(x - center) / width)


def fallback_tod_factor(now: datetime) -> float:
    """Bimodal ridership curve from published Beijing Subway flow studies."""
    hour = now.hour + now.minute / 60.0
    weekday = now.weekday() < 5
    if weekday:
        morning = 0.95 * _bump(hour, 8.0, 1.5)    # 6:30-9:30, peak 08:00
        evening = 0.85 * _bump(hour, 18.0, 1.5)   # 16:30-19:30, peak 18:00
        base = 0.55 if 6.0 <= hour <= 23.0 else 0.20
        return base + morning + evening
    midday = 0.45 * _bump(hour, 15.0, 5.0)        # broad weekend daytime bump
    base = 0.45 if 7.0 <= hour <= 23.0 else 0.20
    return base + midday


def crowd_level(line: str, now: datetime, is_hub: bool) -> float:
    """Crowd multiplier = line load x time-of-day x hub bump.

    Uses measured LibCity factors when available, else fallback estimates,
    decided independently for the line load and the time-of-day term so
    partial dataset coverage still helps.
    """
    load = tod = None
    if _MEASURED is not None:
        load = _MEASURED.line_load.get(line)
        tod = _MEASURED.tod_factor(now)
    if load is None:
        load = fallback_line_load(line)
    if tod is None:
        tod = fallback_tod_factor(now)
    f = load * tod
    if is_hub:
        f *= 1.15
    return round(f, 3)


def crowd_source_label() -> str:
    return "measured (LibCity 2016)" if _MEASURED is not None else "estimated (model)"


def crowd_score(level: float) -> int:
    return 1 + sum(level >= b for b in (0.6, 1.0, 1.4, 1.8))


CROWD_WORDS = {1: "very light", 2: "light", 3: "moderate", 4: "crowded", 5: "packed"}


# ===========================================================================
# PREFERENCES + COST FUNCTION
# ===========================================================================
@dataclass
class Preferences:
    pr_time: int
    pr_crowd: int
    pr_walk: int
    need_accessible: bool

    @property
    def weights(self) -> tuple[float, float, float]:
        total = self.pr_time + self.pr_crowd + self.pr_walk
        return (self.pr_time / total, self.pr_crowd / total, self.pr_walk / total)


TRANSFER_PENALTY_MIN = 2.0   # fixed annoyance per transfer (effective minutes)


def edge_cost(edge: ap.Edge, now: datetime, prefs: Preferences,
              touches_hub: bool) -> float | None:
    """Effective-minutes cost of an edge; None = forbidden under constraints."""
    a, b, g = prefs.weights
    if edge.kind == "ride":
        crowd = crowd_level(edge.line, now, touches_hub)
        return a * edge.time_min + b * (crowd * edge.time_min)
    # transfer
    if prefs.need_accessible and not edge.accessible:
        return None
    return g * (edge.walk_min + TRANSFER_PENALTY_MIN)


# ===========================================================================
# DIJKSTRA  (priority queue over platform nodes; custom cost as edge weight)
# ===========================================================================
SOURCE = ("__SOURCE__", "")
SINK = ("__SINK__", "")


def find_route(net: ap.Network, start_sid: str, end_sid: str,
               now: datetime, prefs: Preferences):
    """Return (total_cost, [(node, edge_into_node)]) or (None, None)."""
    hubs = {sid for sid, st in net.stations.items() if len(st.lines) > 1}
    sink_from = set(net.platforms_at(end_sid))

    dist = {SOURCE: 0.0}
    prev: dict = {}            # node -> (prev_node, edge or None)
    pq = [(0.0, SOURCE)]
    visited = set()

    while pq:
        d, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        if node == SINK:
            break

        neighbours: list[tuple[tuple, float, object]] = []
        if node == SOURCE:
            for p in net.platforms_at(start_sid):
                neighbours.append((p, 0.0, None))           # board
        else:
            if node in sink_from:
                neighbours.append((SINK, 0.0, None))        # alight
            for e in net.adj.get(node, []):
                hub = node[0] in hubs or e.target[0] in hubs
                c = edge_cost(e, now, prefs, hub)
                if c is None:
                    continue
                neighbours.append((e.target, c, e))

        for nxt, c, e in neighbours:
            nd = d + c
            if nd < dist.get(nxt, float("inf")):
                dist[nxt] = nd
                prev[nxt] = (node, e)
                heapq.heappush(pq, (nd, nxt))

    if SINK not in dist:
        return None, None

    # Reconstruct real nodes + the edge used to enter each.
    chain = []
    cur = SINK
    while cur in prev:
        pnode, edge = prev[cur]
        if cur not in (SOURCE, SINK):
            chain.append((cur, edge))
        cur = pnode
    chain.reverse()
    return dist[SINK], chain


# ===========================================================================
# ITINERARY
# ===========================================================================
def build_itinerary(net: ap.Network, chain, now: datetime):
    hubs = {sid for sid, st in net.stations.items() if len(st.lines) > 1}
    legs, transfers = [], []
    cur_line = None
    for (node, edge) in chain:
        sid, line = node
        if edge is not None and edge.kind == "transfer":
            transfers.append({
                "sid": sid,
                "from_line": legs[-1]["line"] if legs else "?",
                "to_line": line,
                "walk_min": edge.walk_min,
                "walk_m": edge.walk_m,
                "accessible": edge.accessible,
            })
            legs.append({"line": line, "sids": [sid], "ride_min": 0.0, "dist_km": 0.0})
            cur_line = line
        else:  # ride edge (or first boarded node)
            if line != cur_line:
                legs.append({"line": line, "sids": [sid], "ride_min": 0.0, "dist_km": 0.0})
                cur_line = line
            else:
                legs[-1]["sids"].append(sid)
            if edge is not None and edge.kind == "ride":
                legs[-1]["ride_min"] += edge.time_min
                legs[-1]["dist_km"] += edge.dist_km

    total_ride = sum(l["ride_min"] for l in legs)
    total_walk = sum(t["walk_min"] for t in transfers)
    crowd_samples = []
    for leg in legs:
        if leg["ride_min"] > 0:
            hub = any(s in hubs for s in leg["sids"])
            crowd_samples.append(crowd_level(leg["line"], now, hub))
    avg_crowd = sum(crowd_samples) / len(crowd_samples) if crowd_samples else 0.0
    return {
        "legs": legs, "transfers": transfers,
        "total_ride_min": total_ride, "total_walk_min": total_walk,
        "total_min": total_ride + total_walk,
        "avg_crowd_level": avg_crowd, "avg_crowd_score": crowd_score(avg_crowd),
    }


def name(net: ap.Network, sid: str) -> str:
    return net.stations[sid].name_en or net.stations[sid].name_cn


def print_itinerary(net, start_sid, end_sid, summ, now, prefs, total_cost):
    a, b, g = prefs.weights
    hubs = {sid for sid, st in net.stations.items() if len(st.lines) > 1}
    print("\n" + "=" * 66)
    print(f"  RECOMMENDED ROUTE:  {name(net, start_sid)}  ->  {name(net, end_sid)}")
    print(f"  Beijing time: {now:%Y-%m-%d %H:%M} ({now:%A})")
    print("=" * 66)
    print(f"  Priorities -> time {prefs.pr_time}/5 (w={a:.2f}), "
          f"crowd {prefs.pr_crowd}/5 (w={b:.2f}), "
          f"walking {prefs.pr_walk}/5 (w={g:.2f})")
    print(f"  Step-free route required: {'YES' if prefs.need_accessible else 'no'}")
    print("-" * 66)

    step = 1
    legs, transfers = summ["legs"], summ["transfers"]
    ti = 0
    for leg in legs:
        if leg["ride_min"] > 0:
            hub = any(s in hubs for s in leg["sids"])
            cs = crowd_score(crowd_level(leg["line"], now, hub))
            stops = len(leg["sids"]) - 1
            print(f"  {step}. Take {leg['line']}: {name(net, leg['sids'][0])} -> "
                  f"{name(net, leg['sids'][-1])}  "
                  f"({stops} stop{'s' if stops != 1 else ''}, "
                  f"{leg['dist_km']:.1f} km, ~{leg['ride_min']:.0f} min)")
            print(f"      crowd now: {cs}/5 ({CROWD_WORDS[cs]})")
            step += 1
        if ti < len(transfers):
            t = transfers[ti]
            ti += 1
            tag = "step-free" if t["accessible"] else "stairs"
            print(f"  {step}. Transfer at {name(net, t['sid'])}: {t['from_line']} -> "
                  f"{t['to_line']}  (~{t['walk_min']:.0f} min walk, "
                  f"{t['walk_m']:.0f} m, {tag})")
            step += 1

    print("-" * 66)
    print(f"  In-train time         : {summ['total_ride_min']:.0f} min")
    print(f"  Transfer walking time : {summ['total_walk_min']:.0f} min")
    print(f"  Total journey time    : ~{summ['total_min']:.0f} min")
    print(f"  Transfers             : {len(transfers)}")
    print(f"  Average crowding      : {summ['avg_crowd_score']}/5 "
          f"({CROWD_WORDS[summ['avg_crowd_score']]})")
    print(f"  Preference cost       : {total_cost:.1f} effective-min "
          f"(lower = better fit to your weights)")
    print("=" * 66 + "\n")


# ===========================================================================
# CONSOLE INPUT
# ===========================================================================
def ask_station(net: ap.Network, prompt: str) -> str:
    while True:
        raw = input(prompt).strip()
        if raw.lower() in ("list", "stations", "?"):
            for lbl in net.all_station_labels():
                print("   -", lbl)
            continue
        sid = net.resolve(raw)
        if sid:
            return sid
        sug = net.suggestions(raw)
        if sug:
            print("  Not found. Did you mean:", "; ".join(sug))
        else:
            print("  Station not found. Type 'list' to see all stations.")


def ask_rating(prompt: str) -> int:
    while True:
        raw = input(prompt).strip()
        if raw.isdigit() and 1 <= int(raw) <= 5:
            return int(raw)
        print("  Enter 1 (don't care) to 5 (very important).")


def ask_yes_no(prompt: str) -> bool:
    while True:
        raw = input(prompt).strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please answer YES or NO.")


# ---------------------------------------------------------------------------
# Accessibility (delta factor) — DEVELOPER-MAINTAINED DATA FILE
# ---------------------------------------------------------------------------
# Accessibility is NOT entered by the end user. A developer maintains
# accessibility.json, a map of station -> step-free (true) / stairs-only
# (false). The program loads it silently at startup.
#
# The file may be keyed by station name (Chinese or romanised) OR by Amap
# station id; both resolve to the same station, e.g.:
#     { "Xuan Wu Men": false, "宣武门": false, "110100023076019": false }
# Stations not listed default to step-free. Generate an editable template
# with:   python3 metro_recommender.py --gen-accessibility
ACC_FILE = "accessibility.json"


def load_accessibility(net: ap.Network, path: str = ACC_FILE) -> dict:
    """Load the developer's accessibility file -> {station_id: step_free}.

    Keys may be station names or station ids. Missing file -> {} (all
    step-free). Unresolved keys are reported and skipped.
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [accessibility] could not read {path}: {exc}")
        return {}

    acc, unresolved = {}, []
    for key, val in raw.items():
        sid = key if key in net.stations else net.resolve(str(key))
        if sid:
            acc[sid] = bool(val)
        else:
            unresolved.append(key)
    if unresolved:
        print(f"  [accessibility] {len(unresolved)} unrecognised station key(s) "
              f"skipped: {', '.join(map(str, unresolved[:8]))}"
              f"{' ...' if len(unresolved) > 8 else ''}")
    return acc


def generate_accessibility_template(net: ap.Network, path: str = ACC_FILE) -> None:
    """Write an editable template: every interchange station -> true.

    Keyed by romanised name for readability; the developer flips the
    stairs-only stations to false.
    """
    interchanges = sorted((st for st in net.stations.values() if len(st.lines) > 1),
                          key=lambda s: s.name_en)
    template = {st.name_en: True for st in interchanges}
    if os.path.exists(path):
        print(f"  {path} already exists — not overwriting. "
              f"Delete it first to regenerate.")
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(template, f, ensure_ascii=False, indent=2)
    print(f"  Wrote template with {len(template)} interchange stations to {path}.\n"
          f"  Edit it: set a station to false where the transfer is stairs-only.")


def main() -> None:
    import sys
    import libcity_loader as lc

    # accessibility table from 本地宝 / MTR would be loaded here, e.g.:
    #   acc = load_accessibility("accessibility.csv")  # {sid: step_free}
    net = ap.build_network(allow_network=True, accessibility=None)
    now = datetime.now(BEIJING_TZ)

    # Developer utility: write an editable accessibility.json template, then exit.
    if "--gen-accessibility" in sys.argv:
        generate_accessibility_template(net)
        return

    # Measured crowding (beta). Point at the LibCity BEIJING_SUBWAY folder via
    # a path argument or the default; falls back to the estimate model if absent.
    data_args = [a for a in sys.argv[1:] if not a.startswith("-")]
    data_dir = data_args[0] if data_args else "raw_data/BEIJING_SUBWAY"
    measured = lc.load_libcity(data_dir, net)
    set_measured(measured)

    # Accessibility (delta) — loaded from the developer-maintained data file.
    acc = load_accessibility(net)
    n_stairs = ap.apply_accessibility(net, acc)

    print("\n" + "#" * 66)
    print("#  Beijing Metro — Preference-Based Route Recommender")
    print("#  " + ap.network_summary(net).replace("\n", "\n#  "))
    print(f"#  Beijing time now: {now:%Y-%m-%d %H:%M} ({now:%A})")
    print(f"#  Crowd data: {crowd_source_label()}")
    if measured is not None:
        print("#  " + measured.summary().replace("\n", "\n#  "))
    else:
        print(f"#  (no LibCity data at '{data_dir}' — using estimate model)")
    if acc:
        print(f"#  Accessibility: loaded {ACC_FILE} "
              f"({n_stairs} interchange station(s) stairs-only)")
    else:
        print(f"#  Accessibility: no {ACC_FILE} found — all stations step-free")
    print("#  (type 'list' at a station prompt to see all stations)")
    print("#" * 66 + "\n")

    start = ask_station(net, "Start station: ")
    end = ask_station(net, "Destination station: ")
    while end == start:
        print("  Destination must differ from start.")
        end = ask_station(net, "Destination station: ")

    print("\nRate each factor 1 (don't care) to 5 (very important):")
    pr_time = ask_rating("  Journey time priority      (1-5): ")
    pr_crowd = ask_rating("  Avoiding crowds priority   (1-5): ")
    pr_walk = ask_rating("  Less transfer walking priority (1-5): ")
    need_acc = ask_yes_no("  Need a step-free / accessible route? (yes/no): ")

    # Start/destination step-free check (only meaningful if entered manually).
    if need_acc:
        for role, sid in (("start", start), ("destination", end)):
            if acc.get(sid) is False:
                print(f"  Warning: {name(net, sid)} ({role}) is marked "
                      f"stairs-only — the station itself may not be step-free.")

    prefs = Preferences(pr_time, pr_crowd, pr_walk, need_acc)
    total_cost, chain = find_route(net, start, end, now, prefs)
    if chain is None:
        print("\n  No route found under your constraints.")
        if need_acc:
            print("  Every option may require a stair-only transfer. Try "
                  "without the accessibility requirement.")
        return

    summ = build_itinerary(net, chain, now)
    print_itinerary(net, start, end, summ, now, prefs, total_cost)


if __name__ == "__main__":
    main()
