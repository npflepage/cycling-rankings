"""
Time-Aware Bradley-Terry export for the static GitHub Pages site.

Runs the same engine as `Cycling_BT_TimeAware.ipynb` and writes 4 JSON files
under ../docs/outputs/ that the landing page consumes:

  - meta.json                 stats bar, last update, race count, tracks
  - rider_timeseries.json     full per-rider μ/σ trajectory (all engines)
  - top5_by_track.json        current top-N per track + composite
  - top5_history.json         top-5 leaderboard at each snapshot (for scrubber)

Usage:
  pip install openskill numpy
  python model/export_for_web.py --data-dir ./data --out-dir ./docs/outputs

The JSON files are small enough that the browser fetches them on load
(~ few hundred KB depending on rider count).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from openskill.models import BradleyTerryPart


# ════════════════════════════════════════════════════════════════════════════
# Engine constants — match the notebook exactly
# ════════════════════════════════════════════════════════════════════════════
MU_INIT      = 25.0
SIGMA_INIT   = 25.0 / 3.0
BETA         = 25.0 / 6.0
KAPPA        = 1e-4
TAU_BASE     = 25.0 / 300.0
WINDOW_SIZE  = 16
TAU_PER_DAY  = 0.02
USE_POINTS_WEIGHTING = True

DISPLAY_BASE  = 1500
DISPLAY_SCALE = 60


# ════════════════════════════════════════════════════════════════════════════
# Engine — verbatim port of the notebook's BTEngine
# ════════════════════════════════════════════════════════════════════════════
class BTEngine:
    def __init__(self, mu_init=MU_INIT, sigma_init=SIGMA_INIT,
                 beta=BETA, kappa=KAPPA, tau_base=TAU_BASE,
                 window_size=WINDOW_SIZE, tau_per_day=TAU_PER_DAY,
                 use_points_weighting=USE_POINTS_WEIGHTING):
        self.model = BradleyTerryPart(
            mu=mu_init, sigma=sigma_init,
            beta=beta, kappa=kappa, tau=tau_base,
            window_size=window_size, limit_sigma=False,
        )
        self.mu_init = mu_init
        self.sigma_init = sigma_init
        self.tau_base = tau_base
        self.tau_per_day = tau_per_day
        self.use_points_weighting = use_points_weighting
        self.ratings = {}
        self.race_counts = {}
        self.last_race_date = {}
        self.history = []

    def _register(self, rider):
        if rider not in self.ratings:
            self.ratings[rider] = self.model.rating(name=rider)
            self.race_counts[rider] = 0

    def _apply_time_decay(self, rider, race_date):
        if rider in self.last_race_date:
            days = max((race_date - self.last_race_date[rider]).days, 0)
            if days > 0:
                r = self.ratings[rider]
                new_sigma_sq = r.sigma ** 2 + (self.tau_per_day ** 2) * days
                self.ratings[rider] = self.model.rating(
                    mu=r.mu, sigma=float(np.sqrt(new_sigma_sq)), name=rider,
                )

    def _race_tau(self, points):
        if not self.use_points_weighting:
            return self.tau_base
        return self.tau_base * float(max(points, 1) / 500.0)

    def process_race(self, race):
        results = race["results"]
        points = race.get("points", 100)
        race_date = datetime.fromisoformat(race["date"])

        for _, rider in results:
            self._register(rider)
            self._apply_time_decay(rider, race_date)

        riders = [r[1] for r in results]
        ranks = [r[0] for r in results]
        teams = [[self.ratings[r]] for r in riders]
        race_tau = self._race_tau(points)
        new_teams = self.model.rate(teams, ranks=ranks, tau=race_tau)

        deltas = {}
        for rider, new_team in zip(riders, new_teams):
            old = self.ratings[rider]
            new = new_team[0]
            deltas[rider] = round(new.mu - old.mu, 4)
            self.ratings[rider] = new
            self.race_counts[rider] += 1
            self.last_race_date[rider] = race_date

        self.history.append({
            "race_id":   race.get("id"),
            "race_name": race["name"],
            "date":      race["date"],
            "category":  race.get("category"),
            "type":      race.get("type"),
            "points":    points,
            "n_riders":  len(results),
            "race_tau":  race_tau,
            "deltas":    deltas,
            "ratings":   {r: (rt.mu, rt.sigma) for r, rt in self.ratings.items()},
        })
        return deltas

    def display_score_from_pair(self, mu, sigma, z=0.0):
        return DISPLAY_BASE + DISPLAY_SCALE * ((mu - z * sigma) - self.mu_init)


# ════════════════════════════════════════════════════════════════════════════
# Multi-track — verbatim port
# ════════════════════════════════════════════════════════════════════════════
BT_TRACKS = {
    "sprint":     lambda r: r["type"] in ["sprint"],
    "TT":         lambda r: r["type"] == "TT",
    "cobbles":    lambda r: r["type"] == "cobbles",
    "punch":      lambda r: r["type"] in ["punch", "mountain"] and r["category"] == "classic",
    "mountain":   lambda r: r["type"] == "mountain",
    "GC":         lambda r: r["category"] == "GC",
    "stage_race": lambda r: r["category"] in ["stage", "GC"] and r["type"] != "TT",
    "classics":   lambda r: r["category"] == "classic",
}

COMPOSITE_TRACKS = ["sprint", "punch", "cobbles", "classics",
                    "mountain", "stage_race", "TT"]


class BTEngineMultiTrack:
    def __init__(self, **kwargs):
        self.tracks = list(BT_TRACKS.keys())
        self.engines = {t: BTEngine(**kwargs) for t in self.tracks}

    def process_race(self, race):
        for track, condition in BT_TRACKS.items():
            if condition(race):
                self.engines[track].process_race(race)


# ════════════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════════════
def load_all_races(data_dir: Path) -> list[dict]:
    races = []
    for f in sorted(data_dir.glob("*.json")):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        races.extend(data.get("races", []))
    races.sort(key=lambda r: r["date"])
    return races


# ════════════════════════════════════════════════════════════════════════════
# Export helpers
# ════════════════════════════════════════════════════════════════════════════
def elo(mu, sigma, z=0.0):
    return DISPLAY_BASE + DISPLAY_SCALE * ((mu - z * sigma) - MU_INIT)


def export_meta(races, engine, engine_mt, out_dir, min_races):
    type_counts = Counter(r.get("type", "unknown") for r in races)
    cat_counts = Counter(r.get("category", "unknown") for r in races)

    per_track_riders = {
        t: sum(1 for n in eng.race_counts.values() if n >= min_races)
        for t, eng in engine_mt.engines.items()
    }

    qualifying_riders = sum(1 for n in engine.race_counts.values()
                            if n >= min_races)

    meta = {
        "first_race_date": races[0]["date"],
        "last_race_date":  races[-1]["date"],
        "total_races":     len(races),
        "total_riders":    len(engine.ratings),
        "qualifying_riders": qualifying_riders,
        "tracks":          list(BT_TRACKS.keys()),
        "type_counts":     dict(type_counts),
        "category_counts": dict(cat_counts),
        "per_track_riders": per_track_riders,
        "generated_at":    datetime.utcnow().isoformat() + "Z",
        "params": {
            "mu_init": MU_INIT, "sigma_init": SIGMA_INIT,
            "beta": BETA, "tau_base": TAU_BASE,
            "tau_per_day": TAU_PER_DAY, "window_size": WINDOW_SIZE,
        },
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ meta.json  ({len(races)} races, {len(engine.ratings)} riders)")


def export_rider_timeseries(engine, engine_mt, out_dir, min_races):
    """
    For each rider that has >= min_races races somewhere, store their elo
    trajectory over time for the overall engine + each track engine.

    Format:
      {
        "Pogačar Tadej": {
          "races": ["2019-..", ...],
          "ALL":  [{"d": "2019-01-15", "elo": 1502, "low": 1498, "high": 1506, "rode": true}, ...],
          "GC":   [...],
          ...
        }
      }

    `rode` = the rider raced this specific race (true vs carried-forward).
    """
    qualifying = {r for r, n in engine.race_counts.items() if n >= min_races}
    # also include riders that qualify on any single track
    for eng in engine_mt.engines.values():
        for r, n in eng.race_counts.items():
            if n >= min_races:
                qualifying.add(r)

    def trajectory(eng, rider):
        """μ/σ at every snapshot of this engine; carry forward when absent."""
        traj = []
        last_mu, last_sg = MU_INIT, SIGMA_INIT
        for snap in eng.history:
            rode = rider in snap["deltas"]
            if rider in snap["ratings"]:
                mu, sg = snap["ratings"][rider]
                last_mu, last_sg = mu, sg
            else:
                mu, sg = last_mu, last_sg
            traj.append({
                "d":    snap["date"],
                "elo":  round(elo(mu, sg, 0.0), 1),
                "low":  round(elo(mu, sg, 2.0), 1),
                "high": round(elo(mu, -sg, 2.0), 1),
                "rode": rode,
            })
        return traj

    out = {}
    for rider in sorted(qualifying):
        out[rider] = {"ALL": trajectory(engine, rider)}
        for track in engine_mt.tracks:
            eng = engine_mt.engines[track]
            if rider in eng.ratings:
                out[rider][track] = trajectory(eng, rider)

    with open(out_dir / "rider_timeseries.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"  ✓ rider_timeseries.json  ({len(out)} riders)")


def export_top5_history(engine, engine_mt, out_dir, min_races, top_n=10,
                        downsample_every_n_races=1):
    """
    For each engine (ALL + per-track), snapshot the top-N (by μ) at every
    race the engine saw. The web page uses this for the leaderboard scrubber.

    Each entry: {"date": ..., "race": ..., "top": [{name, elo, races}, ...]}.
    """
    def top_at_snapshot(snap, race_counts_at_snapshot):
        items = []
        for rider, (mu, sg) in snap["ratings"].items():
            n = race_counts_at_snapshot.get(rider, 0)
            if n < min_races:
                continue
            items.append((rider, mu, sg, n))
        items.sort(key=lambda x: -x[1])  # by μ
        return [
            {"name": r, "elo": round(elo(mu, sg, 0.0), 1),
             "low": round(elo(mu, sg, 2.0), 1), "races": n}
            for r, mu, sg, n in items[:top_n]
        ]

    def history_for(eng):
        running_count = {}
        out = []
        for i, snap in enumerate(eng.history):
            for rider in snap["deltas"]:
                running_count[rider] = running_count.get(rider, 0) + 1
            if i % downsample_every_n_races != 0 and i != len(eng.history) - 1:
                continue
            out.append({
                "date":  snap["date"],
                "race":  snap["race_name"],
                "top":   top_at_snapshot(snap, running_count),
            })
        return out

    out = {"ALL": history_for(engine)}
    for track in engine_mt.tracks:
        out[track] = history_for(engine_mt.engines[track])

    with open(out_dir / "top5_history.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"  ✓ top5_history.json  ({len(out)} engines)")


def export_current_top5(engine, engine_mt, out_dir, min_races, top_n=20):
    """Snapshot of final standings for fast initial render."""
    def current_top(eng):
        rows = []
        for rider, rating in eng.ratings.items():
            n = eng.race_counts.get(rider, 0)
            if n < min_races:
                continue
            rows.append({
                "name":  rider,
                "elo":   round(elo(rating.mu, rating.sigma, 0.0), 1),
                "low":   round(elo(rating.mu, rating.sigma, 2.0), 1),
                "mu":    round(rating.mu, 3),
                "sigma": round(rating.sigma, 3),
                "races": n,
            })
        rows.sort(key=lambda x: -x["elo"])
        return rows[:top_n]

    # Composite: z-scored, inverse-sigma weighted (matches notebook)
    track_data = {}
    for track in COMPOSITE_TRACKS:
        eng = engine_mt.engines[track]
        track_data[track] = {
            r: (rt.mu, rt.sigma, eng.race_counts.get(r, 0))
            for r, rt in eng.ratings.items()
            if eng.race_counts.get(r, 0) >= min_races
        }
    track_stats = {}
    for track, riders in track_data.items():
        mus = [v[0] for v in riders.values()]
        if len(mus) >= 2:
            track_stats[track] = (float(np.mean(mus)), float(np.std(mus)))

    composite_rows = []
    all_riders = set(r for d in track_data.values() for r in d)
    for rider in all_riders:
        ws, wt, cats = 0.0, 0.0, 0
        for track in COMPOSITE_TRACKS:
            if rider not in track_data[track] or track not in track_stats:
                continue
            mu, sigma, _ = track_data[track][rider]
            t_mean, t_std = track_stats[track]
            if t_std == 0:
                continue
            w = 1.0 / sigma
            ws += w * ((mu - t_mean) / t_std)
            wt += w
            cats += 1
        if cats and wt:
            composite_rows.append({
                "name":  rider,
                "composite": round(ws / wt, 4),
                "cats":  cats,
            })
    composite_rows.sort(key=lambda x: -x["composite"])

    out = {
        "ALL": current_top(engine),
        "composite": composite_rows[:top_n],
        **{t: current_top(engine_mt.engines[t]) for t in engine_mt.tracks},
    }
    with open(out_dir / "current_top5.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"  ✓ current_top5.json  ({len(out)} engines + composite)")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data",
                        help="Directory containing 2019.json .. 2026.json")
    parser.add_argument("--out-dir", default="./docs/outputs",
                        help="Where to write the JSON exports")
    parser.add_argument("--min-races", type=int, default=2,
                        help="Minimum race count for a rider to be exported")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Snapshot every Nth race in top5_history (1 = all)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Loading races from {data_dir}/ ...")
    races = load_all_races(data_dir)
    if not races:
        raise SystemExit(f"No races found in {data_dir}/. Expected *.json files.")
    print(f"   loaded {len(races)} races from "
          f"{races[0]['date']} → {races[-1]['date']}")

    print("⚙️  Running cumulative engine ...")
    engine = BTEngine()
    for race in races:
        engine.process_race(race)

    print("⚙️  Running multi-track engine ...")
    engine_mt = BTEngineMultiTrack()
    for race in races:
        engine_mt.process_race(race)

    print(f"💾 Writing exports to {out_dir}/ ...")
    export_meta(races, engine, engine_mt, out_dir, args.min_races)
    export_rider_timeseries(engine, engine_mt, out_dir, args.min_races)
    export_top5_history(engine, engine_mt, out_dir,
                        args.min_races, top_n=10,
                        downsample_every_n_races=args.downsample)
    export_current_top5(engine, engine_mt, out_dir, args.min_races, top_n=20)
    print("\n✅ Done.")


if __name__ == "__main__":
    main()
