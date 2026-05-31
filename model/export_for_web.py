"""
Time-Aware Bradley-Terry export for the static GitHub Pages site.

Runs the same engine as `Cycling_BT_TimeAware.ipynb` and writes 4 JSON files
under ../docs/outputs/ that the landing page consumes:

  - meta.json               stats bar, last update, race count, tracks, params
  - rider_timeseries.json   per-rider (mu, sigma) trajectory (all engines)
                            + per-rider composite z-score trajectory
  - top_history.json        top-N per track at each snapshot (normal + safe)
  - hall_of_fame.json       per track: #1 reigns + all-time peak (GOAT),
                            for both the normal (mu) and safe (mu-2sigma)
                            metric. Not affected by the timeline slider.

Design note — what is precomputed vs. computed in the browser:

  * Everything that depends only on (track, safe-metric) and is expensive
    to recompute (per-snapshot rankings, #1 reigns, all-time peaks, the
    cross-discipline composite) is baked here into static JSON.
  * The timeseries stores raw (mu, sigma) instead of display ELO so the
    page can (a) redraw instantly when the "safe" toggle flips and
    (b) run the head-to-head win-probability calculator *online* for any
    pair of riders without us having to precompute the O(n^2) matrix.

Usage:
  pip install openskill numpy
  python model/export_for_web.py --data-dir ./data --out-dir ./docs/outputs
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

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

    def apply_inactivity_decay(self, reference_date, cap_at_init=True,
                               years_to_cap=3.0):
        """Catch-up sigma inflation for inactive riders.

        The engine only applies time-decay lazily, when a rider *next* races.
        Riders who stopped racing therefore keep a sigma frozen at their last
        race, giving them an artificially high safe-Elo (mu - z*sigma) forever.

        This walks every rider's last-known rating forward from their last race
        in THIS track to `reference_date`, growing sigma LINEARLY toward the
        sigma_init cap. We use a dedicated linear rate rather than the engine's
        between-race tau_per_day, because that rate (tuned for active riders) is
        far too slow — with the quadratic sigma**2 += tau**2*days rule it would
        take centuries to materially inflate a retired rider.

        `years_to_cap` sets how long of inactivity drives a typical active rider
        (sigma ~= a few units) essentially all the way to the cap. mu is left
        unchanged (no new information).

        Mutates self.ratings in place, then appends a synthetic snapshot to
        self.history so that all exports (top_history, timeseries, hall_of_fame)
        see the decayed ratings as the current/final state.
        """
        rate_per_day = self.sigma_init / (years_to_cap * 365.0)
        for rider, last_date in self.last_race_date.items():
            days = max((reference_date - last_date).days, 0)
            if days == 0:
                continue
            r = self.ratings[rider]
            new_sigma = r.sigma + rate_per_day * days
            if cap_at_init:
                new_sigma = min(new_sigma, self.sigma_init)
            self.ratings[rider] = self.model.rating(
                mu=r.mu, sigma=float(new_sigma), name=rider,
            )

        # Append a synthetic snapshot so exports see the decayed ratings.
        # deltas is empty (no race happened); race counts are unchanged so
        # min_races filtering continues to work correctly.
        self.history.append({
            "race_id":   None,
            "race_name": "Current standings (inactivity decay applied)",
            "date":      reference_date.strftime("%Y-%m-%d"),
            "category":  None,
            "type":      None,
            "points":    None,
            "n_riders":  0,
            "race_tau":  None,
            "deltas":    {},
            "ratings":   {r: (rt.mu, rt.sigma) for r, rt in self.ratings.items()},
        })


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

# ── Composite ("Cumulative") track ──────────────────────────────────────────
# The 8 BT tracks overlap (e.g. `classics` ⊃ `punch`, `stage_race` ⊃ `GC`).
# To build a single cross-discipline score we keep only the disjoint *type*
# tracks the user asked for and drop the category aggregates that would
# double-count.  GC is up-weighted: holding form across a 3-week tour is a
# stronger signal of all-round quality than picking up isolated stage wins.
COMPOSITE_TRACKS  = ["cobbles", "punch", "mountain", "sprint", "GC"]
COMPOSITE_WEIGHTS = {
    "cobbles": 1.0, "punch": 1.0, "mountain": 1.0, "sprint": 1.0,
    "GC": 1.75,
}


class BTEngineMultiTrack:
    def __init__(self, **kwargs):
        self.tracks = list(BT_TRACKS.keys())
        self.engines = {t: BTEngine(**kwargs) for t in self.tracks}

    def process_race(self, race):
        for track, condition in BT_TRACKS.items():
            if condition(race):
                self.engines[track].process_race(race)

    def apply_inactivity_decay(self, cap_at_init=True, years_to_cap=3.0):
        """Apply catch-up sigma inflation to every track's final ratings.

        IMPORTANT: each track uses its OWN latest race as the reference "now",
        i.e. the most recent race of that track type — NOT the latest race
        across all tracks. A rider's sprint sigma is inflated relative to the
        last sprint in the dataset; their mountain sigma relative to the last
        mountain race; and so on. A climber who quit sprinting in 2019 but still
        climbs in 2024 thus has an inflated sprint sigma and a fresh mountain one.
        """
        for track, eng in self.engines.items():
            if not eng.last_race_date:
                continue
            reference_date = max(eng.last_race_date.values())
            eng.apply_inactivity_decay(reference_date, cap_at_init=cap_at_init,
                                       years_to_cap=years_to_cap)


# ════════════════════════════════════════════════════════════════════════════
# Composite trajectory — cross-discipline z-score over time
# ════════════════════════════════════════════════════════════════════════════
def compute_composite_history(engine_mt, tracks=COMPOSITE_TRACKS,
                              weights=COMPOSITE_WEIGHTS,
                              safe=False, min_races=2):
    """Reconstruct the composite score for every rider at each point on a
    unified timeline (the union of all per-track snapshots, ordered by date).

    Recipe at every global snapshot:
      - effective skill r = mu - z*sigma   (z = 2 in safe mode, else 0)
      - z-score r against that track's *current* field of qualifiers
      - weight by (1/sigma) * track_weight  (GC weighted up)
      - average the weighted z-scores over the tracks the rider qualifies in

    Returns a date-ordered list of events:
      {idx, date, race_name, track, raced:set, qual_races:{r:int},
       composites:{rider: value}}
    """
    z = 2.0 if safe else 0.0

    events = []
    for track in tracks:
        eng = engine_mt.engines[track]
        for i, snap in enumerate(eng.history):
            events.append((snap["date"], track, i))
    events.sort(key=lambda e: e[0])

    cur_ratings   = {t: {} for t in tracks}   # {track: {rider: (mu, sigma)}}
    cur_racecount = {t: {} for t in tracks}   # {track: {rider: int}}
    history = []

    for global_idx, (date, track, snap_i) in enumerate(events):
        eng  = engine_mt.engines[track]
        snap = eng.history[snap_i]
        raced_here = set(snap["deltas"].keys())

        for rider in snap["deltas"]:
            cur_racecount[track][rider] = cur_racecount[track].get(rider, 0) + 1
        for rider, (mu, sigma) in snap["ratings"].items():
            cur_ratings[track][rider] = (mu, sigma)

        # ── per-track field stats on the effective (safe-aware) skill ──
        track_stats = {}
        for t in tracks:
            effs = [mu - z * sg
                    for r, (mu, sg) in cur_ratings[t].items()
                    if cur_racecount[t].get(r, 0) >= min_races]
            track_stats[t] = ((float(np.mean(effs)), float(np.std(effs)))
                              if len(effs) >= 2 else None)

        all_riders = set(r for t in tracks for r in cur_ratings[t])
        composites = {}
        qual_races = {}
        for rider in all_riders:
            ws = wt = 0.0
            cats = 0
            total_races = 0
            for t in tracks:
                if rider not in cur_ratings[t]:
                    continue
                n = cur_racecount[t].get(rider, 0)
                if n < min_races or track_stats[t] is None:
                    continue
                mu, sigma = cur_ratings[t][rider]
                if sigma == 0:
                    continue
                t_mean, t_std = track_stats[t]
                if t_std == 0:
                    continue
                eff = mu - z * sigma
                w = (1.0 / sigma) * weights.get(t, 1.0)
                ws += w * ((eff - t_mean) / t_std)
                wt += w
                cats += 1
                total_races += n
            if cats == 0 or wt == 0:
                continue
            composites[rider] = ws / wt
            qual_races[rider] = total_races

        history.append({
            "idx":        global_idx,
            "date":       date,
            "race_name":  snap["race_name"],
            "track":      track,
            "raced":      raced_here,
            "qual_races": qual_races,
            "composites": composites,
        })

    return history


# ════════════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════════════
def load_all_races(data_dir: Path) -> list[dict]:
    races = []
    for f in sorted(data_dir.glob("*.json")):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        races.extend(data.get("races", []))
    races = [r for r in races if r.get("date") is not None and r.get("points") is not None]
    races.sort(key=lambda r: r["date"])
    return races


# ════════════════════════════════════════════════════════════════════════════
# Sprint-relabel diagnostic
# ════════════════════════════════════════════════════════════════════════════
def suggest_sprint_relabels(races, engine_mt, *,
                            top_k=20, min_races=2, margin=1.0, z_safe=2.0):
    """For each race currently typed 'sprint', look at the top finishers and
    decide whether their sprint / punch / mountain safe-Elos suggest the race
    was actually a punch or mountain race that lost its PCS profile metadata.

    Scoring per race:
      - take top_k finishers
      - for each, compute safe-Elo (mu - z_safe*sigma) in {sprint, punch, mountain}
        using the *final* ratings from engine_mt (single-pass diagnostic)
      - z-score each rider's safe-Elo *within that track's final field*
        (riders qualified with >= min_races in that track)
      - weight rider by 1/rank, sum z-scores per discipline
      - propose relabel to argmax discipline if (winner_score - sprint_score) >= margin

    Prints a table of proposed relabels. Does NOT modify `races`.
    """
    tracks = ["sprint", "punch", "mountain"]

    # Build the per-track z-score lookup using *final* engine state.
    track_z = {}
    for t in tracks:
        eng = engine_mt.engines[t]
        qualified = {
            r: (rt.mu - z_safe * rt.sigma)
            for r, rt in eng.ratings.items()
            if eng.race_counts.get(r, 0) >= min_races
        }
        if not qualified:
            track_z[t] = {}
            continue
        vals = np.array(list(qualified.values()), dtype=float)
        mean, std = float(vals.mean()), float(vals.std())
        if std == 0.0:
            track_z[t] = {r: 0.0 for r in qualified}
        else:
            track_z[t] = {r: (v - mean) / std for r, v in qualified.items()}

    suggestions = []
    for race in races:
        if race.get("type") != "sprint":
            continue

        results = race.get("results", [])
        # `results` is a list of [rank, rider]; take top_k by rank
        top = sorted(results, key=lambda x: x[0])[:top_k]
        if len(top) < 3:
            continue

        scores = {t: 0.0 for t in tracks}
        contributing = {t: 0 for t in tracks}
        for rank, rider in top:
            w = 1.0 / float(rank)
            for t in tracks:
                if rider in track_z[t]:
                    scores[t] += w * track_z[t][rider]
                    contributing[t] += 1

        # Skip races where almost no top rider qualifies anywhere — too noisy.
        if max(contributing.values()) < 3:
            continue

        winner_disc = max(scores, key=scores.get)
        if winner_disc == "sprint":
            continue
        if (scores[winner_disc] - scores["sprint"]) < margin:
            continue

        top3 = [rider for _, rider in top[:3]]
        suggestions.append({
            "date":        race["date"],
            "name":        race.get("name", "?"),
            "current":     "sprint",
            "suggested":   winner_disc,
            "top3":        top3,
            "scores":      scores,
            "gap":         scores[winner_disc] - scores["sprint"],
        })

    suggestions.sort(key=lambda s: -s["gap"])

    print(f"\n🔍 Sprint-relabel suggestions "
          f"(top_k={top_k}, margin={margin}, z_safe={z_safe})")
    print(f"   found {len(suggestions)} candidate(s)\n")
    if not suggestions:
        return suggestions

    print(f"   {'date':<12} {'→ new':<9} {'gap':>5}  race  |  top 3")
    print(f"   {'-'*12} {'-'*9} {'-'*5}  " + "-"*60)
    for s in suggestions:
        top3_str = ", ".join(s["top3"])
        print(f"   {s['date']:<12} {s['suggested']:<9} "
              f"{s['gap']:>5.2f}  {s['name']}  |  {top3_str}")

    return suggestions


# ════════════════════════════════════════════════════════════════════════════
# Export helpers
# ════════════════════════════════════════════════════════════════════════════
def elo(mu, sigma, z=0.0):
    return DISPLAY_BASE + DISPLAY_SCALE * ((mu - z * sigma) - MU_INIT)


def export_meta(races, engine, engine_mt, out_dir, min_races):
    type_counts = Counter(r.get("type", "unknown") for r in races)
    cat_counts  = Counter(r.get("category", "unknown") for r in races)

    per_track_riders = {
        t: sum(1 for n in eng.race_counts.values() if n >= min_races)
        for t, eng in engine_mt.engines.items()
    }
    qualifying_riders = sum(1 for n in engine.race_counts.values()
                            if n >= min_races)

    meta = {
        "first_race_date":   races[0]["date"],
        "last_race_date":    races[-1]["date"],
        "total_races":       len(races),
        "total_riders":      len(engine.ratings),
        "qualifying_riders": qualifying_riders,
        "tracks":            list(BT_TRACKS.keys()),
        "composite_tracks":  COMPOSITE_TRACKS,
        "composite_weights": COMPOSITE_WEIGHTS,
        "type_counts":       dict(type_counts),
        "category_counts":   dict(cat_counts),
        "per_track_riders":  per_track_riders,
        "generated_at":      datetime.utcnow().isoformat() + "Z",
        "params": {
            "mu_init": MU_INIT, "sigma_init": SIGMA_INIT,
            "beta": BETA, "tau_base": TAU_BASE,
            "tau_per_day": TAU_PER_DAY, "window_size": WINDOW_SIZE,
            "display_base": DISPLAY_BASE, "display_scale": DISPLAY_SCALE,
        },
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ meta.json  ({len(races)} races, {len(engine.ratings)} riders)")


def export_rider_timeseries(engine, engine_mt, comp_norm, comp_safe,
                             out_dir, min_races):
    """Per-rider trajectory.

    For ALL + each of the 8 tracks we store raw (mu, sigma) at every race the
    rider actually started; the browser turns that into the ELO / safe-ELO
    line *and* feeds the head-to-head calculator. For the composite track we
    store the cross-discipline z-score in normal and safe form.

    Point shape:
      tracks/ALL : {"d": "2019-..", "m": <mu>, "s": <sigma>}
      composite  : {"d": "2019-..", "v": <z>, "vs": <z safe>}
    """
    qualifying = {r for r, n in engine.race_counts.items() if n >= min_races}
    for eng in engine_mt.engines.values():
        for r, n in eng.race_counts.items():
            if n >= min_races:
                qualifying.add(r)

    def trajectory(eng, rider):
        traj, last = [], None
        for snap in eng.history:
            if rider in snap["ratings"]:
                last = snap["ratings"][rider]
            if rider in snap["deltas"] and last is not None:
                mu, sg = last
                traj.append({"d": snap["date"],
                             "m": round(float(mu), 5),
                             "s": round(float(sg), 5)})
        return traj

    # composite series, keyed by rider, aligned by event index
    comp_series = {}
    safe_by_idx = {h["idx"]: h["composites"] for h in comp_safe}
    for h in comp_norm:
        s_comp = safe_by_idx.get(h["idx"], {})
        for rider in h["raced"]:
            if rider not in h["composites"]:
                continue
            comp_series.setdefault(rider, []).append({
                "d":  h["date"],
                "v":  round(float(h["composites"][rider]), 5),
                "vs": round(float(s_comp.get(rider, h["composites"][rider])), 5),
            })

    out = {}
    for rider in sorted(qualifying):
        rec = {"ALL": trajectory(engine, rider)}
        for track in engine_mt.tracks:
            eng = engine_mt.engines[track]
            if rider in eng.ratings:
                rec[track] = trajectory(eng, rider)
        if rider in comp_series:
            rec["composite"] = comp_series[rider]
        out[rider] = rec

    with open(out_dir / "rider_timeseries.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"  ✓ rider_timeseries.json  ({len(out)} riders)")


# ── shared: rank a normal BT engine snapshot ────────────────────────────────
def _ranked_rows(snap, race_counts, min_races):
    rows = []
    for rider, (mu, sg) in snap["ratings"].items():
        n = race_counts.get(rider, 0)
        if n < min_races:
            continue
        rows.append((rider,
                     round(elo(mu, sg, 0.0), 1),
                     round(elo(mu, sg, 2.0), 1),
                     n))
    return rows


def export_top_history(engine, engine_mt, comp_norm, comp_safe,
                       out_dir, min_races, top_n=10,
                       downsample_every_n_races=1):
    """Per-engine top-N at every (optionally down-sampled) snapshot.

    Each snapshot carries BOTH orderings so the page can switch the "safe"
    toggle with no recompute:
      {"date","race","top":[{name,elo,low,races}], "top_safe":[...]}
    Composite snapshots use {name, val, races} instead of elo/low.
    """
    def history_for(eng):
        running, out = {}, []
        n_snaps = len(eng.history)
        for i, snap in enumerate(eng.history):
            for rider in snap["deltas"]:
                running[rider] = running.get(rider, 0) + 1
            if i % downsample_every_n_races != 0 and i != n_snaps - 1:
                continue
            rows = _ranked_rows(snap, running, min_races)
            by_mu  = sorted(rows, key=lambda x: -x[1])[:top_n]
            by_low = sorted(rows, key=lambda x: -x[2])[:top_n]
            out.append({
                "date": snap["date"], "race": snap["race_name"],
                "top":      [{"name": r, "elo": e, "low": lo, "races": n}
                             for r, e, lo, n in by_mu],
                "top_safe": [{"name": r, "elo": e, "low": lo, "races": n}
                             for r, e, lo, n in by_low],
            })
        return out

    def composite_history():
        safe_by_idx = {h["idx"]: h for h in comp_safe}
        out = []
        n_ev = len(comp_norm)
        for i, h in enumerate(comp_norm):
            if i % downsample_every_n_races != 0 and i != n_ev - 1:
                continue
            sh = safe_by_idx.get(h["idx"], h)

            def pack(hist):
                items = sorted(hist["composites"].items(),
                               key=lambda kv: -kv[1])[:top_n]
                return [{"name": r,
                         "val": round(float(v), 4),
                         "races": hist["qual_races"].get(r, 0)}
                        for r, v in items]

            out.append({"date": h["date"], "race": h["race_name"],
                        "top": pack(h), "top_safe": pack(sh)})
        return out

    out = {"ALL": history_for(engine)}
    for track in engine_mt.tracks:
        out[track] = history_for(engine_mt.engines[track])
    out["composite"] = composite_history()

    with open(out_dir / "top_history.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"  ✓ top_history.json  ({len(out)} engines)")


def export_hall_of_fame(engine, engine_mt, comp_norm, comp_safe,
                        out_dir, min_races, goat_n=10):
    """Per engine, for BOTH the normal (mu) and safe (mu-2sigma) metric:

      reigns : merged spells where a rider held #1, [{name,start,end,days}]
      goat   : the `goat_n` highest career peaks ever recorded in the time
               frame, [{name, peak, date, races}]

    Computed from the FULL history (independent of the scrubber down-sample)
    and never filtered by the timeline slider.
    """
    def days_between(a, b):
        return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).days

    def merge_reigns(leader_seq):
        # leader_seq : [(date, name)] in chronological order
        reigns = []
        for date, name in leader_seq:
            if name is None:
                continue
            if reigns and reigns[-1]["name"] == name:
                reigns[-1]["end"] = date
            else:
                reigns.append({"name": name, "start": date, "end": date})
        for r in reigns:
            r["days"] = max(days_between(r["start"], r["end"]), 0)
        return reigns

    def hof_for_engine(eng):
        running = {}
        lead_norm, lead_safe = [], []
        peak_norm, peak_safe = {}, {}   # rider -> (val, date, races)
        for snap in eng.history:
            for rider in snap["deltas"]:
                running[rider] = running.get(rider, 0) + 1
            rows = _ranked_rows(snap, running, min_races)
            if not rows:
                lead_norm.append((snap["date"], None))
                lead_safe.append((snap["date"], None))
                continue
            top_mu  = max(rows, key=lambda x: x[1])
            top_low = max(rows, key=lambda x: x[2])
            lead_norm.append((snap["date"], top_mu[0]))
            lead_safe.append((snap["date"], top_low[0]))
            for rider, e, lo, n in rows:
                if rider not in peak_norm or e > peak_norm[rider][0]:
                    peak_norm[rider] = (e, snap["date"], n)
                if rider not in peak_safe or lo > peak_safe[rider][0]:
                    peak_safe[rider] = (lo, snap["date"], n)

        def goat(peak):
            rows = sorted(peak.items(), key=lambda kv: -kv[1][0])[:goat_n]
            return [{"name": r, "peak": round(v, 1), "date": d, "races": n}
                    for r, (v, d, n) in rows]

        return {
            "normal": {"reigns": merge_reigns(lead_norm),
                       "goat":   goat(peak_norm)},
            "safe":   {"reigns": merge_reigns(lead_safe),
                       "goat":   goat(peak_safe)},
        }

    def hof_composite():
        safe_by_idx = {h["idx"]: h for h in comp_safe}
        lead_norm, lead_safe = [], []
        peak_norm, peak_safe = {}, {}
        for h in comp_norm:
            sh = safe_by_idx.get(h["idx"], h)
            if h["composites"]:
                bn = max(h["composites"].items(), key=lambda kv: kv[1])
                lead_norm.append((h["date"], bn[0]))
                for r, v in h["composites"].items():
                    if r not in peak_norm or v > peak_norm[r][0]:
                        peak_norm[r] = (v, h["date"], h["qual_races"].get(r, 0))
            else:
                lead_norm.append((h["date"], None))
            if sh["composites"]:
                bs = max(sh["composites"].items(), key=lambda kv: kv[1])
                lead_safe.append((h["date"], bs[0]))
                for r, v in sh["composites"].items():
                    if r not in peak_safe or v > peak_safe[r][0]:
                        peak_safe[r] = (v, h["date"], sh["qual_races"].get(r, 0))
            else:
                lead_safe.append((h["date"], None))

        def goat(peak):
            rows = sorted(peak.items(), key=lambda kv: -kv[1][0])[:goat_n]
            return [{"name": r, "peak": round(v, 3), "date": d, "races": n}
                    for r, (v, d, n) in rows]

        return {
            "normal": {"reigns": merge_reigns(lead_norm),
                       "goat":   goat(peak_norm)},
            "safe":   {"reigns": merge_reigns(lead_safe),
                       "goat":   goat(peak_safe)},
        }

    out = {"ALL": hof_for_engine(engine)}
    for track in engine_mt.tracks:
        out[track] = hof_for_engine(engine_mt.engines[track])
    out["composite"] = hof_composite()

    with open(out_dir / "hall_of_fame.json", "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"  ✓ hall_of_fame.json  ({len(out)} engines)")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out-dir", default="./docs/outputs")
    parser.add_argument("--min-races", type=int, default=2)
    parser.add_argument("--downsample", type=int, default=1,
                        help="Snapshot every Nth race in top_history (1 = all)")
    parser.add_argument("--relabel-iters", type=int, default=1,
                        help="Number of relabel iterations to apply (default 1). "
                             "One extra diagnostic-only pass always runs after.")
    parser.add_argument("--decay-years-to-cap", type=float, default=3.0,
                        help="Years of inactivity that inflate a rider's sigma "
                             "to the sigma_init cap, dropping them out of current "
                             "standings. Set to 0 to disable inactivity decay.")
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

    # Races we know are genuine sprints despite the heuristic flagging them.
    # Format: (date, race_name) — must match exactly what's in the JSON data.
    RELABEL_EXCLUSIONS = {
        ("2020-09-04", "Tour de France Stage 7"),    # flat sprint to Lavaur, won by van Aert in a bunch
        ("2015-05-03", "Tour de Romandie Stage 6"),  # ITT finale, Tony Martin
        ("2017-06-12", "Tour de Suisse Stage 3"),    # sprint stage to Bern, Sagan/Matthews/Degenkolb
        ("2017-07-18", "Tour de France Stage 16"),   # reduced bunch sprint to Romans-sur-Isère
    }

    n_iters = max(0, args.relabel_iters)
    total_applied = 0

    for iteration in range(n_iters):
        pass_label = f"Pass {iteration + 1}/{n_iters}"
        print(f"\n🔍 {pass_label}: detecting mislabeled sprint races ...")
        suggestions = suggest_sprint_relabels(races, engine_mt,
                                              min_races=args.min_races)

        # Filter out known false positives
        suggestions = [s for s in suggestions
                       if (s["date"], s["name"]) not in RELABEL_EXCLUSIONS]

        if not suggestions:
            print(f"   No relabels to apply in {pass_label} — stopping early.")
            break

        relabel_key = {(s["date"], s["name"]): s["suggested"] for s in suggestions}
        n_applied = 0
        for race in races:
            key = (race.get("date"), race.get("name"))
            if key in relabel_key:
                race["type"] = relabel_key[key]
                n_applied += 1
        total_applied += n_applied
        print(f"\n✏️  {pass_label}: applied {n_applied} relabel(s) "
              f"({total_applied} total). Re-running multi-track engine ...")

        engine_mt = BTEngineMultiTrack()
        for race in races:
            engine_mt.process_race(race)

    # Always run one final diagnostic-only pass (results not applied to data).
    diag_label = f"Pass {n_iters + 1} (diagnostic only — not applied)"
    print(f"\n🔍 {diag_label} ...")
    suggest_sprint_relabels(races, engine_mt, min_races=args.min_races)

    # Inactivity decay: inflate sigma for riders who stopped racing, so retired
    # riders fall out of the *current* safe-Elo standings. Applied per-track
    # using each track's own latest race as the reference. This runs AFTER the
    # relabel diagnostic (which must see un-inflated ratings). Each call appends
    # a synthetic snapshot so that compute_composite_history and all exports see
    # the decayed ratings as the final/current state.
    if args.decay_years_to_cap and args.decay_years_to_cap > 0:
        print(f"⌛ Applying inactivity decay to current standings "
              f"(cap reached after ~{args.decay_years_to_cap:g} yr inactive) ...")
        engine.apply_inactivity_decay(
            max(engine.last_race_date.values()),
            years_to_cap=args.decay_years_to_cap)
        engine_mt.apply_inactivity_decay(
            cap_at_init=True, years_to_cap=args.decay_years_to_cap)
    else:
        print("⌛ Inactivity decay disabled (--decay-years-to-cap 0).")

    print("⚙️  Computing composite (cumulative) trajectory ...")
    comp_norm = compute_composite_history(engine_mt, safe=False,
                                          min_races=args.min_races)
    comp_safe = compute_composite_history(engine_mt, safe=True,
                                          min_races=args.min_races)

    print(f"💾 Writing exports to {out_dir}/ ...")
    export_meta(races, engine, engine_mt, out_dir, args.min_races)
    export_rider_timeseries(engine, engine_mt, comp_norm, comp_safe,
                            out_dir, args.min_races)
    export_top_history(engine, engine_mt, comp_norm, comp_safe,
                       out_dir, args.min_races, top_n=10,
                       downsample_every_n_races=args.downsample)
    export_hall_of_fame(engine, engine_mt, comp_norm, comp_safe,
                        out_dir, args.min_races, goat_n=10)
    print("\n✅ Done.")


if __name__ == "__main__":
    main()