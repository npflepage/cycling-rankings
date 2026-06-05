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

Race-importance K-scaling (replaces the old tau-based points weighting):

  After openskill computes the raw BT update, we intercept the (mu, sigma)
  deltas and re-scale them by a per-race importance factor K = points / K_ANCHOR:

    Δμ_i        ← K        · Ω_i · (S_i − P_i)   [full K on skill update]
    Δσ²_i       ← sqrt(K)  · Ω_i · σ²_i · γ_i    [dampened K on uncertainty]

  K = 1 at the anchor race (a Vuelta/Giro stage, 80 pts), so the bulk of
  World-Tour racing is unchanged.  Monuments / GC wins push K > 1; minor
  tour stages push K < 1.  sqrt(K) on σ reflects that big races are higher-
  quality evidence, but with less aggressive compounding than full K would
  produce.  tau is now constant (no longer scaled by points).

Usage:
  pip install openskill numpy
  python model/export_for_web.py --data-dir ./data --out-dir ./docs/outputs
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from openskill.models import BradleyTerryPart

try:
    import orjson as _orjson
except ImportError:
    _orjson = None


def _write_json(obj, path, indent=False):
    """Write obj to path as JSON. Uses orjson if available (~10× faster)."""
    if _orjson is not None:
        # OPT_NON_STR_KEYS matches stdlib behaviour: None/int keys become strings.
        opt = _orjson.OPT_NON_STR_KEYS
        if indent:
            opt |= _orjson.OPT_INDENT_2
        with open(path, "wb") as f:
            f.write(_orjson.dumps(obj, option=opt))
    else:
        with open(path, "w") as f:
            if indent:
                json.dump(obj, f, indent=2)
            else:
                json.dump(obj, f, separators=(",", ":"))


# ════════════════════════════════════════════════════════════════════════════
# Engine constants — match the notebook exactly
# ════════════════════════════════════════════════════════════════════════════
MU_INIT      = 25.0
SIGMA_INIT   = 25.0 / 3.0
BETA         = 25.0 / 6.0
KAPPA        = 1e-4
TAU_BASE     = 25.0 / 1875.0  # anchored at K_ANCHOR (80 pts): matches old effective
                               # tau for a Vuelta/Giro stage (was 25/300 × 80/500)
WINDOW_SIZE  = 16
TAU_PER_DAY  = 0.02

# Race-importance scaling anchor.  A race worth exactly K_ANCHOR points gets
# K = 1 (pure BT, no scaling).  Races above/below scale proportionally.
# 80 pts ≈ a Vuelta / Giro stage — the neutral reference point.
K_ANCHOR = 80.0

DISPLAY_BASE  = 1500
DISPLAY_SCALE = 60


# ════════════════════════════════════════════════════════════════════════════
# Engine — verbatim port of the notebook's BTEngine
# ════════════════════════════════════════════════════════════════════════════
class BTEngine:
    def __init__(self, mu_init=MU_INIT, sigma_init=SIGMA_INIT,
                 beta=BETA, kappa=KAPPA, tau_base=TAU_BASE,
                 window_size=WINDOW_SIZE, tau_per_day=TAU_PER_DAY,
                 k_anchor=K_ANCHOR):
        self.model = BradleyTerryPart(
            mu=mu_init, sigma=sigma_init,
            beta=beta, kappa=kappa, tau=tau_base,
            window_size=window_size, limit_sigma=False,
        )
        self.mu_init    = mu_init
        self.sigma_init = sigma_init
        self.tau_base   = tau_base
        self.tau_per_day = tau_per_day
        self.k_anchor   = k_anchor
        self.ratings     = {}
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

    def _race_k(self, points):
        """Race-importance factor K = points / K_ANCHOR.

        K = 1 for the anchor race (80 pts, a Vuelta/Giro stage).
        K > 1 for monuments, Grand Tour GC wins, Worlds, Olympics.
        K < 1 for minor tour stages and smaller races.

        Applied as:
          Δμ  scaled by K          (full importance on skill update)
          Δσ² scaled by sqrt(K)    (dampened — big races are better evidence
                                    but we don't want aggressive σ collapse)
        tau is constant — no longer mixed with race importance.
        """
        return float(max(points, 1)) / self.k_anchor

    def process_race(self, race):
        results   = race["results"]
        points    = race.get("points", self.k_anchor)
        race_date = datetime.fromisoformat(race["date"])

        for _, rider in results:
            self._register(rider)
            self._apply_time_decay(rider, race_date)

        riders = [r[1] for r in results]
        ranks  = [r[0] for r in results]
        teams  = [[self.ratings[r]] for r in riders]

        # BT requires at least 2 riders — skip degenerate races silently
        if len(teams) < 2:
            return {}

        # tau is now constant — importance is handled by K, not tau
        new_teams = self.model.rate(teams, ranks=ranks, tau=self.tau_base)

        k      = self._race_k(points)
        sqrt_k = float(np.sqrt(k))

        deltas = {}
        for rider, new_team in zip(riders, new_teams):
            old = self.ratings[rider]
            new = new_team[0]

            # Raw openskill deltas
            d_mu   = new.mu - old.mu
            d_sig2 = new.sigma ** 2 - old.sigma ** 2   # always <= 0 (shrink)

            # Apply K-scaling:
            #   μ  moves by full K  — race importance directly scales skill transfer
            #   σ² shrinks by √K   — big races are better evidence, but damped
            scaled_mu   = old.mu + k * d_mu
            scaled_sig2 = old.sigma ** 2 + sqrt_k * d_sig2

            # Guard against numerical underflow (kappa is the model's sigma floor)
            scaled_sigma = float(np.sqrt(max(scaled_sig2, self.model.kappa)))

            self.ratings[rider] = self.model.rating(
                mu=scaled_mu, sigma=scaled_sigma, name=rider,
            )
            deltas[rider] = round(k * d_mu, 4)   # log the K-scaled delta
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
            "race_tau":  self.tau_base,   # constant now; kept for schema compat
            "k":         round(k, 4),
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
            "race_id":          None,
            "race_name":        "Current standings (inactivity decay applied)",
            "date":             reference_date.strftime("%Y-%m-%d"),
            "category":         None,
            "type":             None,
            "points":           None,
            "n_riders":         0,
            "race_tau":         None,
            "k":                None,
            "deltas":           {},
            "ratings":          {r: (rt.mu, rt.sigma) for r, rt in self.ratings.items()},
            "is_decay_snapshot": True,
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
def compute_composite_history_both(engine_mt, tracks=COMPOSITE_TRACKS,
                                    weights=COMPOSITE_WEIGHTS, min_races=2):
    """Vectorised: compute both z=0 (norm) and z=2 (safe) composite histories
    in a single pass over the unified timeline.

    Returns (hist_norm, hist_safe) — same format as two calls to the old
    compute_composite_history() but ~100× faster:
      * per-rider × per-track Python loops replaced by T×N numpy operations
      * both z values computed simultaneously, halving the pass count

    Recipe at every global snapshot:
      - effective skill  eff_norm = mu,  eff_safe = mu - 2σ
      - z-score against that track's current field of qualifiers
      - weight by (1/σ) × track_weight (GC up-weighted)
      - weighted average across the tracks the rider qualifies in
    """
    # ── Pre-index all riders that ever appear across composite tracks ─────────
    all_riders = sorted({r for t in tracks for r in engine_mt.engines[t].ratings})
    rider_idx  = {r: i for i, r in enumerate(all_riders)}
    N          = len(all_riders)
    T          = len(tracks)
    track_idx  = {t: i for i, t in enumerate(tracks)}
    w_arr      = np.array([weights.get(t, 1.0) for t in tracks])  # [T]

    # State arrays — updated incrementally as events are processed.
    mu_mat    = np.full((T, N), np.nan)     # [T, N]
    sigma_mat = np.full((T, N), np.nan)     # [T, N]
    count_mat = np.zeros((T, N), np.int32)  # [T, N]

    # ── Build & sort unified event list ──────────────────────────────────────
    events = []
    for t in tracks:
        for snap_i, snap in enumerate(engine_mt.engines[t].history):
            events.append((snap["date"], t, snap_i))
    events.sort(key=lambda e: e[0])

    hist_norm, hist_safe = [], []

    for global_idx, (date, track, snap_i) in enumerate(events):
        ti   = track_idx[track]
        snap = engine_mt.engines[track].history[snap_i]

        # ── Incremental update: only touched rows change ──────────────────────
        for rider in snap["deltas"]:
            count_mat[ti, rider_idx[rider]] += 1

        if snap.get("is_decay_snapshot"):
            # Decay inflates sigma for every rider in this track.
            for rider, (mu, sg) in snap["ratings"].items():
                ri = rider_idx[rider]
                mu_mat[ti, ri]    = mu
                sigma_mat[ti, ri] = sg
        else:
            # Only riders who actually raced have new ratings.
            for rider in snap["deltas"]:
                mu, sg = snap["ratings"][rider]
                ri = rider_idx[rider]
                mu_mat[ti, ri]    = mu
                sigma_mat[ti, ri] = sg

        # ── Qualifying mask [T, N] ────────────────────────────────────────────
        qual = (count_mat >= min_races) & ~np.isnan(sigma_mat) & (sigma_mat > 0)

        # ── Effective ratings [T, N] ──────────────────────────────────────────
        eff_n = mu_mat                    # z=0
        eff_s = mu_mat - 2.0 * sigma_mat  # z=2

        # ── Per-track mean/std (population) over qualifying riders ────────────
        cnt_t   = qual.sum(axis=1).astype(float)           # [T]
        enough  = cnt_t >= 2                                # [T] bool
        c_safe  = np.where(enough, cnt_t, 1.0)             # avoid /0

        sum_n   = np.where(qual, eff_n, 0.0).sum(axis=1)   # [T]
        sum_s   = np.where(qual, eff_s, 0.0).sum(axis=1)
        mean_n  = np.where(enough, sum_n / c_safe, np.nan)  # [T]
        mean_s  = np.where(enough, sum_s / c_safe, np.nan)

        d_n = np.where(qual, eff_n - mean_n[:, None], 0.0)  # [T, N]
        d_s = np.where(qual, eff_s - mean_s[:, None], 0.0)
        std_n = np.where(enough, np.sqrt((d_n**2).sum(axis=1) / c_safe), np.nan)
        std_s = np.where(enough, np.sqrt((d_s**2).sum(axis=1) / c_safe), np.nan)

        # ── Valid [T, N]: qualifies AND track std > 0 ─────────────────────────
        v_n = qual & (std_n[:, None] > 0) & ~np.isnan(std_n[:, None])
        v_s = qual & (std_s[:, None] > 0) & ~np.isnan(std_s[:, None])

        # ── Weights and z-scores [T, N] ───────────────────────────────────────
        w_n = np.where(v_n, (1.0 / sigma_mat) * w_arr[:, None], 0.0)
        w_s = np.where(v_s, (1.0 / sigma_mat) * w_arr[:, None], 0.0)
        z_n = np.where(v_n, (eff_n - mean_n[:, None]) / std_n[:, None], 0.0)
        z_s = np.where(v_s, (eff_s - mean_s[:, None]) / std_s[:, None], 0.0)

        # ── Weighted composite per rider [N] ──────────────────────────────────
        ws_n = (w_n * z_n).sum(axis=0)
        wt_n = w_n.sum(axis=0)
        ws_s = (w_s * z_s).sum(axis=0)
        wt_s = w_s.sum(axis=0)

        ok_n = (v_n.sum(axis=0) > 0) & (wt_n > 0)  # [N] bool
        ok_s = (v_s.sum(axis=0) > 0) & (wt_s > 0)

        # qual_races = sum of race counts across valid tracks per rider
        qr_n = (count_mat * v_n).sum(axis=0)  # [N]
        qr_s = (count_mat * v_s).sum(axis=0)

        # ── Build output dicts (only qualifying riders) ───────────────────────
        idx_n = ok_n.nonzero()[0]
        idx_s = ok_s.nonzero()[0]

        cv_n = (ws_n / np.where(wt_n > 0, wt_n, 1.0))[idx_n]
        cv_s = (ws_s / np.where(wt_s > 0, wt_s, 1.0))[idx_s]

        comp_n  = {all_riders[i]: float(v) for i, v in zip(idx_n.tolist(), cv_n.tolist())}
        comp_s  = {all_riders[i]: float(v) for i, v in zip(idx_s.tolist(), cv_s.tolist())}
        qrace_n = {all_riders[i]: int(qr_n[i]) for i in idx_n.tolist()}
        qrace_s = {all_riders[i]: int(qr_s[i]) for i in idx_s.tolist()}

        raced_here = set(snap["deltas"].keys())
        is_decay   = snap.get("is_decay_snapshot", False)

        hist_norm.append({
            "idx": global_idx, "date": date,
            "race_name": snap["race_name"], "track": track,
            "raced": raced_here, "qual_races": qrace_n,
            "composites": comp_n, "is_decay_snapshot": is_decay,
        })
        hist_safe.append({
            "idx": global_idx, "date": date,
            "race_name": snap["race_name"], "track": track,
            "raced": raced_here, "qual_races": qrace_s,
            "composites": comp_s, "is_decay_snapshot": is_decay,
        })

    return hist_norm, hist_safe


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
        "generated_at":      datetime.now(timezone.utc).isoformat(),
        "params": {
            "mu_init": MU_INIT, "sigma_init": SIGMA_INIT,
            "beta": BETA, "tau_base": TAU_BASE,
            "tau_per_day": TAU_PER_DAY, "window_size": WINDOW_SIZE,
            "k_anchor": K_ANCHOR,
            "display_base": DISPLAY_BASE, "display_scale": DISPLAY_SCALE,
        },
    }
    _write_json(meta, out_dir / "meta.json", indent=True)
    print(f"  ✓ meta.json  ({len(races)} races, {len(engine.ratings)} riders)")


def build_trajectories(eng):
    """Single-pass build of {rider: [{d,m,s}...]} for all riders in one engine.

    Replaces the old per-rider trajectory() scan: complexity drops from
    O(riders × snapshots) to O(sum of field sizes across all snapshots).
    Only riders in snap["deltas"] have changed ratings between non-decay
    snapshots, so we update last_known lazily rather than scanning all ratings.
    """
    traj       = {}   # rider -> list of datapoints
    last_known = {}   # rider -> (mu, sigma)

    for snap in eng.history:
        if snap.get("is_decay_snapshot"):
            # Decay modifies sigma for every rider — capture the full state.
            date = snap["date"]
            for rider, (mu, sg) in snap["ratings"].items():
                last_known[rider] = (mu, sg)
                series = traj.get(rider)
                if series is None:
                    traj[rider] = [{"d": date,
                                    "m": round(float(mu), 5),
                                    "s": round(float(sg), 5)}]
                elif series[-1]["d"] != date:
                    series.append({"d": date,
                                   "m": round(float(mu), 5),
                                   "s": round(float(sg), 5)})
        else:
            date = snap["date"]
            for rider in snap["deltas"]:
                mu, sg = snap["ratings"][rider]
                last_known[rider] = (mu, sg)
                traj.setdefault(rider, []).append({"d": date,
                                                   "m": round(float(mu), 5),
                                                   "s": round(float(sg), 5)})
    return traj


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

    # One pass per engine instead of one pass per (rider × engine).
    all_trajs = {"ALL": build_trajectories(engine)}
    for track in engine_mt.tracks:
        all_trajs[track] = build_trajectories(engine_mt.engines[track])

    # composite series, keyed by rider, aligned by event index
    comp_series = {}
    safe_by_idx = {h["idx"]: h["composites"] for h in comp_safe}
    for h in comp_norm:
        s_comp = safe_by_idx.get(h["idx"], {})
        if h.get("is_decay_snapshot"):
            for rider, v in h["composites"].items():
                if rider not in comp_series:
                    continue
                if comp_series[rider][-1]["d"] != h["date"]:
                    comp_series[rider].append({
                        "d":  h["date"],
                        "v":  round(float(v), 5),
                        "vs": round(float(s_comp.get(rider, v)), 5),
                    })
        else:
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
        rec = {"ALL": all_trajs["ALL"].get(rider, [])}
        for track in engine_mt.tracks:
            traj = all_trajs[track].get(rider)
            if traj:
                rec[track] = traj
        if rider in comp_series:
            rec["composite"] = comp_series[rider]
        out[rider] = rec

    _write_json(out, out_dir / "rider_timeseries.json")
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

    _write_json(out, out_dir / "top_history.json")
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

    _write_json(out, out_dir / "hall_of_fame.json")
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
    comp_norm, comp_safe = compute_composite_history_both(
        engine_mt, min_races=args.min_races)

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