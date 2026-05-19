# Cycling BT Rankings · World Tour 2019–2026

Time-aware Bradley–Terry (Partial Pairing) rider rankings, rendered as a
static dashboard on GitHub Pages.

## What it does

1. **`Cycling_BT_TimeAware.ipynb`** — your existing notebook builds the
   `BTEngine` and `BTEngineMultiTrack` from the per-year JSON race files.
2. **`model/export_for_web.py`** — the *same* engine logic, refactored
   into a CLI script. Loads `data/*.json`, runs the cumulative engine, the
   8 per-track engines and the cross-discipline **composite** trajectory,
   then writes 4 lightweight JSON files into `docs/outputs/`.
3. **`docs/index.html`** — single-file static dashboard. Fetches the four
   JSON files on load and renders the interactive chart, snapshot Top 10,
   a live head-to-head calculator, the "#1 club", and an all-time GOAT
   board. No backend, no build step.

## Repo layout

```
.
├── data/                        ← per-year race data (one JSON per year)
│   ├── 2019.json … 2026.json
├── model/
│   └── export_for_web.py        ← runs the model, writes JSON exports
├── docs/                        ← what GitHub Pages serves
│   ├── index.html               ← the landing page
│   └── outputs/                 ← generated
│       ├── meta.json
│       ├── rider_timeseries.json
│       ├── top_history.json
│       └── hall_of_fame.json
└── Cycling_BT_TimeAware.ipynb   ← your original notebook (untouched)
```

## Quick start (locally)

```bash
pip install openskill numpy
python model/export_for_web.py --data-dir ./data --out-dir ./docs/outputs
cd docs && python -m http.server 8000      # → http://localhost:8000
```

## Deploying to GitHub Pages

Settings → Pages → Deploy from a branch → `main` → `/docs`. Live in ~1 min
at `https://<user>.github.io/<repo>/`.

## The four JSON exports

| File                    | Purpose                                                                 |
|-------------------------|-------------------------------------------------------------------------|
| `meta.json`             | stats bar, params (incl. `beta`, display scale), composite config       |
| `rider_timeseries.json` | per-rider `{d, m, s}` per track + `{d, v, vs}` for the composite        |
| `top_history.json`      | per snapshot, `top` (μ order) **and** `top_safe` (μ−2σ order)           |
| `hall_of_fame.json`     | per track: #1 `reigns` + all-time `goat`, for `normal` and `safe`       |

### Why these, and what's computed where

- The timeseries stores **raw (μ, σ)**, not display ELO. The page derives
  the ELO / safe-ELO line itself, so the **safe toggle redraws instantly**,
  and — more importantly — it can run the **head-to-head win probability
  online** for *any* pair without a precomputed O(n²) matrix.
- Anything that depends only on *(track, metric)* and is expensive
  (per-snapshot rankings, #1 reigns, all-time peaks, the composite) is
  baked into static JSON so the page stays a dumb fast renderer.

### Head-to-head probability

Uses the Thurstone–Mosteller link (openskill's `predict_win`), which
accounts for **both** riders' uncertainty:

```
P(A beats B) = Φ( (rA − rB) / sqrt(2·β² + σA² + σB²) )
r = μ − z·σ        (z = 2 in safe mode, else 0)
```

β is exported in `meta.params.beta`; Φ is the normal CDF (an `erf`
approximation in JS). Evaluated live at the selected snapshot.

## The composite ("Cumulative") score

The 8 BT tracks overlap (`classics ⊃ punch`, `stage_race ⊃ GC`, …), so the
composite is built only from the **disjoint type tracks**:

```
cobbles · punch · mountain · sprint · GC
```

At every point on the unified timeline each track's effective skill
(`μ − zσ`) is z-scored against that track's current field, weighted by
`(1/σ) · track_weight`, then averaged over the tracks the rider qualifies
in. **GC is up-weighted** (default `1.75`) because holding form across a
3-week tour is a stronger all-round signal than isolated stage wins.

Tweak in `export_for_web.py`:

```python
COMPOSITE_TRACKS  = ["cobbles", "punch", "mountain", "sprint", "GC"]
COMPOSITE_WEIGHTS = {"cobbles":1.0,"punch":1.0,"mountain":1.0,"sprint":1.0,"GC":1.75}
```

## Engine constants

| Parameter      | Value           |
|----------------|----------------:|
| `MU_INIT`      | 25.0            |
| `SIGMA_INIT`   | 25/3            |
| `BETA`         | 25/6            |
| `TAU_BASE`     | 25/300          |
| `TAU_PER_DAY`  | 0.02            |
| `WINDOW_SIZE`  | 16              |
| Display scale  | 1500 + 60·(μ−μ₀) |

## What the controls affect

- **Track tab + safe toggle** → *everything* (chart, Top 10, head-to-head,
  #1 club, GOAT). The Top 10 now flips between μ and μ−2σ order.
- **Timeline slider** → chart, snapshot Top 10, head-to-head only. The #1
  club and GOAT always span the full period.
- All charts use **linear interpolation** between points (no spline) so the
  genuinely noisy trajectory isn't smoothed away.

## Tweaking the dashboard

- **Default chart riders**: `defaults` array in `init()` in `index.html`.
- **GC weight / composite tracks**: constants above in `export_for_web.py`.
- **Top-N / GOAT-N**: `top_n` / `goat_n` args in the export functions.
- **Scrubber density**: `--downsample 5` (reigns/GOAT stay full-resolution).
- **Colors / typography**: CSS variables at the top of `index.html`.