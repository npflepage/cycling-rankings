# Cycling BT Rankings · World Tour 2019–2026

Time-aware Bradley–Terry (Partial Pairing) rider rankings, rendered as a
static dashboard on GitHub Pages.

## What it does

1. **`Cycling_BT_TimeAware.ipynb`** — your existing notebook builds the
   `BTEngine` and `BTEngineMultiTrack` from the per-year JSON race files.
2. **`model/export_for_web.py`** — the *same* engine logic, refactored
   into a CLI script. Loads `data/*.json`, runs the cumulative engine and
   the 8 per-track engines, then writes 4 lightweight JSON files into
   `docs/outputs/`.
3. **`docs/index.html`** — single-file static dashboard. Fetches the four
   JSON files on load and renders an interactive Plotly chart, a top-10
   panel, and a timeline scrubber. No backend, no build step.

## Repo layout

```
.
├── data/                        ← per-year race data (one JSON per year)
│   ├── 2019.json
│   ├── 2020.json
│   └── ...
├── model/
│   └── export_for_web.py        ← runs the model, writes JSON exports
├── docs/                        ← what GitHub Pages serves
│   ├── index.html               ← the landing page
│   └── outputs/                 ← generated, gitignored if you prefer
│       ├── meta.json
│       ├── rider_timeseries.json
│       ├── top5_history.json
│       └── current_top5.json
└── Cycling_BT_TimeAware.ipynb   ← your original notebook (untouched)
```

## Quick start (locally)

```bash
# 1. install deps
pip install openskill numpy

# 2. run the model and write JSON outputs
python model/export_for_web.py --data-dir ./data --out-dir ./docs/outputs

# 3. preview the page locally (any static server will do)
cd docs && python -m http.server 8000
# → open http://localhost:8000
```

## Deploying to GitHub Pages

1. Push this repo to GitHub.
2. Go to **Settings → Pages**.
3. Set **Source** to **Deploy from a branch**, **Branch** = `main`, **Folder** = `/docs`.
4. Save. After ~1 minute, your dashboard is live at
   `https://<your-username>.github.io/<repo-name>/`.

That's it — no build pipeline, no Node, no React.

## Auto-rebuild on push (optional)

The included `.github/workflows/build-exports.yml` runs the model on
every push to `main` and commits the regenerated `docs/outputs/*.json`
files back. Skip this if you'd rather run the exports locally.

## How the engine maps to the JSON

The exporter calls the same `BTEngine.process_race` and
`BTEngineMultiTrack.process_race` you defined in the notebook, with
identical constants:

| Parameter      | Value           |
|----------------|----------------:|
| `MU_INIT`      | 25.0            |
| `SIGMA_INIT`   | 25/3            |
| `BETA`         | 25/6            |
| `TAU_BASE`     | 25/300          |
| `TAU_PER_DAY`  | 0.02            |
| `WINDOW_SIZE`  | 16              |
| Display scale  | 1500 + 60·(μ−μ₀) |

Tracks: `sprint`, `TT`, `cobbles`, `punch`, `mountain`, `GC`,
`stage_race`, `classics`. The "All races" view uses the cumulative
single-engine view.

## Tweaking the dashboard

- **Default riders**: edit the `defaults` array near the bottom of
  `docs/index.html` (`init()` function).
- **Top-N count**: pass `top_n=...` to `export_top5_history` /
  `export_current_top5` in `export_for_web.py`.
- **Snapshot density**: pass `--downsample 5` to skip every 5th race in
  the top-N history (smaller JSON, coarser scrubber).
- **Colors / typography**: CSS variables at the top of `index.html`.
