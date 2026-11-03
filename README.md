# Tire Tracks — dirty-air pace penalty, 2026 Mexico City GP

Measures how much pace an F1 car loses running close behind another, using only public data that passed the audit
([reports/data_audit.md](reports/data_audit.md)).

## Research questions
1. How much time does following within X seconds cost, by sector (straight vs. corners)?
2. At what gap does the slipstream gain on the straight turn into a net loss?
3. Is the loss the car's own aero/tyre penalty, or just being held up by a slower car?
4. Does the loss persist after the car gets clean air (tyre damage)?
5. *(later, multi-race)* Is the effect larger at altitude than at sea-level tracks?

Dropped from the original proposal for lack of reliable data: tyre/brake/PU temperatures, active-aero and
Overtake Mode states, MGU-K/battery, mini-sector timing, fuel correction, 3D/Z analysis.

## Layout
```
src/tire_tracks/
  openf1.py     download OpenF1 session tables → data/raw/<session_key>/*.parquet
  load.py       FastF1 loader (results only without an F1 account)
  audit.py      data-quality checks
  laps.py       per-lap analysis table with exclusion flags + on-road gaps → data/processed/
  firstlook.py  fixed-effects gap model by sector + placebo
  heldup.py     clean-air potential (mixed model + quali), held-up vs wake split, within-lap gap dynamics
  persistence.py  wake accumulation within following runs; exit events
  catalog.py    race/quali session list (2026 + Mexico 2023-25) and bulk download
  multirace.py  per-race pipeline pooled across races; persistence, accumulation, track regression
  trackchar.py  full-throttle fraction / mean speed from the pole lap's telemetry
reports/        audit write-up and figures
```

## Run
```bash
uv sync
PYTHONPATH=src uv run python -c "from tire_tracks.openf1 import download_session; download_session(11404)"
PYTHONPATH=src uv run python -m tire_tracks.audit 11404
PYTHONPATH=src uv run python -m tire_tracks.laps 11404
PYTHONPATH=src uv run python -m tire_tracks.firstlook
PYTHONPATH=src uv run python -c "from tire_tracks.openf1 import download_session; download_session(11400, telemetry=False)"  # qualifying
PYTHONPATH=src uv run python -m tire_tracks.heldup
PYTHONPATH=src uv run python -m tire_tracks.persistence
# multi-race
PYTHONPATH=src uv run python -m tire_tracks.catalog
PYTHONPATH=src uv run python -m tire_tracks.multirace
PYTHONPATH=src uv run python -m tire_tracks.trackchar
```

## Findings
Single race: [reports/heldup_and_persistence.md](reports/heldup_and_persistence.md).
Multi-race (22 races): [reports/multirace_results.md](reports/multirace_results.md).
- Most of the apparent dirty-air penalty is being held up by a slower car, at every track.
- Beyond 0.5 s, the wake's cornering loss and the tow's straight-line gain cancel. Inside 0.5 s it costs ~0.4 s/lap.
- Layout sets the sign: power tracks give a net gain from following, twisty tracks a net loss.
- Mexico City's wake effect is smaller than its layout predicts, and its Esses loss halved under the 2026 rules.
- No tyre damage from following in 2026: no build-up in the wake, no after-effect once in free air.
  Old rules (Mexico 2023-25): ~0.015 s/lap build-up and a ~0.2 s after-effect (15 exits).
