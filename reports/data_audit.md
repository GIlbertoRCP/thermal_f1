# Data audit — 2026 Mexico City GP (OpenF1 session 11404)

Run: `PYTHONPATH=src uv run python -m tire_tracks.audit 11404`

## Sources

| Source | Status |
|---|---|
| FastF1 3.9.2 live timing | **Blocked** without an F1 account (`F1_USERNAME`/`F1_PASSWORD`). Only results load (via Jolpica). |
| OpenF1 (no auth) | **Primary source.** All tables below. Cached as parquet in `data/raw/11404/`. |

Results check out against the proposal narrative: VER P1, HAM +0.799s, LAW P22→P7 (+20.899s), red flag on lap 45.

## What is reliable (keep)

| Data | Quality | Use |
|---|---|---|
| Lap times | 0.4% missing | Main target |
| Sector times S1/S2/S3 | 0.4–2.1% missing, ms precision | Sector-level effects (the key split: straight vs. corners) |
| Speed traps i1/i2 | 1–5% missing | Slipstream check |
| Speed trap `st_speed` | **20.7% missing** | Secondary only |
| Stints / compound / tyre age | Contiguous for all drivers; 1 `UNKNOWN` stint | Tyre state. Only SOFT and MEDIUM were raced. |
| Pit stops | 67 rows, 58 with lane time | Exclusions |
| Race control | SC/VSC/red/yellow/blue messages with timestamps | Exclusions |
| Gap to car ahead | Derived from line-crossing timestamps; agrees with the OpenF1 `intervals` feed to **0.054s median** (p90 0.13s, r=0.96) | Main exposure variable, measured **on-road**, so traffic a lap down counts too |
| Car telemetry speed/throttle/brake/gear/RPM | ~4.2 Hz (median 0.24s), p95 0.44s | Speed profiles, minimum corner speeds |
| XY location | ~4.2 Hz, 1.3% zeros | Track map, binning |
| XY location, other races | Monaco 2026 has holes at the source (whole 20-min windows return no results) | Monaco dropped from the track map; each map reports its lap coverage |
| Weather | Air 23–25 °C, track 40 °C mean, pressure ~783 hPa (confirms altitude) | Covariate; one `track_temperature = 0` sensor glitch is dropped |

## What is not reliable or doesn't exist (drop)

| Proposal item | Finding | Decision |
|---|---|---|
| Tyre / brake / PU temperatures ("thermal ingestion", "core temperature") | No public feed has them | **Drop.** "Thermal" can only be inferred from pace persistence later. |
| Active aero mode, Overtake Mode, MGU-K deployment, battery state | Not in `car_data`. The `drs` column is **100% null** in 2026. | **Drop** the "Active Aero/Overtake burst" component. |
| Mini-sector timing from `segments_*` | Values are colour/status codes (2048/2049/2051/2064), not times | **Drop** |
| Brake pressure | Binary 0/100 (plus a 104 sentinel) | On/off only |
| Throttle | 12.8% of samples are 104 (sentinel) | Clip/mask |
| 200m micro-sector *timing* from telemetry | At 4 Hz a 200m bin at 250 km/h holds ~3 samples. Integrated distance is 4,211m vs the official 4,304m (2% short, tight spread). | Use for **speed profiles** only; time effects use official sectors |
| Fuel correction (0.03 s/kg) | Fuel load isn't published, and within a stint fuel burn, track evolution and tyre age move one-for-one (rank-deficient design) | **Drop.** Report the *net in-stint slope* instead. |
| Z / 3D track | Flat circuit; Z adds nothing analytic | Drop for analysis |
| Telemetry dropouts | 1,870 gaps >1s (2,699s total across 22 cars), incl. the red-flag period | Mask; not used for lap-level work |

## Clean-lap filter

Excluded: lap 1, pit in/out laps, laps overlapping VSC (laps 7–8), SC → red flag → standing restart (43–46), SC (59–63), restart laps, double-yellow overlap, blue flags, and being lapped (car behind is a lap up and within 3s).

**1,023 clean laps** out of 1,360. STR and ALO (early retirements) have <3 laps; every other driver has 36–56.

Gap distribution of clean laps: <0.5s: 54 · 0.5–1.0: 252 · 1.0–1.5: 142 · 1.5–2.0: 86 · 2–3: 156 · 3–5: 158 · >5: 196.
There are 35 runs of ≥5 consecutive laps within 1.5s, 17 of them ≥10 laps, which is enough to test persistence.

## First look (descriptive, not causal)

`PYTHONPATH=src uv run python -m tire_tracks.firstlook` → `gap_effect_by_sector.png`, `gap_effect_by_sector.csv`

Model: time ~ driver×stint FE + tyre age×compound + gap-to-car-ahead bins (ref: >3s); 861 laps, 20 drivers, SEs clustered by driver.

| Gap ahead | Lap | S1 (straight) | S2 (Esses) | S3 (stadium) |
|---|---|---|---|---|
| <0.5s (n≈46) | **+0.69** [0.31, 1.06] | −0.06 | **+0.53** [0.36, 0.70] | **+0.22** [0.12, 0.33] |
| 0.5–1.0s (n≈225) | +0.26 [−0.08, 0.59] | **−0.14** [−0.22, −0.05] | **+0.28** [0.12, 0.45] | **+0.11** [0.00, 0.21] |
| 1.0–1.5s | +0.11 | −0.05 | +0.10 | +0.06 |
| 1.5–3.0s | ~+0.1 | ~0 | +0.06–0.09 | ~+0.05 |

- **The sector signature matches aero physics:** a slipstream gain on the straight and losses in the corners, mostly in the Esses.
- **The placebo passes:** gap to the car *behind* (in clean air ahead) has no significant effect.
- Net in-stint slope: SOFT +0.006 s/lap (±0.005), MEDIUM +0.019 s/lap (±0.005). This is tyre wear minus fuel burn plus track evolution, not tyre wear alone.

### Main caveat: being held up (resolved in [heldup_and_persistence.md](heldup_and_persistence.md))
When within 1s, the follower's lap time tracks the car ahead's at **r = 0.97** (median difference 0.11s), against r = 0.41 in clean air. Much of the "penalty" is therefore likely followers matching a slower leader, not their own aero loss. **Separating these is the next modelling step.** Options:
1. Restrict to followers whose clean-air pace is slower than or equal to the leader's, so they aren't being held up.
2. Compare the follower's sector to the *leader's* sector on the same lap: being held up means equal times; aero loss shows up in specific corners.
3. Test persistence: is pace still lost for k laps after the driver gets clean air? Only tyre damage would persist.
