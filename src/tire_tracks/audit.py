"""Data-quality checks for one OpenF1 race session. Prints findings; run as a script."""
import sys

import numpy as np
import pandas as pd

from tire_tracks.openf1 import read

TRACK_LENGTH_M = 4304  # Autódromo Hermanos Rodríguez, official


def race_window(laps: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = laps.date_start + pd.to_timedelta(laps.lap_duration, unit="s")
    return laps.date_start.min(), end.max()


def section(title: str):
    print(f"\n## {title}\n")


def audit(key: int):
    laps, car, loc = read(key, "laps"), read(key, "car_data"), read(key, "location")
    iv, stints, pit = read(key, "intervals"), read(key, "stints"), read(key, "pit")
    wx, drivers = read(key, "weather"), read(key, "drivers")
    t0, t1 = race_window(laps)
    car = car[(car.date >= t0) & (car.date <= t1)]
    loc = loc[(loc.date >= t0) & (loc.date <= t1)]
    iv = iv[(iv.date >= t0) & (iv.date <= t1)]
    wx = wx[(wx.date >= t0 - pd.Timedelta("5min")) & (wx.date <= t1)]

    section("Laps")
    print("rows", len(laps), "drivers", laps.driver_number.nunique(), "max lap", laps.lap_number.max())
    for c in ["lap_duration", "duration_sector_1", "duration_sector_2", "duration_sector_3",
              "i1_speed", "i2_speed", "st_speed"]:
        print(f"  missing {c:18s} {laps[c].isna().mean():.1%}")
    seg = pd.Series(np.concatenate([np.asarray(s) for s in laps.segments_sector_1.dropna()]))
    print("  mini-sector 'segments' value set:", sorted(seg.dropna().unique().astype(int))[:12],
          "(status codes, not times)")

    section("Car telemetry (car_data), race window only")
    dt = car.sort_values("date").groupby("driver_number").date.diff().dt.total_seconds().dropna()
    print(f"  sample interval median {dt.median():.3f}s (≈{1/dt.median():.2f} Hz), p95 {dt.quantile(.95):.3f}s")
    print(f"  dropouts >1s: {(dt > 1).sum()} gaps, total {dt[dt > 1].sum():.0f}s across all drivers")
    print("  drs values:", car.drs.value_counts(dropna=False).to_dict())
    print("  brake values:", sorted(car.brake.unique())[:10])
    print("  throttle range:", car.throttle.min(), car.throttle.max(),
          f"(>100: {(car.throttle > 100).mean():.2%})")
    print("  speed max:", car.speed.max(), " rpm max:", car.rpm.max(), " gears:", sorted(car.n_gear.unique()))

    section("Distance from integrating speed (needed for micro-sectors)")
    rows = []
    for _, lap in laps.dropna(subset=["lap_duration"]).iterrows():
        a, b = lap.date_start, lap.date_start + pd.Timedelta(seconds=lap.lap_duration)
        c = car[(car.driver_number == lap.driver_number) & (car.date >= a) & (car.date < b)]
        if len(c) < 50:
            continue
        t = (c.date - a).dt.total_seconds().to_numpy()
        rows.append(np.trapezoid(c.speed.to_numpy() / 3.6, t))
    d = pd.Series(rows)
    print(f"  laps {len(d)}; integrated distance median {d.median():.0f} m vs official {TRACK_LENGTH_M} m;"
          f" IQR {d.quantile(.25):.0f}–{d.quantile(.75):.0f} m")

    section("Location (XYZ)")
    dl = loc.sort_values("date").groupby("driver_number").date.diff().dt.total_seconds().dropna()
    print(f"  sample interval median {dl.median():.3f}s, zeros (0,0,0): {((loc.x == 0) & (loc.y == 0)).mean():.2%}")
    print(f"  z range {loc.z.min()}–{loc.z.max()} (units 1/10 m); x span {loc.x.max()-loc.x.min()}")

    section("Intervals (gap to car ahead)")
    di = iv.sort_values("date").groupby("driver_number").date.diff().dt.total_seconds().dropna()
    print(f"  rows {len(iv)}, update interval median {di.median():.1f}s, p95 {di.quantile(.95):.1f}s")
    print(f"  interval missing {iv.interval.isna().mean():.1%}; non-numeric gap_to_leader values:",
          iv.gap_to_leader[pd.to_numeric(iv.gap_to_leader, errors='coerce').isna()].value_counts().head(5).to_dict())

    section("Gap at the timing line (from lap start timestamps) vs intervals feed")
    ls = laps[["driver_number", "lap_number", "date_start"]].dropna()
    ls = ls.sort_values(["lap_number", "date_start"])
    ls["gap_line"] = ls.groupby("lap_number").date_start.diff().dt.total_seconds()
    ivn = iv.dropna(subset=["interval"]).sort_values("date")
    m = pd.merge_asof(ls.sort_values("date_start"), ivn[["date", "driver_number", "interval"]],
                      left_on="date_start", right_on="date", by="driver_number",
                      direction="nearest", tolerance=pd.Timedelta("6s")).dropna(subset=["gap_line", "interval"])
    m = m[(m.lap_number > 1) & (m.gap_line < 5)]
    err = m.gap_line - m.interval
    print(f"  matched {len(m)} driver-laps with gap<5s; median abs diff {err.abs().median():.3f}s,"
          f" p90 {err.abs().quantile(.9):.3f}s, corr {m.gap_line.corr(m.interval):.3f}")

    section("Stints / pits")
    print("  compounds:", stints.compound.value_counts().to_dict())
    bad = []
    for drv, g in stints.sort_values("stint_number").groupby("driver_number"):
        if (g.lap_start.iloc[1:].to_numpy() != g.lap_end.iloc[:-1].to_numpy() + 1).any():
            bad.append(int(drv))
    print("  drivers with non-contiguous stints:", bad or "none")
    print("  pit rows:", len(pit), " with lane duration:", pit.lane_duration.notna().sum())

    section("Weather")
    print(wx[["air_temperature", "track_temperature", "humidity", "pressure", "wind_speed", "rainfall"]]
          .describe().loc[["min", "mean", "max"]].round(1).to_string())

    section("Signals that do not exist in any public feed")
    print("  tyre surface/carcass temperature, brake temperature, PU/coolant temperature,")
    print("  battery state of charge, MGU-K deployment, active-aero mode, Overtake Mode usage")
    print("  car_data columns:", [c for c in read(key, 'car_data').columns])


if __name__ == "__main__":
    audit(int(sys.argv[1]) if len(sys.argv) > 1 else 11404)
