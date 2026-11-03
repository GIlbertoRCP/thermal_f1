"""Speed by track position: where on the lap a follower loses or gains speed.

Car telemetry (speed, throttle, brake) and XY location both arrive at ~4 Hz on different clocks.
Each speed sample gets the nearest location sample (<= 250 ms), is projected onto a reference line
(the race's fastest free-air clean lap) and binned into BIN_M-metre bins along it. Per-lap bin means
are stored; the dashboard compares close-following laps with the same driver's free-air laps.

At 4 Hz a 50 m bin at 300 km/h holds ~0.6 samples per lap, so single laps are noisy: only averages over
many laps are meaningful. The XY path cuts corners slightly (4.19 km vs 4.30 km official at Mexico), so
distances are approximate. Z (elevation) is not used.
"""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from tire_tracks import laps as lapmod
from tire_tracks.openf1 import RAW, download_session, read

BIN_M = 50
MATCH_TOL = pd.Timedelta("250ms")
MIN_BIN_FRAC = 0.95  # laps with telemetry dropouts are skipped


def has_telemetry(key: int) -> bool:
    return (RAW / str(key) / "car_data.parquet").exists() and (RAW / str(key) / "location.parquet").exists()


def reference_line(key: int, laps: pd.DataFrame, loc: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Densified XY path of the fastest clean free-air lap, with distance in metres, and the sector
    boundaries on it."""
    free = laps[laps.clean & (laps.gap_ahead > 3) & (laps.gap_ahead_end > 3)]
    for _, r in free.sort_values("lap_duration").iterrows():  # fastest lap with full ~4 Hz location coverage
        p = loc[(loc.driver_number == r.driver_number) & (loc.date >= r.date_start) & (loc.date <= r.date_end)]
        if len(p) >= 3 * r.lap_duration:
            break
    else:
        raise ValueError(f"{key}: no free-air lap with full location coverage")
    xy = p[["x", "y"]].to_numpy(float) / 10  # decimetres -> metres
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    dist = np.arange(0, seg[-1], 2.0)
    line = pd.DataFrame({"dist": dist, "x": np.interp(dist, seg, xy[:, 0]), "y": np.interp(dist, seg, xy[:, 1])})
    # sector boundaries: where the reference car was when its sector times ran out
    t = (p.date - r.date_start).dt.total_seconds().to_numpy()
    s1, s2 = r.duration_sector_1, r.duration_sector_1 + r.duration_sector_2
    sectors = {"s1_end": float(np.interp(s1, t, seg)), "s2_end": float(np.interp(s2, t, seg)),
               "length": float(seg[-1]), "ref_driver": int(r.driver_number), "ref_lap": int(r.lap_number)}
    return line, sectors


def lap_profiles(key: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Per clean lap and distance bin: mean speed, throttle, brake. Plus the binned reference line."""
    laps = pd.read_parquet(lapmod.PROCESSED / f"laps_{key}.parquet")
    car, loc = read(key, "car_data"), read(key, "location")
    for col in ("throttle", "brake"):  # 104 is a sentinel (audit), not a reading
        car[col] = car[col].where(car[col] <= 100)
    loc = loc[(loc.x != 0) | (loc.y != 0)]
    line, sectors = reference_line(key, laps, loc)
    tree = cKDTree(line[["x", "y"]].to_numpy())

    clean = laps[laps.clean][["driver_number", "lap_number", "date_start", "date_end"]].dropna()
    out = []
    for drv, c in car.groupby("driver_number"):
        l = loc[loc.driver_number == drv].sort_values("date")
        c = c.sort_values("date")
        m = pd.merge_asof(c[["date", "speed", "throttle", "brake"]], l[["date", "x", "y"]], on="date",
                          direction="nearest", tolerance=MATCH_TOL).dropna(subset=["x", "y"])
        cl = clean[clean.driver_number == drv].sort_values("date_start")
        m = pd.merge_asof(m, cl, left_on="date", right_on="date_start", direction="backward").dropna(subset=["lap_number"])
        m = m[m.date < m.date_end]
        if m.empty:
            continue
        _, idx = tree.query(m[["x", "y"]].to_numpy() / 10)
        m["bin"] = (line.dist.to_numpy()[idx] // BIN_M).astype(int)
        out.append(m.groupby(["driver_number", "lap_number", "bin"])[["speed", "throttle", "brake"]].mean().reset_index())
    prof = pd.concat(out, ignore_index=True)
    prof["race_key"] = key
    # OpenF1 location has holes at some races (Monaco 2026: whole 20-min windows return no results)
    cover = prof.groupby(["driver_number", "lap_number"]).bin.nunique()
    sectors["coverage"] = float((cover >= MIN_BIN_FRAC * (line.dist.max() // BIN_M + 1)).sum() / len(clean))

    line["bin"] = (line.dist // BIN_M).astype(int)
    bins = line.groupby("bin").agg(x=("x", "mean"), y=("y", "mean"), dist=("dist", "mean")).reset_index()
    return prof, bins, sectors


def build(key: int):
    prof, bins, sectors = lap_profiles(key)
    prof.to_parquet(lapmod.PROCESSED / f"profile_{key}.parquet")
    bins.assign(race_key=key, **sectors).to_parquet(lapmod.PROCESSED / f"trackline_{key}.parquet")
    return prof, bins, sectors


def load(key: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profiles of laps with >= MIN_BIN_FRAC of bins covered (telemetry dropouts), and the track line."""
    prof = pd.read_parquet(lapmod.PROCESSED / f"profile_{key}.parquet")
    line = pd.read_parquet(lapmod.PROCESSED / f"trackline_{key}.parquet")
    cover = prof.groupby(["driver_number", "lap_number"]).bin.transform("nunique")
    return prof[cover >= MIN_BIN_FRAC * len(line)], line



def wake_speed_delta(prof: pd.DataFrame, line: pd.DataFrame, close: pd.DataFrame, free: pd.DataFrame,
                     n_boot: int = 1000, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Speed of close-following laps minus what the same driver does in free air at that point.

    close/free: driver_number, lap_number, P_lap (clean-air potential). The free-air profile is scaled by
    the potential ratio, so fuel load and tyre age (close laps come later in the race) don't show up
    as wake effects. Returns per bin: mean delta and a driver-clustered bootstrap CI, plus the time
    those deltas add up to per sector."""
    base = prof.merge(free[["driver_number", "lap_number"]]).groupby(["driver_number", "bin"])[
        ["speed", "throttle"]].mean().add_prefix("b_").reset_index()
    f_pot = free.groupby("driver_number").P_lap.mean().rename("f_pot")
    x = (prof.merge(close[["driver_number", "lap_number", "P_lap"]])
         .merge(base, on=["driver_number", "bin"]).merge(f_pot, left_on="driver_number", right_index=True))
    x["exp_speed"] = x.b_speed * x.f_pot / x.P_lap
    x["d_speed"] = x.speed - x.exp_speed
    x["d_throttle"] = x.throttle - x.b_throttle
    step = line.dist.diff().median()  # bin length in metres
    x["dt"] = step / (x.speed / 3.6) - step / (x.exp_speed / 3.6)

    drivers = np.sort(x.driver_number.unique())
    di = np.searchsorted(drivers, x.driver_number.to_numpy())
    bi = x.bin.to_numpy()
    nb = int(line.bin.max()) + 1
    w = np.random.default_rng(seed).multinomial(len(drivers), np.full(len(drivers), 1 / len(drivers)), size=n_boot)

    def boot(col):
        s, n = np.zeros((len(drivers), nb)), np.zeros((len(drivers), nb))
        np.add.at(s, (di, bi), x[col].to_numpy())
        np.add.at(n, (di, bi), 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            point = s.sum(0) / n.sum(0)
            reps = (w @ s) / (w @ n)
        return point, reps

    d_speed, r_speed = boot("d_speed")
    d_thr, _ = boot("d_throttle")
    d_t, r_t = boot("dt")
    out = line.set_index("bin").reindex(range(nb)).reset_index()
    out["d_speed"], out["d_throttle"], out["dt"] = d_speed, d_thr, d_t
    out["lo"], out["hi"] = np.nanpercentile(r_speed, 2.5, axis=0), np.nanpercentile(r_speed, 97.5, axis=0)
    out["n_laps"] = np.bincount(bi, minlength=nb)

    sec = np.where(out.dist < line.s1_end.iloc[0], "S1", np.where(out.dist < line.s2_end.iloc[0], "S2", "S3"))
    out["sector"] = sec
    rows = []
    for s in ("S1", "S2", "S3", "Lap"):
        m = np.ones(nb, bool) if s == "Lap" else sec == s
        reps = np.nansum(r_t[:, m], axis=1)
        rows.append({"sector": s, "time_s": np.nansum(d_t[m]), "lo": np.percentile(reps, 2.5),
                     "hi": np.percentile(reps, 97.5)})
    meta = pd.DataFrame(rows).assign(n_laps=x.groupby(["driver_number", "lap_number"]).ngroups, n_drivers=len(drivers))
    return out, meta


if __name__ == "__main__":
    import sys
    keys = [int(k) for k in sys.argv[1:]] or [11404]
    for k in keys:
        if not has_telemetry(k):
            download_session(k, telemetry=True)
        try:
            prof, bins, sectors = build(k)
        except Exception as e:  # noqa: BLE001 - a race with too little location data is reported and skipped
            print(k, "failed:", repr(e)[:120], flush=True)
            continue
        print(k, "laps:", prof.groupby(["driver_number", "lap_number"]).ngroups, "bins:", len(bins), sectors, flush=True)
