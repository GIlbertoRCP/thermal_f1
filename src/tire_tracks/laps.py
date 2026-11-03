"""Build the per-lap analysis table: one row per driver-lap with exclusion flags,
tyre state, on-road gaps and track temperature. Only uses fields the audit found reliable."""
from pathlib import Path

import numpy as np
import pandas as pd

from tire_tracks.openf1 import read

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"


def neutralised_windows(rc: pd.DataFrame, race_end: pd.Timestamp | None = None
                        ) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """(start, end, kind) for VSC, SC and red-flag periods, from race control messages.
    Handles both 'VSC ...' (2026) and 'VIRTUAL SAFETY CAR ...' (earlier) wording."""
    rc = rc.sort_values("date")
    msg = rc.message.fillna("").str.upper()
    out, open_ = [], {}
    for t, m in zip(rc.date, msg):
        vsc = m.startswith("VSC") or m.startswith("VIRTUAL SAFETY CAR")
        if vsc and "DEPLOYED" in m:
            open_.setdefault("VSC", t)
        elif vsc and "ENDING" in m and "VSC" in open_:
            out.append((open_.pop("VSC"), t + pd.Timedelta("20s"), "VSC"))
        elif m.startswith("SAFETY CAR DEPLOYED"):
            if "VSC" in open_:  # VSC upgraded to SC without a 'VSC ENDING' message
                out.append((open_.pop("VSC"), t, "VSC"))
            open_.setdefault("SC", t)
        elif m.startswith("SAFETY CAR IN THIS LAP") and "SC" in open_:
            # SC peels off this lap; the following green lap is caught by the restart rule
            out.append((open_.pop("SC"), t + pd.Timedelta("90s"), "SC"))
        elif m.startswith("RED FLAG"):
            for k in ("SC", "VSC"):
                if k in open_:
                    out.append((open_.pop(k), t, k))
            open_.setdefault("RED", t)
        elif "RED" in open_ and (m in ("STANDING START", "SESSION STARTED") or "ROLLING START" in m
                                 or m.startswith("SAFETY CAR IN THIS LAP")):
            out.append((open_.pop("RED"), t + pd.Timedelta("90s"), "RED"))
    end = race_end if race_end is not None else rc.date.max()
    out += [(s, end, k) for k, s in open_.items()]
    return out


def double_yellow_windows(rc: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    rc = rc[rc.category == "Flag"].sort_values("date")
    out, open_ = [], {}
    for t, flag, sec in zip(rc.date, rc.flag, rc.sector):
        if flag == "DOUBLE YELLOW":
            open_.setdefault(sec, t)
        elif flag == "CLEAR" and sec in open_:
            out.append((open_.pop(sec), t))
    return out


def _overlaps(a, b, windows) -> np.ndarray:
    hit = np.zeros(len(a), dtype=bool)
    for s, e, *_ in windows:
        hit |= (a < e) & (b > s)
    return hit


def road_gaps(crossings: pd.DataFrame) -> pd.DataFrame:
    """Time gap to the physical car ahead/behind at each line crossing, whatever its lap."""
    c = crossings.sort_values("date_start").reset_index(drop=True)
    t = c.date_start.astype("int64").to_numpy() / 1e9
    drv, lap = c.driver_number.to_numpy(), c.lap_number.to_numpy()
    ahead, behind, ahead_drv = np.full(len(c), np.nan), np.full(len(c), np.nan), np.zeros(len(c), int)
    lap_ahead, lap_behind, behind_drv = np.zeros(len(c), int), np.zeros(len(c), int), np.zeros(len(c), int)
    for i in range(len(c)):
        j = i - 1
        while j >= 0 and drv[j] == drv[i]:
            j -= 1
        if j >= 0:
            ahead[i], ahead_drv[i], lap_ahead[i] = t[i] - t[j], drv[j], lap[j] - lap[i]
        k = i + 1
        while k < len(c) and drv[k] == drv[i]:
            k += 1
        if k < len(c):
            behind[i], lap_behind[i], behind_drv[i] = t[k] - t[i], lap[k] - lap[i], drv[k]
    c["gap_ahead"], c["gap_behind"], c["car_ahead"], c["car_behind"] = ahead, behind, ahead_drv, behind_drv
    # car ahead a lap down = backmarker traffic; car behind a lap up = we are being lapped
    c["ahead_is_backmarker"] = lap_ahead < 0
    c["being_lapped"] = (lap_behind > 0) & (behind < 3)
    return c


def build(key: int) -> pd.DataFrame:
    laps, stints, pit = read(key, "laps"), read(key, "stints"), read(key, "pit")
    rc, wx, drivers = read(key, "race_control"), read(key, "weather"), read(key, "drivers")

    df = laps[["driver_number", "lap_number", "date_start", "lap_duration",
               "duration_sector_1", "duration_sector_2", "duration_sector_3",
               "i1_speed", "i2_speed", "st_speed", "is_pit_out_lap"]].copy()
    df["date_end"] = df.date_start + pd.to_timedelta(df.lap_duration, unit="s")
    # session tables can carry other days' messages (e.g. qualifying red flags): clip to the race
    t0, t1 = df.date_start.min(), df.date_end.max()
    rc = rc[(rc.date >= t0 - pd.Timedelta("15min")) & (rc.date <= t1 + pd.Timedelta("5min"))]
    wx = wx[(wx.date >= t0 - pd.Timedelta("15min")) & (wx.date <= t1 + pd.Timedelta("5min"))]
    df = df.merge(drivers[["driver_number", "name_acronym", "team_name"]], on="driver_number", how="left")

    # tyre state
    # some stints lack lap_end (or lap_start): infer from the next stint / the driver's last lap
    stints = stints.sort_values(["driver_number", "stint_number"]).copy()
    last = df.groupby("driver_number").lap_number.max()
    nxt_start = stints.groupby("driver_number").lap_start.shift(-1)
    stints["lap_end"] = stints.lap_end.fillna(nxt_start - 1).fillna(stints.driver_number.map(last))
    stints = stints.dropna(subset=["lap_start", "lap_end"])
    stints = stints.astype({"lap_start": int, "lap_end": int})
    stints["tyre_age_at_start"] = stints.tyre_age_at_start.fillna(0)
    st = stints.assign(lap=lambda s: [list(range(a, b + 1)) for a, b in zip(s.lap_start, s.lap_end)]).explode("lap")
    st = st.dropna(subset=["lap"])  # stints with lap_end < lap_start explode to nothing
    st["lap"] = st.lap.astype(int)
    st["tyre_age"] = st.tyre_age_at_start + (st.lap - st.lap_start)
    df = df.merge(st[["driver_number", "lap", "stint_number", "compound", "tyre_age"]]
                  .rename(columns={"lap": "lap_number"}), on=["driver_number", "lap_number"], how="left")

    # exclusions
    pit_in = set(zip(pit.driver_number, pit.lap_number))
    df["pit_in"] = [(d, l) in pit_in for d, l in zip(df.driver_number, df.lap_number)]
    windows = neutralised_windows(rc, t1)
    df["neutralised"] = _overlaps(df.date_start, df.date_end, windows)
    restarts = [e for _, e, k in windows if k in ("SC", "RED")]
    df["restart_lap"] = _overlaps(df.date_start, df.date_end, [(r, r + pd.Timedelta("1s")) for r in restarts])
    df["double_yellow"] = _overlaps(df.date_start, df.date_end, double_yellow_windows(rc))
    df["start_lap"] = df.lap_number == 1
    blue = rc[rc.flag == "BLUE"].dropna(subset=["driver_number", "lap_number"])
    blue = set(zip(blue.driver_number.astype(int), blue.lap_number.astype(int)))
    df["blue_flag"] = [(d, l) in blue for d, l in zip(df.driver_number, df.lap_number)]
    df["wet_tyre"] = df.compound.isin(["INTERMEDIATE", "WET"])
    # pace backstop for neutralisations the message parser misses, and for damaged/spun cars
    by_lap = df.groupby("lap_number").lap_duration.median()
    green_ref = by_lap.quantile(0.25)
    df["field_slow_lap"] = df.lap_number.map(by_lap > 1.07 * green_ref).fillna(False)
    df["outlier_lap"] = df.lap_duration > 1.10 * green_ref
    df["clean"] = ~(df.pit_in | df.is_pit_out_lap | df.neutralised | df.restart_lap
                    | df.double_yellow | df.start_lap | df.blue_flag | df.lap_duration.isna()
                    | df.compound.isin(["UNKNOWN"]) | df.compound.isna()
                    | df.wet_tyre | df.field_slow_lap | df.outlier_lap)

    # on-road gaps at the start and end of each lap (end = next lap's start crossing)
    g = road_gaps(df[["driver_number", "lap_number", "date_start"]].dropna())
    g = g[["driver_number", "lap_number", "gap_ahead", "gap_behind", "car_ahead", "car_behind",
           "ahead_is_backmarker", "being_lapped"]]
    df = df.merge(g, on=["driver_number", "lap_number"], how="left")
    nxt = g.assign(lap_number=g.lap_number - 1).rename(
        columns={"gap_ahead": "gap_ahead_end", "gap_behind": "gap_behind_end", "car_ahead": "car_ahead_end", "car_behind": "car_behind_end",
                 "ahead_is_backmarker": "ahead_is_backmarker_end", "being_lapped": "being_lapped_end"})
    df = df.merge(nxt, on=["driver_number", "lap_number"], how="left")
    df["gap_ahead_mean"] = df[["gap_ahead", "gap_ahead_end"]].mean(axis=1)
    df["gap_behind_mean"] = df[["gap_behind", "gap_behind_end"]].mean(axis=1)
    df["same_car_ahead"] = df.car_ahead == df.car_ahead_end
    df["being_lapped"] = df.being_lapped.fillna(False) | df.being_lapped_end.fillna(False)
    df["clean"] &= ~df.being_lapped

    # track temperature (drop sensor zeros), nearest reading to lap midpoint
    wx = wx[wx.track_temperature > 5].sort_values("date")
    df["mid"] = (df.date_start + (df.date_end - df.date_start) / 2).fillna(df.date_start)
    timed = df.mid.notna()
    df = pd.concat([pd.merge_asof(df[timed].sort_values("mid"),
                                  wx[["date", "track_temperature", "air_temperature", "rainfall"]],
                                  left_on="mid", right_on="date", direction="nearest").drop(columns="date"),
                    df[~timed]], ignore_index=True).drop(columns="mid")
    df["clean"] &= ~(df.rainfall.fillna(0) > 0)
    return df.sort_values(["driver_number", "lap_number"]).reset_index(drop=True)


if __name__ == "__main__":
    import sys
    key = int(sys.argv[1]) if len(sys.argv) > 1 else 11404
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = build(key)
    out.to_parquet(PROCESSED / f"laps_{key}.parquet")
    print(out.shape, "clean:", out.clean.sum())
