"""Tyre-damage tests: does the wake penalty accumulate, and does it outlast the wake?

Aero loss and being held up are instantaneous; overheating builds up and persists. Two tests:

1. Accumulation inside the wake (enough data): within a run of consecutive green laps < CLOSE s
   behind the *same* car, regress the per-sector gap change to that car (dgap_s, follower minus
   leader sector time, same lap) on lap-in-run, controlling for the sector potential difference
   and the start-of-lap gap. Overheating -> dgap_s2/s3 grows with lap-in-run.
   Same regression on the follower's excess vs its own clean-air potential.
2. Exit events (reported, too few in one race): runs >= 3 laps ending because the leader pitted or
   the follower passed. Exits where the gap simply opened are excluded: they are selected on the
   follower's own slow lap.

The proposal's kernel E_t = λE_{t-1} + 1(g<2)exp(-αg) cannot be identified from one race: λ needs
post-exposure laps, and there are 8 clean exits. Revisit with multiple races.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from tire_tracks.heldup import TARGETS, load, potentials
from tire_tracks.openf1 import read

ROOT = Path(__file__).resolve().parents[2]
CLOSE, MIN_RUN = 1.5, 3


def runs(race_key: int) -> pd.DataFrame:
    """Lap-in-run index for consecutive green laps < CLOSE s behind the same car, within a stint."""
    a = pd.read_parquet(ROOT / "data" / "processed" / f"laps_{race_key}.parquet")
    a = a.sort_values(["driver_number", "lap_number"]).reset_index(drop=True)
    green = ~(a.neutralised | a.restart_lap | a.start_lap | a.pit_in | a.is_pit_out_lap)
    a["close"] = (a.gap_ahead_mean < CLOSE) & green & a.same_car_ahead
    new = (~a.close) | (a.car_ahead != a.groupby("driver_number").car_ahead.shift()) \
        | (a.stint_number != a.groupby("driver_number").stint_number.shift()) \
        | (a.lap_number != a.groupby("driver_number").lap_number.shift() + 1)
    a["run_id"] = new.cumsum()
    a["lap_in_run"] = a.groupby("run_id").close.cumsum().where(a.close, 0)
    a["run_len"] = a.groupby("run_id").lap_in_run.transform("max")
    return a[["driver_number", "lap_number", "run_id", "lap_in_run", "run_len"]]


def exit_events(race_key: int) -> pd.DataFrame:
    a = pd.read_parquet(ROOT / "data" / "processed" / f"laps_{race_key}.parquet").merge(
        runs(race_key), on=["driver_number", "lap_number"])
    pit = read(race_key, "pit")
    pits = set(zip(pit.driver_number, pit.lap_number))
    ends = a[(a.lap_in_run >= MIN_RUN) & (a.lap_in_run == a.run_len)]
    nxt = a.set_index(["driver_number", "lap_number"])
    rows = []
    for _, r in ends.iterrows():
        # the run ends on its last close lap; the pass or the leader's stop shows up on it or the lap after
        laps = [r.lap_number, r.lap_number + 1]
        own_pit = {(r.driver_number, l) for l in laps + [r.lap_number + 2]} & pits
        after = [nxt.loc[(r.driver_number, l)] for l in laps if (r.driver_number, l) in nxt.index]
        if any((r.car_ahead, l) in pits for l in laps) and not own_pit:
            cause = "leader pitted"
        elif any(x.car_behind_end == r.car_ahead or x.car_behind == r.car_ahead for x in after):
            cause = "follower passed"
        else:
            continue
        rows.append({"driver": r.name_acronym, "lap": int(r.lap_number), "run_laps": int(r.run_len), "cause": cause})
    return pd.DataFrame(rows)


def accumulation(race_key: int = 11404, quali_key: int = 11400):
    d, _ = potentials(load(race_key, quali_key))
    d = d.merge(runs(race_key), on=["driver_number", "lap_number"])
    cols = ["driver_number", "lap_number"] + [f"P_{s}" for s in TARGETS.values()] + list(TARGETS)
    lead = d[cols].rename(columns=lambda c: c if c == "lap_number" else f"L_{c}")
    c = d[(d.lap_in_run >= 1) & (d.run_len >= MIN_RUN)].merge(
        lead, left_on=["car_ahead", "lap_number"], right_on=["L_driver_number", "lap_number"])
    rows = []
    for y, s in TARGETS.items():
        c[f"dgap_{s}"] = c[y] - c[f"L_{y}"]
        c[f"dP_{s}"] = c[f"P_{s}"] - c[f"L_P_{s}"]
        for outcome, label in ((f"dgap_{s}", "gap change to leader"), (f"x_{s}", "excess vs own potential")):
            f = f"{outcome} ~ lap_in_run + dP_{s} + gap_ahead"
            cc = c.dropna(subset=[outcome, f"dP_{s}"])
            m = smf.ols(f, cc).fit(cov_type="cluster", cov_kwds={"groups": cc.driver_number})
            ci = m.conf_int().loc["lap_in_run"]
            rows.append({"sector": s, "outcome": label, "per_lap_in_wake": m.params.lap_in_run,
                         "lo": ci[0], "hi": ci[1], "p": m.pvalues.lap_in_run, "n": len(cc),
                         "runs": cc.run_id.nunique()})
    return c, pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    c, acc = accumulation()
    ev = exit_events(11404)
    acc.to_csv(ROOT / "reports" / "wake_accumulation.csv", index=False)
    print(f"laps inside runs >= {MIN_RUN}: {len(c)}, runs: {c.run_id.nunique()}, drivers: {c.driver_number.nunique()},"
          f" max lap_in_run: {c.lap_in_run.max()}")
    print("\nchange per extra consecutive lap in the wake (s/lap), controls: potential diff, start gap:\n",
          acc.round(4).to_string(index=False))
    print(f"\nclean exit events (n={len(ev)}):\n", ev.to_string(index=False))
