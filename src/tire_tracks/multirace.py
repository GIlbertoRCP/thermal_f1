"""Multi-race study: same per-race pipeline (lap table -> clean-air potential -> held-up split),
pooled across all 2026 races plus Mexico City 2023-25 (previous regulations).

Sector definitions differ by circuit, so cross-track comparisons are lap-level; sector splits are only
compared within the same circuit (Mexico across eras).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from tire_tracks import laps as lapmod
from tire_tracks.catalog import catalog
from tire_tracks.heldup import TARGETS, gap_dynamics, load, potentials
from tire_tracks.openf1 import read
from tire_tracks.persistence import MIN_RUN, exit_events, runs

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports" / "multirace"
ADV_BINS = [-np.inf, 0, 0.3, 0.6, np.inf]
ADV_LABELS = ["not faster", "faster 0-0.3", "faster 0.3-0.6", "faster >0.6"]
GAP_BINS = [0, 0.5, 1.0, 1.5]
GAP_LABELS = ["<0.5", "0.5-1.0", "1.0-1.5"]


def build_all(cat: pd.DataFrame) -> pd.DataFrame:
    lapmod.PROCESSED.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in cat.iterrows():
        d = lapmod.build(int(r.race_key))
        d.to_parquet(lapmod.PROCESSED / f"laps_{int(r.race_key)}.parquet")
        rows.append({"race_key": r.race_key, "year": r.year, "circuit": r.circuit, "laps": len(d),
                     "max_lap": d.lap_number.max(), "clean": int(d.clean.sum()),
                     "neutralised": int(d.neutralised.sum()), "field_slow": int(d.field_slow_lap.sum()),
                     "field_slow_not_neutralised": int((d.field_slow_lap & ~d.neutralised).sum()),
                     "wet_laps": int(d.wet_tyre.sum()), "rain_laps": int((d.rainfall.fillna(0) > 0).sum()),
                     "close_<1.5": int((d.clean & (d.gap_ahead < 1.5)).sum()),
                     "free_>3": int((d.clean & (d.gap_ahead_mean > 3)).sum())})
    return pd.DataFrame(rows)


def race_pairs(r) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Per race: all clean laps with potentials, and start-gap close laps joined to the leader."""
    d, _ = potentials(load(int(r.race_key), int(r.quali_key)))
    _, c = gap_dynamics(d)
    c = c[c.gap_ahead < GAP_BINS[-1]].copy()
    c["adv"] = c.L_P_lap - c.P_lap
    for df in (d, c):
        df["race_key"], df["circuit"], df["year"], df["regs"] = r.race_key, r.circuit, r.year, r.regs
        df["altitude_m"] = r.altitude_m
    return d, c


def boot_mean(df: pd.DataFrame, col: str, cluster: str, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    groups = [g[col].dropna().to_numpy() for _, g in df.groupby(cluster)]
    groups = [g for g in groups if len(g)]
    if not groups:
        return np.nan, np.nan, np.nan
    sums = np.array([g.sum() for g in groups]); ns = np.array([len(g) for g in groups])
    idx = rng.integers(0, len(groups), (n, len(groups)))
    means = sums[idx].sum(1) / ns[idx].sum(1)
    return sums.sum() / ns.sum(), *np.quantile(means, [0.025, 0.975])


def pooled_heldup(c: pd.DataFrame) -> pd.DataFrame:
    c = c.assign(adv_bin=pd.cut(c.adv, ADV_BINS, labels=ADV_LABELS))
    rows = []
    for lab, g in c.groupby("adv_bin", observed=True):
        m, lo, hi = boot_mean(g, "x_lap", "race_key")
        rows.append({"adv_bin": lab, "n": len(g), "races": g.race_key.nunique(), "mean_adv": g.adv.mean(),
                     "excess_lap": m, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def wake_by_gap(c: pd.DataFrame) -> pd.DataFrame:
    """Lap-level penalty for followers that cannot be held up, by start gap."""
    nf = c[c.adv <= 0].assign(gap_bin=lambda x: pd.cut(x.gap_ahead, GAP_BINS, labels=GAP_LABELS))
    rows = []
    for lab, g in nf.groupby("gap_bin", observed=True):
        m, lo, hi = boot_mean(g, "x_lap", "race_key")
        rows.append({"start_gap": lab, "n": len(g), "races": g.race_key.nunique(), "excess_lap": m, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def per_race_wake(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (key, circ, yr), g in c.groupby(["race_key", "circuit", "year"]):
        nf, hu = g[g.adv <= 0], g[g.adv > 0.3]
        m, lo, hi = boot_mean(nf, "x_lap", "driver_number")
        row = {"race_key": key, "circuit": circ, "year": yr, "altitude_m": g.altitude_m.iloc[0],
               "n_not_faster": len(nf), "wake_lap": m, "lo": lo, "hi": hi,
               "n_faster_0.3": len(hu), "heldup_share": (hu.x_lap / hu.adv).median() if len(hu) else np.nan}
        if circ == "Mexico City":
            for s in ("s1", "s2", "s3"):
                row[f"wake_{s}"], row[f"{s}_lo"], row[f"{s}_hi"] = boot_mean(nf, f"x_{s}", "driver_number")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["year", "race_key"])


def pooled_persistence(cat: pd.DataFrame, dall: pd.DataFrame, k_max: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean-air laps after a wake exit the follower did not cause by a slow lap (leader pit / pass).
    The run's last close lap is e.lap; the exit happens during lap e.lap + 1, which starts in the wake
    (k = 0, the transition lap: the pass or the leader's in-lap). Post-exit laps are k >= 1 and must
    start and end > 2 s from the car ahead. Excess is vs potential; because potentials include these
    laps, effects are slightly attenuated."""
    evs, post = [], []
    for _, r in cat.iterrows():
        key = int(r.race_key)
        ev = exit_events(key)
        if ev.empty:
            continue
        a = pd.read_parquet(lapmod.PROCESSED / f"laps_{key}.parquet")[["driver_number", "name_acronym", "lap_number", "stint_number"]]
        ev = ev.merge(a.drop_duplicates(["name_acronym"])[["name_acronym", "driver_number"]], left_on="driver", right_on="name_acronym")
        ev["race_key"], ev["circuit"], ev["regs"] = key, r.circuit, r.regs
        evs.append(ev)
        d = dall[dall.race_key == key]
        for _, e in ev.iterrows():
            stint = a[(a.driver_number == e.driver_number) & (a.lap_number == e.lap)].stint_number
            if stint.empty:
                continue
            after = d[(d.driver_number == e.driver_number) & (d.stint_number == stint.iloc[0])
                      & (d.lap_number > e.lap) & (d.lap_number <= e.lap + 1 + k_max)]
            post.append(after.assign(k=after.lap_number - e.lap - 1, cause=e.cause, run_laps=e.run_laps,
                                     race_key=key, event=f"{key}_{e.driver}_{e.lap}"))
    ev, post = pd.concat(evs, ignore_index=True), pd.concat(post, ignore_index=True)
    free = (post.gap_ahead > 2.0) & (post.gap_ahead_end > 2.0)  # out of the wake for the whole lap
    post = post[free | (post.k == 0)]
    return ev, post


def persistence_table(post: pd.DataFrame, dall: pd.DataFrame) -> pd.DataFrame:
    ref = dall[(dall.gap_ahead_mean > 3)]
    rows = [{"k": "reference (all free-air laps)", "n": len(ref), "events": np.nan,
             **dict(zip(["x_lap", "lo", "hi"], boot_mean(ref, "x_lap", "race_key")))}]
    for k, g in post.groupby("k"):
        rows.append({"k": "0 (exit lap, starts in wake)" if k == 0 else int(k), "n": len(g), "events": g.event.nunique(),
                     **dict(zip(["x_lap", "lo", "hi"], boot_mean(g, "x_lap", "race_key")))})
    for lab, g in post[post.k == 0].groupby("cause"):
        rows.append({"k": f"0, {lab}", "n": len(g), "events": g.event.nunique(),
                     **dict(zip(["x_lap", "lo", "hi"], boot_mean(g, "x_lap", "race_key")))})
    early = post[post.k.between(1, 2)]
    for lab, g in early.groupby(np.where(early.run_laps >= 8, "k1-2, run >= 8 laps", "k1-2, run 3-7 laps")):
        rows.append({"k": lab, "n": len(g), "events": g.event.nunique(),
                     **dict(zip(["x_lap", "lo", "hi"], boot_mean(g, "x_lap", "race_key")))})
    for lab, g in early.groupby("cause"):
        rows.append({"k": f"k1-2, {lab}", "n": len(g), "events": g.event.nunique(),
                     **dict(zip(["x_lap", "lo", "hi"], boot_mean(g, "x_lap", "race_key")))})
    return pd.DataFrame(rows)


def accumulation_pooled(dall: pd.DataFrame, cat: pd.DataFrame) -> pd.DataFrame:
    """Per extra consecutive lap behind the same car: change in follower excess vs own potential (lap level)."""
    frames = []
    for _, r in cat.iterrows():
        key = int(r.race_key)
        d = dall[dall.race_key == key].merge(runs(key), on=["driver_number", "lap_number"])
        lead = d[["driver_number", "lap_number", "P_lap", "lap_duration"]].rename(
            columns={"driver_number": "car_ahead", "P_lap": "L_P_lap", "lap_duration": "L_lap"})
        c = d[(d.lap_in_run >= 1) & (d.run_len >= MIN_RUN)].merge(lead, on=["car_ahead", "lap_number"])
        frames.append(c)
    c = pd.concat(frames, ignore_index=True)
    c["adv"] = c.L_P_lap - c.P_lap
    c["dgap_lap"] = c.lap_duration - c.L_lap
    c["race_driver"] = c.race_key.astype(str) + "_" + c.driver_number.astype(str)
    rows = []
    for regs, g in [("all", c)] + list(c.groupby("regs")):
        for y in ("x_lap", "dgap_lap"):
            m = smf.ols(f"{y} ~ lap_in_run + adv + gap_ahead + C(race_key)", g).fit(
                cov_type="cluster", cov_kwds={"groups": g.race_driver})
            ci = m.conf_int().loc["lap_in_run"]
            rows.append({"regs": regs, "outcome": y, "per_lap_in_wake": m.params.lap_in_run, "lo": ci[0], "hi": ci[1],
                         "p": m.pvalues.lap_in_run, "n": len(g), "runs": g.groupby(["race_key", "run_id"]).ngroups})
    return pd.DataFrame(rows)


def run():
    REPORTS.mkdir(parents=True, exist_ok=True)
    cat = catalog()
    summary = build_all(cat)
    summary.to_csv(REPORTS / "race_summary.csv", index=False)
    ds, cs, failed = [], [], []
    for _, r in cat.iterrows():
        try:
            d, c = race_pairs(r)
            ds.append(d); cs.append(c)
        except Exception as e:  # noqa: BLE001 — report and continue; a race without quali/potential is skipped
            failed.append((r.circuit, r.year, repr(e)[:120]))
    dall, call = pd.concat(ds, ignore_index=True), pd.concat(cs, ignore_index=True)
    dall.to_parquet(lapmod.PROCESSED / "all_clean_laps.parquet")
    call.to_parquet(lapmod.PROCESSED / "all_close_laps.parquet")
    return cat, summary, dall, call, failed


def plot_per_race(pr: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt
    pr = pr.sort_values("wake_lap").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(pr) + 1.5))
    y = np.arange(len(pr))
    col = np.where(pr.circuit == "Mexico City", np.where(pr.year == 2026, "crimson", "darkorange"), "steelblue")
    ax.errorbar(pr.wake_lap, y, xerr=[pr.wake_lap - pr.lo, pr.hi - pr.wake_lap], fmt="none", ecolor="grey", lw=0.8)
    ax.scatter(pr.wake_lap, y, c=col, zorder=3)
    ax.set_yticks(y, [f"{c} {yr} (n={n})" for c, yr, n in zip(pr.circuit, pr.year, pr.n_not_faster)], fontsize=8)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("lap time vs own clean-air potential (s), followers not faster than the car ahead, start gap <1.5 s")
    ax.set_title("Net wake penalty by race (red: Mexico 2026, orange: Mexico 2023-25)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def track_regression(pr: pd.DataFrame, tc: pd.DataFrame, path: Path | None = None):
    """Per-race net wake penalty vs full-throttle fraction (2026 races), inverse-variance weighted.
    Mexico's residual is its altitude/layout-specific effect; with one high-altitude venue it can't be
    attributed to altitude alone."""
    import matplotlib.pyplot as plt
    x = pr.merge(tc[["race_key", "full_throttle_frac", "mean_speed_kmh"]], on="race_key")
    x["se"] = (x.hi - x.lo) / (2 * 1.96)
    x26 = x[x.year == 2026].copy()
    fits = {}
    for var in ("full_throttle_frac", "mean_speed_kmh"):
        fits[var] = smf.wls(f"wake_lap ~ {var}", x26, weights=1 / x26.se ** 2).fit()
    loo = smf.wls("wake_lap ~ full_throttle_frac", x26[x26.circuit != "Mexico City"],
                  weights=1 / x26[x26.circuit != "Mexico City"].se ** 2).fit()
    mex = x26[x26.circuit == "Mexico City"].iloc[0]
    pred = loo.get_prediction(pd.DataFrame({"full_throttle_frac": [mex.full_throttle_frac]})).summary_frame(alpha=0.05)
    if path is not None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(x26.full_throttle_frac, x26.wake_lap, yerr=[x26.wake_lap - x26.lo, x26.hi - x26.wake_lap],
                    fmt="o", color="steelblue", ecolor="lightgrey", capsize=2)
        m = x26.circuit == "Mexico City"
        ax.scatter(x26[m].full_throttle_frac, x26[m].wake_lap, color="crimson", zorder=5, s=60, label="Mexico City 2026")
        for _, r in x26.iterrows():
            ax.annotate(r.circuit, (r.full_throttle_frac, r.wake_lap), fontsize=7, xytext=(3, 3), textcoords="offset points")
        g = np.linspace(x26.full_throttle_frac.min(), x26.full_throttle_frac.max(), 50)
        f = fits["full_throttle_frac"]
        ax.plot(g, f.params.Intercept + f.params.full_throttle_frac * g, color="black", lw=1, label="weighted fit")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel("full-throttle fraction of the pole lap (qualifying telemetry)")
        ax.set_ylabel("net wake penalty, followers not faster (s/lap)")
        ax.set_title("2026: net wake penalty vs how much of the lap is flat out", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=130)
    return x, fits, mex, pred


if __name__ == "__main__":
    pd.set_option("display.width", 240)
    cat, summary, dall, call, failed = run()
    print("races:", len(cat), " failed:", failed)
    print(summary.to_string(index=False))
    print(f"\npooled clean laps {len(dall)}, close laps (start gap <1.5 s) {len(call)}")

    for regs, g in [("all", call)] + list(call.groupby("regs")):
        print(f"\n[{regs}] excess vs own potential by pace advantage (CI clustered by race):")
        print(pooled_heldup(g).round(3).to_string(index=False))
    c26 = call[call.regs == "2026"]
    print("\n[2026] followers not faster, by start gap:\n", wake_by_gap(c26).round(3).to_string(index=False))
    pr = per_race_wake(call)
    pr.to_csv(REPORTS / "per_race_wake.csv", index=False)
    print("\nper race:\n", pr.round(3).to_string(index=False))
    plot_per_race(pr, REPORTS / "per_race_wake.png")

    acc = accumulation_pooled(dall, cat)
    acc.to_csv(REPORTS / "accumulation_pooled.csv", index=False)
    print("\naccumulation (per extra lap behind same car):\n", acc.round(4).to_string(index=False))

    ev, post = pooled_persistence(cat, dall)
    ev.to_csv(REPORTS / "exit_events.csv", index=False)
    pt = persistence_table(post, dall)
    pt.to_csv(REPORTS / "persistence_pooled.csv", index=False)
    print(f"\nexit events: {len(ev)}  ({ev.cause.value_counts().to_dict()})")
    print(pt.round(3).to_string(index=False))

    from tire_tracks.trackchar import characteristics
    tc = characteristics()
    tc.to_csv(REPORTS / "track_characteristics.csv", index=False)
    _, fits, mex, _ = track_regression(pr, tc, REPORTS / "wake_vs_full_throttle.png")
    f = fits["full_throttle_frac"]
    print(f"\nwake vs full-throttle fraction (2026): slope {f.params.iloc[1]:.3f} p={f.pvalues.iloc[1]:.4f} R2={f.rsquared:.2f}")
