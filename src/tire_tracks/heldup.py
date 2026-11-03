"""Separate 'held up by a slower car' from the follower's own wake penalty.

1. Clean-air potential P: mixed model on free-air laps (gap ahead > 3 s), per sector:
     y ~ quali_best_y + compound + tyre_age:compound + lap_number + (1 | driver)
   Pit stops reset tyre age, so across stints tyre wear and race progress are separable.
   Qualifying anchors drivers with few free-air laps (BLUP shrinks toward it).
2. For close laps (gap ahead < 1.5 s), compare follower and leader on the same lap:
     excess_f = actual_f - P_f,  excess_l = actual_l - P_l,  adv = P_l - P_f  (>0: follower faster)
   Held up      -> follower matches the leader: excess_f ≈ excess_l + adv  (slope ~1 on adv>0)
   Wake penalty -> excess_f > 0 even when the follower is NOT faster (adv <= 0)
   Relative-to-leader wake effect per sector: excess_f,s - excess_l,s (track state cancels).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from tire_tracks.openf1 import read

ROOT = Path(__file__).resolve().parents[2]
TARGETS = {"lap_duration": "lap", "duration_sector_1": "s1", "duration_sector_2": "s2", "duration_sector_3": "s3"}
FREE_AIR = 3.0
MIN_FREE_PER_COMPOUND = 15
CLOSE = 1.5


def quali_bests(quali_key: int) -> pd.DataFrame:
    q = read(quali_key, "laps").dropna(subset=["lap_duration"])
    b = q.groupby("driver_number")[list(TARGETS)].min()
    return b.add_prefix("q_").reset_index()


def load(race_key: int, quali_key: int) -> pd.DataFrame:
    d = pd.read_parquet(ROOT / "data" / "processed" / f"laps_{race_key}.parquet")
    d = d[d.clean & d.tyre_age.notna()].merge(quali_bests(quali_key), on="driver_number", how="inner")
    d = d[d.groupby("driver_number").lap_number.transform("size") >= 10].copy()
    d["free"] = d.gap_ahead_mean > FREE_AIR
    # a compound needs enough free-air laps to estimate its own level and wear slope
    ok = d[d.free].compound.value_counts()
    d = d[d.compound.isin(ok[ok >= MIN_FREE_PER_COMPOUND].index)].copy()
    return d


def _formula(y: str) -> str:
    return f"{y} ~ q_{y} + C(compound) + tyre_age:C(compound) + lap_number"


def fit_potential(free: pd.DataFrame, y: str):
    f = free.dropna(subset=[y])
    return smf.mixedlm(_formula(y), f, groups=f.driver_number).fit(reml=True)


def predict(m, d: pd.DataFrame) -> np.ndarray:
    re = {g: v.iloc[0] for g, v in m.random_effects.items()}
    return m.predict(d).to_numpy() + d.driver_number.map(re).fillna(0).to_numpy()


def potentials(d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    free, models = d[d.free], {}
    for y, s in TARGETS.items():
        models[s] = fit_potential(free, y)
        d[f"P_{s}"] = predict(models[s], d)
        d[f"x_{s}"] = d[y] - d[f"P_{s}"]
    return d, models


def cv_potential(d: pd.DataFrame, k: int = 5, seed: int = 0) -> pd.DataFrame:
    """Grouped K-fold over driver-stints on free-air laps: is P a usable baseline?"""
    free = d[d.free].copy()
    stints = free.driver_number.astype(str) + "_" + free.stint_number.astype(str)
    rng = np.random.default_rng(seed)
    fold = stints.map(dict(zip(u := stints.unique(), rng.integers(0, k, len(u)))))
    rows = []
    for y, s in TARGETS.items():
        err, err_q = [], []
        for i in range(k):
            tr, te = free[fold != i].dropna(subset=[y]), free[fold == i].dropna(subset=[y])
            m = fit_potential(tr, y)
            err.append(te[y] - predict(m, te))
            mq = smf.ols(_formula(y), tr).fit()  # quali only, no driver effect
            err_q.append(te[y] - mq.predict(te))
        e, eq = pd.concat(err), pd.concat(err_q)
        rows.append({"target": s, "rmse_mixed": np.sqrt((e ** 2).mean()), "rmse_quali_only": np.sqrt((eq ** 2).mean()),
                     "sd_y": free[y].std(), "n": len(e)})
    return pd.DataFrame(rows)


def pairs(d: pd.DataFrame) -> pd.DataFrame:
    """Close-following laps joined with the leader's same lap."""
    cols = ["driver_number", "lap_number"] + [f"{p}_{s}" for p in ("P", "x") for s in TARGETS.values()] + list(TARGETS)
    lead = d[cols].rename(columns=lambda c: c if c == "lap_number" else f"L_{c}")
    c = d[(d.gap_ahead_mean < CLOSE) & d.same_car_ahead & ~d.ahead_is_backmarker.fillna(False)]
    c = c.merge(lead, left_on=["car_ahead", "lap_number"], right_on=["L_driver_number", "lap_number"])
    c["adv"] = c.L_P_lap - c.P_lap
    for s in TARGETS.values():
        c[f"rel_{s}"] = c[f"x_{s}"] - c[f"L_x_{s}"]  # follower's loss relative to the leader, same lap
    c["adv_pos"], c["adv_neg"] = c.adv.clip(lower=0), c.adv.clip(upper=0)
    return c


def cluster_boot(df: pd.DataFrame, col: str, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    groups = [g[col].dropna().to_numpy() for _, g in df.groupby("driver_number")]
    means = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        v = np.concatenate([groups[i] for i in pick])
        means.append(v.mean())
    return df[col].mean(), *np.quantile(means, [0.025, 0.975])


ADV_BINS = [-np.inf, -0.3, 0.0, 0.3, 0.6, np.inf]
ADV_LABELS = ["follower slower >0.3", "slower 0-0.3", "faster 0-0.3", "faster 0.3-0.6", "faster >0.6"]


def run(race_key: int = 11404, quali_key: int = 11400):
    d, models = potentials(load(race_key, quali_key))
    cv = cv_potential(d)
    c = pairs(d)
    c["adv_bin"] = pd.cut(c.adv, ADV_BINS, labels=ADV_LABELS)

    by_adv = []
    for lab, g in c.groupby("adv_bin", observed=True):
        row = {"adv_bin": lab, "n": len(g), "drivers": g.driver_number.nunique()}
        for s in ("lap", "s1", "s2", "s3"):
            m, lo, hi = cluster_boot(g, f"x_{s}")
            row[f"excess_{s}"] = f"{m:+.2f} [{lo:+.2f},{hi:+.2f}]"
        by_adv.append(row)
    by_adv = pd.DataFrame(by_adv)

    # structural check: held up => excess_f ≈ excess_l + adv for adv > 0
    reg = smf.ols("x_lap ~ adv_pos + adv_neg + L_x_lap", c).fit(cov_type="cluster", cov_kwds={"groups": c.driver_number})

    # wake penalty where the follower cannot be held up (not faster than the leader)
    nh = c[c.adv <= 0]
    wake = []
    for s in ("lap", "s1", "s2", "s3"):
        for col, kind in ((f"x_{s}", "vs own potential"), (f"rel_{s}", "vs leader, same lap")):
            m, lo, hi = cluster_boot(nh, col)
            wake.append({"sector": s, "measure": kind, "mean": m, "lo": lo, "hi": hi})
    wake = pd.DataFrame(wake)

    out = ROOT / "reports"
    by_adv.to_csv(out / "heldup_by_advantage.csv", index=False)
    wake.to_csv(out / "wake_penalty_not_held_up.csv", index=False)
    return d, c, models, cv, by_adv, reg, wake, nh


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    d, c, models, cv, by_adv, reg, wake, nh = run()
    m = models["lap"]
    print("potential model (lap):\n", m.summary().tables[1].iloc[:, :4].to_string())
    print("\ncross-validated potential (free-air laps, grouped by stint):\n", cv.round(3).to_string(index=False))
    print(f"\nclose laps with leader matched: {len(c)}  drivers: {c.driver_number.nunique()}")
    print("\nfollower excess vs own potential, by pace advantage over leader:\n", by_adv.to_string(index=False))
    print("\nheld-up regression x_lap ~ adv_pos + adv_neg + leader excess:\n",
          pd.DataFrame({"coef": reg.params, "se": reg.bse}).round(3).to_string())
    print(f"\nwake penalty where follower is not faster (adv<=0): n={len(nh)}, drivers={nh.driver_number.nunique()}")
    print(wake.round(3).to_string(index=False))


# --- within-lap gap dynamics ("accordion") -------------------------------------------------
# Over a lap that starts and ends close, follower lap ≈ leader lap by construction, so lap-level
# leader comparisons are mechanical. Where the gap opens/closes *within* the lap is not.
# dgap_s = follower sector time - leader sector time (same lap) = change in gap across sector s.
# Regress on the sector potential difference dP_s = P_f,s - P_l,s: unconstrained cars -> slope ~1;
# gap-bin terms are the wake effect relative to 1.5-3 s behind.
DYN_BINS = [0, 0.5, 1.0, 1.5, 3.0]
DYN_LABELS = ["<0.5", "0.5-1.0", "1.0-1.5", "1.5-3.0"]


def gap_dynamics(d: pd.DataFrame) -> pd.DataFrame:
    cols = ["driver_number", "lap_number"] + [f"P_{s}" for s in TARGETS.values()] + list(TARGETS)
    lead = d[cols].rename(columns=lambda c: c if c == "lap_number" else f"L_{c}")
    c = d[(d.gap_ahead < 3.0) & ~d.ahead_is_backmarker.fillna(False)]  # start-of-lap gap only
    c = c.merge(lead, left_on=["car_ahead", "lap_number"], right_on=["L_driver_number", "lap_number"])
    c["start_bin"] = pd.Categorical(pd.cut(c.gap_ahead, DYN_BINS, labels=DYN_LABELS), categories=DYN_LABELS[::-1])
    rows = []
    for y, s in TARGETS.items():
        c[f"dgap_{s}"] = c[y] - c[f"L_{y}"]
        c[f"dP_{s}"] = c[f"P_{s}"] - c[f"L_P_{s}"]
        cc = c.dropna(subset=[f"dgap_{s}", f"dP_{s}", "start_bin"])
        m = smf.ols(f"dgap_{s} ~ dP_{s} + C(start_bin)", cc).fit(
            cov_type="cluster", cov_kwds={"groups": cc.driver_number})
        rows.append({"sector": s, "term": "slope on potential diff", "coef": m.params[f"dP_{s}"],
                     "lo": m.conf_int().loc[f"dP_{s}", 0], "hi": m.conf_int().loc[f"dP_{s}", 1], "n": len(cc)})
        for lab in DYN_LABELS[:-1]:
            k = f"C(start_bin)[T.{lab}]"
            rows.append({"sector": s, "term": f"start gap {lab}", "coef": m.params[k],
                         "lo": m.conf_int().loc[k, 0], "hi": m.conf_int().loc[k, 1],
                         "n": int((cc.start_bin == lab).sum())})
    return pd.DataFrame(rows), c


def start_gap_excess(d: pd.DataFrame, c_dyn: pd.DataFrame) -> pd.DataFrame:
    """Follower excess vs own potential, conditioning only on start-of-lap gap (< 1.5 s)."""
    c = c_dyn[c_dyn.gap_ahead < CLOSE].copy()
    c["adv"] = c.L_P_lap - c.P_lap
    c["adv_bin"] = pd.cut(c.adv, [-np.inf, 0, 0.3, 0.6, np.inf],
                          labels=["not faster", "faster 0-0.3", "faster 0.3-0.6", "faster >0.6"])
    rows = []
    for lab, g in c.groupby("adv_bin", observed=True):
        row = {"adv_bin": lab, "n": len(g), "drivers": g.driver_number.nunique(), "mean_adv": g.adv.mean()}
        for s in TARGETS.values():
            m, lo, hi = cluster_boot(g, f"x_{s}")
            row[s] = f"{m:+.2f} [{lo:+.2f},{hi:+.2f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def plot_heldup(c_dyn: pd.DataFrame, path: Path):
    import matplotlib.pyplot as plt
    c = c_dyn[c_dyn.gap_ahead < CLOSE].copy()
    c["adv"] = c.L_P_lap - c.P_lap
    labels = ["not faster", "faster 0-0.3", "faster 0.3-0.6", "faster >0.6"]
    c["adv_bin"] = pd.cut(c.adv, [-np.inf, 0, 0.3, 0.6, np.inf], labels=labels)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sectors = [("s1", "S1 straight"), ("s2", "S2 Esses"), ("s3", "S3 stadium"), ("lap", "Lap")]
    w = 0.2
    for j, (s, name) in enumerate(sectors):
        stats = [cluster_boot(g, f"x_{s}") for _, g in c.groupby("adv_bin", observed=True)]
        m = np.array([x[0] for x in stats]); lo = np.array([x[1] for x in stats]); hi = np.array([x[2] for x in stats])
        x = np.arange(len(stats)) + (j - 1.5) * w
        ax.bar(x, m, w, label=name)
        ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="none", ecolor="black", capsize=2, lw=0.8)
    means = c.groupby("adv_bin", observed=True).adv.mean()
    ax.scatter(np.arange(len(means)) + 1.5 * w, means.clip(lower=0), marker="_", s=300, color="black",
               zorder=5, label="pace advantage (held-up prediction)")
    ax.axhline(0, color="grey", lw=0.8)
    ns = c.groupby("adv_bin", observed=True).size()
    ax.set_xticks(np.arange(len(ns)), [f"{l}\n(n={n})" for l, n in zip(ns.index, ns)])
    ax.set_xlabel("follower's clean-air pace advantage over the car ahead (s/lap)")
    ax.set_ylabel("time lost vs own clean-air potential (s)")
    ax.set_title("Mexico City 2026, laps starting <1.5 s behind: being held up vs wake penalty")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
