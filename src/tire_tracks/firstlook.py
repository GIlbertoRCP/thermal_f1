"""First-look fixed-effects model: does following closely change lap/sector time?

y = driver×stint FE + tyre_age×compound + gap-ahead bins (ref: >3 s)   [gap-behind bins as placebo]

Within a stint tyre age and lap number move one-for-one, so tyre wear cannot be separated from
fuel burn + track evolution. The tyre_age slope is the *net* in-stint trend; no fuel correction.
SEs clustered by driver. Descriptive, not causal: being held up is not separated from aero loss yet.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
BINS = [0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf]
LABELS = ["<0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0-3.0", ">3.0"]
TARGETS = {"lap_duration": "Lap", "duration_sector_1": "S1 (main straight, T1-3)",
           "duration_sector_2": "S2 (Esses)", "duration_sector_3": "S3 (stadium)"}


def prepare(key: int) -> pd.DataFrame:
    d = pd.read_parquet(ROOT / "data" / "processed" / f"laps_{key}.parquet")
    d = d[d.clean & d.same_car_ahead].copy()
    d = d[d.groupby("driver_number").lap_number.transform("size") >= 10]
    d["stint_id"] = d.driver_number.astype(str) + "_" + d.stint_number.astype(int).astype(str)
    d["gap_bin"] = pd.Categorical(pd.cut(d.gap_ahead_mean, BINS, labels=LABELS), categories=LABELS[::-1])
    d["behind_bin"] = pd.Categorical(pd.cut(d.gap_behind_mean, BINS, labels=LABELS), categories=LABELS[::-1])
    return d.dropna(subset=["gap_bin", "behind_bin", "tyre_age"])


def fit(d: pd.DataFrame, y: str, term: str = "gap_bin"):
    f = f"{y} ~ C(stint_id) + tyre_age:C(compound) + C({term})"
    dd = d.dropna(subset=[y])
    m = smf.ols(f, dd).fit(cov_type="cluster", cov_kwds={"groups": dd.driver_number})
    rows = []
    for lab in LABELS[:-1]:
        k = f"C({term})[T.{lab}]"
        rows.append({"bin": lab, "coef": m.params[k], "lo": m.conf_int().loc[k, 0],
                     "hi": m.conf_int().loc[k, 1], "n": int((dd[term] == lab).sum())})
    return m, pd.DataFrame(rows)


def run(key: int = 11404):
    d = prepare(key)
    out = ROOT / "reports"
    print(f"laps used: {len(d)}  drivers: {d.driver_number.nunique()}  stints: {d.stint_id.nunique()}")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
    tables = []
    for ax, (y, name) in zip(axes, TARGETS.items()):
        m, t = fit(d, y)
        tables.append(t.assign(target=name))
        x = np.arange(len(t))
        ax.errorbar(x, t.coef, yerr=[t.coef - t.lo, t.hi - t.coef], fmt="o", capsize=3)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xticks(x, t.bin)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("gap to car ahead (s)")
    axes[0].set_ylabel("time vs. >3 s clean air (s)")
    fig.suptitle(f"Mexico City 2026: time delta by gap to car ahead (95% CI, n={len(d)} laps)")
    fig.tight_layout()
    fig.savefig(out / "gap_effect_by_sector.png", dpi=130)
    res = pd.concat(tables)

    # placebo: car behind should not matter. Restrict to laps in clean air ahead (>3 s)
    p = d[d.gap_bin == ">3.0"]
    _, plac = fit(p.assign(behind_bin=p.behind_bin.cat.remove_unused_categories()), "lap_duration", "behind_bin")

    m, _ = fit(d, "lap_duration")
    slope = {k: (m.params[k], m.bse[k]) for k in m.params.index if k.startswith("tyre_age")}
    res.to_csv(out / "gap_effect_by_sector.csv", index=False)
    return d, res, plac, slope


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    d, res, plac, slope = run()
    print(res.round(3).to_string(index=False))
    print("\nplacebo (gap behind, laps with >3 s clear ahead):\n", plac.round(3).to_string(index=False))
    print("\nnet in-stint slope s/lap (tyre wear + fuel burn + track evolution):",
          {k: tuple(round(v, 4) for v in x) for k, x in slope.items()})
