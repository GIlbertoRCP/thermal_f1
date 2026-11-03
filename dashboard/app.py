"""Tire Tracks dashboard: dirty air in 2026, from timing and telemetry only.

    PYTHONPATH=src uv run streamlit run dashboard/app.py

Reads the pipeline outputs (data/processed, reports/multirace); nothing here re-fits the models except
the bootstrap CIs for the filters you pick.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import streamlit as st

from tire_tracks import trackmap as tm
from tire_tracks.multirace import ADV_BINS, ADV_LABELS, GAP_BINS, GAP_LABELS, boot_mean

ROOT = Path(__file__).resolve().parents[1]
PROCESSED, REPORTS = ROOT / "data" / "processed", ROOT / "reports" / "multirace"

# reference palette (dataviz skill): series slots, diverging pair, neutral midpoint, chart ink
SERIES = ["#2a78d6", "#eb6834"]
GAIN, LOSS, MID = "#2a78d6", "#e34948", "#b4b2a9"
MUTED, GRID = "#898781", "rgba(137,135,129,0.25)"
LABELLED = {"Monte Carlo", "Zandvoort", "Singapore", "Hungaroring", "Spa-Francorchamps", "Monza", "Suzuka",
            "Mexico City"}  # direct labels for the tracks the story names; hover has the rest

st.set_page_config(page_title="Tire Tracks", layout="wide")
SURFACE = "#0e1117" if st.context.theme.type == "dark" else "#ffffff"  # Streamlit's page background


def style(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), hovermode="closest",
                      legend=dict(orientation="h", y=1.08, x=0), font=dict(family="system-ui, sans-serif"))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=MUTED, zerolinewidth=1)
    return fig


@st.cache_data
def close_laps() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "all_close_laps.parquet")


@st.cache_data
def clean_laps() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "all_clean_laps.parquet")


@st.cache_data
def per_race() -> pd.DataFrame:
    pr = pd.read_csv(REPORTS / "per_race_wake.csv")
    tc = pd.read_csv(REPORTS / "track_characteristics.csv")
    return pr.merge(tc[["race_key", "full_throttle_frac", "mean_speed_kmh"]], on="race_key")


def race_label(circuit: str, year: int) -> str:
    return f"{circuit} {year}"


# ---------------------------------------------------------------- page 1
def page_layout():
    st.header("Track layout sets whether following costs or pays")
    st.caption("Net lap penalty for followers that are not faster than the car ahead (so they can't be held up), "
               "starting the lap < 1.5 s behind, against how much of the pole lap is flat out. "
               "Bars are 95% CIs, clustered by driver. Weighted fit over 2026 races.")
    pr = per_race()
    c1, c2 = st.columns([1, 3])
    xvar = c1.radio("Track measure", ["full_throttle_frac", "mean_speed_kmh"],
                    format_func={"full_throttle_frac": "Full-throttle fraction", "mean_speed_kmh": "Mean speed (km/h)"}.get)
    show_old = c2.checkbox("Show Mexico City 2023-25 (previous rules, not in the fit)", value=True)

    x26 = pr[pr.year == 2026].copy()
    x26["se"] = (x26.hi - x26.lo) / (2 * 1.96)
    fit = smf.wls(f"wake_lap ~ {xvar}", x26, weights=1 / x26.se ** 2).fit()
    g = np.linspace(x26[xvar].min(), x26[xvar].max(), 50)
    band = fit.get_prediction(pd.DataFrame({xvar: g})).summary_frame(alpha=0.05)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[g, g[::-1]], y=np.r_[band.mean_ci_upper, band.mean_ci_lower[::-1]],
                             fill="toself", fillcolor="rgba(137,135,129,0.15)", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=g, y=band["mean"], mode="lines", line=dict(color=MUTED, width=2),
                             name="weighted fit (2026)", hoverinfo="skip"))
    groups = [("2026 races", x26, SERIES[0])]
    if show_old:
        groups.append(("Mexico City 2023-25", pr[pr.year < 2026], SERIES[1]))
    for name, d, col in groups:
        fig.add_trace(go.Scatter(
            x=d[xvar], y=d.wake_lap, mode="markers+text", name=name,
            text=[(r.circuit if r.circuit in LABELLED else "") if r.year == 2026 else str(r.year) for r in d.itertuples()],
            textposition="top center" if col == SERIES[0] else "middle right", textfont=dict(size=10, color=MUTED),
            marker=dict(size=10, color=col, line=dict(width=2, color=SURFACE)),
            error_y=dict(type="data", symmetric=False, array=d.hi - d.wake_lap, arrayminus=d.wake_lap - d.lo,
                         color=col, thickness=1, width=0),
            customdata=np.c_[d.circuit, d.year, d.lo, d.hi, d.n_not_faster],
            hovertemplate="<b>%{customdata[0]} %{customdata[1]}</b><br>penalty %{y:+.2f} s/lap "
                          "[%{customdata[2]:+.2f}, %{customdata[3]:+.2f}]<br>%{customdata[4]} laps<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_xaxes(title="full-throttle fraction of the pole lap" if xvar == "full_throttle_frac" else "pole-lap mean speed (km/h)")
    fig.update_yaxes(title="net wake penalty (s/lap)  ·  below 0 = following gains")
    st.plotly_chart(style(fig, 500), use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Slope", f"{fit.params.iloc[1]:+.2f} s/lap per unit" if xvar == "full_throttle_frac" else f"{fit.params.iloc[1] * 10:+.3f} s/lap per 10 km/h")
    m2.metric("p-value", f"{fit.pvalues.iloc[1]:.3f}")
    m3.metric("R²", f"{fit.rsquared:.2f}")
    with st.expander("Table"):
        st.dataframe(pr[["circuit", "year", "full_throttle_frac", "mean_speed_kmh", "n_not_faster", "wake_lap", "lo", "hi"]]
                     .sort_values("wake_lap").round(3), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------- page 2
def page_heldup():
    st.header("Being held up vs. the wake itself")
    st.caption("Lap time lost against the follower's own clean-air potential (pace model: qualifying anchor, "
               "compound, tyre age, lap number, driver). Laps that start < 1.5 s behind the car ahead. "
               "95% CIs clustered by race (by driver when one race is picked).")
    c = close_laps()
    c["race"] = [race_label(a, b) for a, b in zip(c.circuit, c.year)]
    f1, f2, f3 = st.columns([1, 2, 1])
    regs = f1.radio("Rules", ["2026", "2022-25"], format_func={"2026": "2026", "2022-25": "2022-25 (Mexico only)"}.get)
    races = sorted(c[c.regs == regs].race.unique())
    pick = f2.multiselect("Races (empty = all)", races)
    same = f3.checkbox("Same car ahead all lap", help="Drops laps with a position change")
    d = c[c.regs == regs]
    if pick:
        d = d[d.race.isin(pick)]
    if same:
        d = d[d.same_car_ahead]
    cluster = "race_key" if d.race_key.nunique() > 1 else "driver_number"

    rows = []
    for lab, g in d.assign(b=pd.cut(d.adv, ADV_BINS, labels=ADV_LABELS)).groupby("b", observed=True):
        m, lo, hi = boot_mean(g, "x_lap", cluster)
        rows.append({"group": lab, "n": len(g), "mean_adv": g.adv.mean(), "excess": m, "lo": lo, "hi": hi})
    hu = pd.DataFrame(rows)
    rows = []
    for lab, g in d[d.adv <= 0].assign(b=pd.cut(d.gap_ahead, GAP_BINS, labels=GAP_LABELS)).groupby("b", observed=True):
        m, lo, hi = boot_mean(g, "x_lap", cluster)
        rows.append({"start gap (s)": lab, "n": len(g), "excess": m, "lo": lo, "hi": hi})
    wg = pd.DataFrame(rows)

    left, right = st.columns(2)
    with left:
        st.subheader("By how much faster the follower is")
        fig = go.Figure(go.Bar(
            x=hu.group, y=hu.excess, name="time lost", marker=dict(color=SERIES[0], cornerradius=4),
            error_y=dict(type="data", symmetric=False, array=hu.hi - hu.excess, arrayminus=hu.excess - hu.lo,
                         color=MUTED, thickness=1),
            customdata=np.c_[hu.n, hu.mean_adv, hu.lo, hu.hi],
            hovertemplate="<b>%{x}</b><br>lost %{y:+.2f} s/lap [%{customdata[2]:+.2f}, %{customdata[3]:+.2f}]"
                          "<br>pace advantage %{customdata[1]:.2f} s<br>%{customdata[0]} laps<extra></extra>"))
        adv = hu[hu.mean_adv > 0]
        fig.add_trace(go.Scatter(x=adv.group, y=adv.mean_adv, mode="markers", name="follower's pace advantage",
                                 marker=dict(symbol="line-ew-open", size=28, line=dict(width=2, color=MUTED)),
                                 hovertemplate="pace advantage %{y:.2f} s<extra></extra>"))
        fig.update_yaxes(title="time lost vs own potential (s/lap)")
        fig.update_layout(showlegend=True, bargap=0.45)
        st.plotly_chart(style(fig), use_container_width=True)
        st.caption("Faster followers lose about their whole advantage: that's being held up, not aero loss.")
    with right:
        st.subheader("Followers that can't be held up, by start gap")
        fig = go.Figure(go.Bar(
            x=wg["start gap (s)"], y=wg.excess, marker=dict(color=SERIES[0], cornerradius=4),
            error_y=dict(type="data", symmetric=False, array=wg.hi - wg.excess, arrayminus=wg.excess - wg.lo,
                         color=MUTED, thickness=1),
            customdata=np.c_[wg.n, wg.lo, wg.hi],
            hovertemplate="<b>%{x} s</b><br>lost %{y:+.2f} s/lap [%{customdata[1]:+.2f}, %{customdata[2]:+.2f}]"
                          "<br>%{customdata[0]} laps<extra></extra>"))
        fig.update_yaxes(title="time lost vs own potential (s/lap)")
        fig.update_xaxes(title="gap to car ahead at the start of the lap (s)")
        fig.update_layout(bargap=0.45)
        st.plotly_chart(style(fig), use_container_width=True)
        st.caption("This is the wake on its own. Beyond 0.5 s it nets to about zero across tracks.")
    st.caption(f"{len(d):,} laps, {d.race_key.nunique()} races, {d.driver_number.nunique()} drivers.")
    with st.expander("Tables"):
        st.dataframe(hu.round(3), hide_index=True, use_container_width=True)
        st.dataframe(wg.round(3), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------- page 3
@st.cache_data
def speed_delta(key: int, max_gap: float, follower: str):
    prof, line = tm.load(key)
    c, d = close_laps(), clean_laps()
    c, d = c[c.race_key == key], d[d.race_key == key]
    free = d[(d.gap_ahead > 3) & (d.gap_ahead_end > 3)]
    close = c[c.gap_ahead < max_gap]
    if follower == "not faster":
        close = close[close.adv <= 0]
    return tm.wake_speed_delta(prof, line, close, free)


def page_map():
    st.header("Where on the lap the wake costs or pays")
    st.caption("Speed of close-following laps minus the same driver's free-air speed at that point (> 3 s clear), "
               "with the free-air profile scaled to the lap's expected pace so fuel and tyre age don't show as wake "
               "effects. ~4 Hz telemetry in 50 m bins: only averages over many laps mean anything. "
               "Use it for *where* time goes: at all five mapped races it matches the sector-time model's signs, "
               "but its totals run ~20-40% larger, so *how much* comes from the Held up vs. wake page.")
    c = close_laps()
    keys = [k for k in sorted(c.race_key.unique()) if (PROCESSED / f"profile_{k}.parquet").exists()]
    labels = {k: race_label(*c.loc[c.race_key == k, ["circuit", "year"]].iloc[0]) for k in keys}
    f1, f2, f3, f4 = st.columns([2, 1, 1, 2])
    key = f1.selectbox("Race", keys, index=keys.index(11404) if 11404 in keys else 0, format_func=labels.get)
    max_gap = f2.select_slider("Start gap under (s)", [0.5, 1.0, 1.5], value=1.0)
    follower = f3.radio("Followers", ["not faster", "all"], help="'Not faster' can't be held up, so what's left is the wake")
    others = [k for k in keys if k != key and labels[k].rsplit(" ", 1)[0] == labels[key].rsplit(" ", 1)[0]]
    cmp_key = f4.selectbox("Compare with (same track)", [None] + others,
                           format_func=lambda k: "—" if k is None else labels[k]) if others else None

    out, meta = speed_delta(key, max_gap, follower)
    lim = float(np.nanpercentile(np.abs(out.d_speed), 95)) or 1.0

    left, right = st.columns([3, 2])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=out.x, y=out.y, mode="lines", line=dict(color=GRID, width=14),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=out.x, y=out.y, mode="markers",
            marker=dict(size=13, color=out.d_speed, cmin=-lim, cmax=lim,
                        colorscale=[[0, LOSS], [0.5, MID], [1, GAIN]],
                        colorbar=dict(title="km/h", thickness=10, len=0.6)),
            customdata=np.c_[out.dist, out.d_speed, out.lo, out.hi, out.d_throttle, out.n_laps, out.sector],
            hovertemplate="<b>%{customdata[6]}, %{customdata[0]:.0f} m</b><br>speed %{customdata[1]:+.1f} km/h "
                          "[%{customdata[2]:+.1f}, %{customdata[3]:+.1f}]<br>throttle %{customdata[4]:+.0f} pts"
                          "<br>%{customdata[5]} laps<extra></extra>", showlegend=False))
        start = out.iloc[0]
        fig.add_annotation(x=start.x, y=start.y, text="start/finish", showarrow=True, arrowhead=0,
                           font=dict(size=10, color=MUTED), arrowcolor=MUTED)
        for name, d in (("S1 | S2", out.loc[out.sector.eq("S2").idxmax()]),
                        ("S2 | S3", out.loc[out.sector.eq("S3").idxmax()])):
            fig.add_annotation(x=d.x, y=d.y, text=name, showarrow=True, arrowhead=0, font=dict(size=10, color=MUTED),
                               arrowcolor=MUTED, ax=25, ay=-25)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False, scaleanchor="x")
        st.plotly_chart(style(fig, 600), use_container_width=True)
        st.caption("Blue: faster than in free air (the tow). Red: slower (lost downforce, earlier lift).")
    with right:
        st.subheader("Time it adds up to")
        for r in meta.itertuples():
            st.metric(r.sector, f"{r.time_s:+.2f} s", help=f"95% CI [{r.lo:+.2f}, {r.hi:+.2f}], driver-clustered")
        cov = f" Telemetry covers {out.coverage.iloc[0]:.0%} of clean laps." if "coverage" in out else ""
        st.caption(f"{meta.n_laps.iloc[0]} close laps from {meta.n_drivers.iloc[0]} drivers. "
                   f"Positive = slower than in free air.{cov}")

    st.subheader("Speed difference along the lap")
    fig = go.Figure()
    series = [(labels[key], out, SERIES[0])]
    if cmp_key is not None:
        series.append((labels[cmp_key], speed_delta(cmp_key, max_gap, follower)[0], SERIES[1]))
    for name, o, col in series:
        fig.add_trace(go.Scatter(x=np.r_[o.dist, o.dist[::-1]], y=np.r_[o.hi, o.lo[::-1]], fill="toself",
                                 fillcolor=col, opacity=0.15, line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=o.dist, y=o.d_speed, mode="lines", name=name, line=dict(color=col, width=2),
                                 customdata=np.c_[o.lo, o.hi, o.sector],
                                 hovertemplate=f"<b>{name}</b> %{{customdata[2]}}, %{{x:.0f}} m<br>%{{y:+.1f}} km/h "
                                               "[%{customdata[0]:+.1f}, %{customdata[1]:+.1f}]<extra></extra>"))
    b1, b2, end = out.s1_end.iloc[0], out.s2_end.iloc[0], out.dist.max()
    for b in (b1, b2):
        fig.add_vline(x=b, line=dict(color=MUTED, width=1, dash="dot"))
    for name, mid in (("S1", b1 / 2), ("S2", (b1 + b2) / 2), ("S3", (b2 + end) / 2)):
        fig.add_annotation(x=mid, y=1.02, yref="paper", text=name, showarrow=False, font=dict(size=11, color=MUTED))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_xaxes(title="distance from the line (m, approx.)")
    fig.update_yaxes(title="speed vs own free air (km/h)")
    fig.update_layout(hovermode="x unified", showlegend=len(series) > 1, legend=dict(y=1.18))
    st.plotly_chart(style(fig, 380), use_container_width=True)
    if any(labels[k].endswith(("2023", "2024", "2025")) for k in [key, cmp_key] if k is not None):
        st.caption("2023-25 cars had DRS (open when < 1 s behind at the detection point), so their straight-line "
                   "gain is tow + DRS. 2026 cars have no DRS.")
    with st.expander("Table"):
        st.dataframe(out[["bin", "dist", "sector", "d_speed", "lo", "hi", "d_throttle", "n_laps"]].round(2),
                     hide_index=True, use_container_width=True)
        st.dataframe(meta.round(3), hide_index=True, use_container_width=True)


PAGES = {"Track layout": page_layout, "Held up vs. wake": page_heldup, "Track map": page_map}
with st.sidebar:
    st.title("Tire Tracks")
    page = st.radio("Page", list(PAGES), label_visibility="collapsed")
    st.caption("Dirty air in F1, 2026 season + Mexico City 2023-25. OpenF1 timing and telemetry; "
               "see reports/multirace_results.md for methods.")
PAGES[page]()
