"""Download OpenF1 session data to local parquet files.

FastF1 3.9 needs an F1 account for live-timing data, so OpenF1 is the primary
source. Results (grid, finish, status) still come from FastF1/Jolpica.
"""
import time
from pathlib import Path

import httpx
import pandas as pd

BASE = "https://api.openf1.org/v1"
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"

SESSION_ENDPOINTS = ["drivers", "laps", "stints", "pit", "intervals", "position",
                     "race_control", "weather", "overtakes", "session_result",
                     "starting_grid"]
DRIVER_ENDPOINTS = ["car_data", "location"]


REQUEST_PAUSE_S = 0.6  # OpenF1 free tier rate-limits bursts


def _get(client: httpx.Client, endpoint: str, **params) -> pd.DataFrame:
    for attempt in range(8):
        time.sleep(REQUEST_PAUSE_S)
        try:
            r = client.get(f"{BASE}/{endpoint}", params=params)
        except httpx.TransportError:
            time.sleep(min(5 * 2 ** attempt, 120))
            continue
        if r.status_code == 404:
            return pd.DataFrame()
        if r.status_code == 429 or r.status_code >= 500:
            wait = float(r.headers.get("retry-after", 0) or 0)
            time.sleep(max(wait, min(5 * 2 ** attempt, 120)))
            continue
        r.raise_for_status()
        return _fix_mixed(pd.DataFrame(r.json()))
    r.raise_for_status()
    return pd.DataFrame()


def _fix_mixed(df: pd.DataFrame) -> pd.DataFrame:
    # e.g. intervals.gap_to_leader mixes floats with "+1 LAP"; store those as strings
    for col in df.columns[df.dtypes == object]:
        kinds = {type(v) for v in df[col].dropna()}
        if len(kinds) > 1 and not kinds & {list, dict}:
            df[col] = df[col].map(lambda v: None if v is None else str(v))
    return df


def session_key(year: int, country: str, session_name: str = "Race") -> int:
    with httpx.Client(timeout=60) as c:
        df = _get(c, "sessions", year=year, country_name=country, session_name=session_name)
    return int(df["session_key"].iloc[0])


def download_session(key: int, telemetry: bool = True, force: bool = False) -> Path:
    out = RAW / str(key)
    out.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=180) as c:
        for ep in SESSION_ENDPOINTS:
            path = out / f"{ep}.parquet"
            if path.exists() and not force:
                continue
            _get(c, ep, session_key=key).to_parquet(path)
        if not telemetry:
            return out
        drivers = pd.read_parquet(out / "drivers.parquet")["driver_number"].unique()
        for ep in DRIVER_ENDPOINTS:
            path = out / f"{ep}.parquet"
            if path.exists() and not force:
                continue
            frames = [_get(c, ep, session_key=key, driver_number=int(d)) for d in drivers]
            pd.concat(frames, ignore_index=True).to_parquet(path)
    return out


def read(key: int, endpoint: str) -> pd.DataFrame:
    df = pd.read_parquet(RAW / str(key) / f"{endpoint}.parquet")
    for col in ("date", "date_start"):
        if col in df:
            df[col] = pd.to_datetime(df[col], format="ISO8601", utc=True)
    return df
