"""Race/qualifying session pairs for the multi-race study, from OpenF1's session list."""
from pathlib import Path

import httpx
import pandas as pd

from tire_tracks.openf1 import BASE, ROOT, _get, download_session

CATALOG = ROOT / "data" / "catalog.csv"
# Autódromo Hermanos Rodríguez, Interlagos, Las Vegas, Spielberg are the only venues above ~500 m
ALTITUDE_M = {"Mexico City": 2240, "Interlagos": 760, "Spielberg": 680, "Las Vegas": 620, "Madring": 650}


def build_catalog(years=(2023, 2024, 2025, 2026), until: str = "2026-11-02") -> pd.DataFrame:
    with httpx.Client(timeout=60) as c:
        s = pd.concat([_get(c, "sessions", year=y) for y in years], ignore_index=True)
    s = s[(pd.to_datetime(s.date_start) < pd.Timestamp(until, tz="UTC")) & ~s.is_cancelled.astype(bool)]
    races = s[s.session_name == "Race"]
    quali = s[s.session_name == "Qualifying"][["meeting_key", "session_key"]].rename(columns={"session_key": "quali_key"})
    cat = races.merge(quali, on="meeting_key", how="left")
    cat = cat[(cat.year == 2026) | (cat.circuit_short_name == "Mexico City")]
    cat = cat.rename(columns={"session_key": "race_key", "circuit_short_name": "circuit"})
    cat["altitude_m"] = cat.circuit.map(ALTITUDE_M).fillna(0).astype(int)
    cat["regs"] = (cat.year >= 2026).map({True: "2026", False: "2022-25"})
    cat = cat[["race_key", "quali_key", "meeting_key", "year", "circuit", "country_name", "date_start",
               "altitude_m", "regs"]].sort_values("date_start").reset_index(drop=True)
    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    cat.to_csv(CATALOG, index=False)
    return cat


def catalog() -> pd.DataFrame:
    return pd.read_csv(CATALOG)


if __name__ == "__main__":
    cat = catalog() if CATALOG.exists() else build_catalog()
    print(cat.to_string())
    for _, r in cat.iterrows():
        download_session(int(r.race_key), telemetry=False)
        download_session(int(r.quali_key), telemetry=False)
        print("downloaded", r.year, r.circuit, flush=True)
