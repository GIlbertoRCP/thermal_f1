"""Circuit characteristics measured from data: the pole lap's telemetry in qualifying."""
import httpx
import numpy as np
import pandas as pd

from tire_tracks.catalog import catalog
from tire_tracks.openf1 import RAW, _get, read


def pole_lap_profile(quali_key: int) -> dict:
    path = RAW / str(quali_key) / "pole_car_data.parquet"
    q = read(quali_key, "laps").dropna(subset=["lap_duration", "date_start"])
    best = q.loc[q.lap_duration.idxmin()]
    if path.exists():
        car = pd.read_parquet(path)
    else:
        with httpx.Client(timeout=120) as c:
            car = _get(c, "car_data", session_key=quali_key, driver_number=int(best.driver_number))
        car.to_parquet(path)
    car["date"] = pd.to_datetime(car.date, format="ISO8601", utc=True)
    a, b = best.date_start, best.date_start + pd.Timedelta(seconds=best.lap_duration)
    lap = car[(car.date >= a) & (car.date < b)].sort_values("date")
    t = (lap.date - a).dt.total_seconds().to_numpy()
    dt = np.diff(t, append=best.lap_duration)
    full = (lap.throttle.clip(upper=100) >= 98).to_numpy()
    return {"pole_lap_s": best.lap_duration, "full_throttle_frac": float((dt * full).sum() / dt.sum()),
            "mean_speed_kmh": float((lap.speed * dt).sum() / dt.sum()), "samples": len(lap)}


def characteristics() -> pd.DataFrame:
    rows = []
    for _, r in catalog().iterrows():
        rows.append({"race_key": r.race_key, "circuit": r.circuit, "year": r.year, **pole_lap_profile(int(r.quali_key))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(characteristics().round(3).to_string(index=False))
