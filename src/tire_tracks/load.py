"""Session loading with a project-local FastF1 cache."""
from pathlib import Path

import fastf1

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"


def load_session(year: int, event: str | int, kind: str = "R", **kwargs):
    CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE))
    session = fastf1.get_session(year, event, kind)
    session.load(**{"laps": True, "telemetry": True, "weather": True, "messages": True, **kwargs})
    return session
