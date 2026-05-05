import functools
import json
import os

_POOL_PATH = os.path.join(os.path.dirname(__file__), "data", "pool.json")


@functools.lru_cache(maxsize=None)
def load_pool() -> tuple[list[str], list[str], list[str]]:
    """Return (items, first_names, surnames) loaded from data/pool.json."""
    with open(_POOL_PATH, encoding="utf-8") as f:
        d = json.load(f)
    return d["items"], d["first_names"], d["surnames"]
