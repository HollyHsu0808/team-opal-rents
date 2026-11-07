import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "synthetic"))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def small_world(tmp_path_factory):
    """A 600-flat synthetic world in a temp dir and a config whose synthetic paths point at it. No network."""
    from make_synthetic import make
    from rents.load import cfg
    d = tmp_path_factory.mktemp("syn")
    make(n_flats=600, seed=11, out=str(d))
    c = cfg()
    c["paths"]["synthetic"] = {
        "input": str(d / "rents-input-flats-city-of-sydney.csv"),
        "past_leases": str(d / "rents-past-leases-city-of-sydney.csv"),
        "api_rent_history": str(d / "api" / "listings_rent_history.parquet"),
        "scoring": str(d / "scoring_adverts_2026-08_09.csv"),
        "out": str(d / "out")}
    os.makedirs(d / "out", exist_ok=True)
    return c
