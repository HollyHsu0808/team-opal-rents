import json
import os

import pandas as pd

import run_all
from rents.load import path_for, read_input


def test_final_output_contract(small_world, monkeypatch):
    c = small_world
    monkeypatch.setattr(run_all, "cfg", lambda: c)
    assert run_all.main(["--data", "synthetic", "--window", "final", "--team", "pytest"]) == 0
    out = os.path.join(c["paths"]["synthetic"]["out"], "rents_pytest.csv")
    sub = pd.read_csv(out, dtype={"gnaf_id": str})
    flats = read_input(path_for(c, "synthetic", "input"), c)
    assert run_all.check_contract(sub, flats["gnaf_id"]) == []
    assert list(sub.columns) == ["gnaf_id", "rent_with_car_space", "rent_without_car_space"]
    assert len(sub) == len(flats) and sub["gnaf_id"].is_unique
    assert ((sub["rent_with_car_space"] * 2) % 1 == 0).all()
    with open(os.path.join(c["paths"]["synthetic"]["out"], "metrics_final.json")) as f:
        m = json.load(f)
    assert m["contract"] == "pass" and m["synthetic_scoring"]


def test_window_A_then_B_write_metrics(small_world, monkeypatch):
    c = small_world
    monkeypatch.setattr(run_all, "cfg", lambda: c)
    assert run_all.main(["--data", "synthetic", "--window", "A"]) == 0
    assert run_all.main(["--data", "synthetic", "--window", "B"]) == 0
    for w in ("A", "B"):
        with open(os.path.join(c["paths"]["synthetic"]["out"], f"metrics_{w}.json")) as f:
            m = json.load(f)
        for v in m["variants"].values():
            assert set(v) >= {"own_history", "no_history", "all"}
            assert set(v["all"]) == {"n", "medae", "within100", "r2", "coverage"}


def test_check_contract_catches_violations():
    ids = ["G1", "G2", "G3"]
    ok = pd.DataFrame({"gnaf_id": ids, "rent_with_car_space": [500.0, 610.5, 700.0],
                       "rent_without_car_space": [450.0, 560.0, 650.5]})
    assert run_all.check_contract(ok, ids) == []
    dup = pd.concat([ok, ok.iloc[[0]]])
    assert any("duplicate" in e for e in run_all.check_contract(dup, ids))
    assert any("missing" in e for e in run_all.check_contract(ok.iloc[:2], ids))
    extra = pd.concat([ok, pd.DataFrame({"gnaf_id": ["X9"], "rent_with_car_space": [1.0], "rent_without_car_space": [1.0]})])
    assert any("extra" in e for e in run_all.check_contract(extra, ids))
    blank = ok.copy()
    blank.loc[1, "rent_without_car_space"] = None
    assert any("blank" in e for e in run_all.check_contract(blank, ids))
    odd = ok.copy()
    odd.loc[0, "rent_with_car_space"] = 500.25
    assert any("0.50" in e for e in run_all.check_contract(odd, ids))
    assert run_all.check_contract(ok[["gnaf_id", "rent_with_car_space"]], ids)
