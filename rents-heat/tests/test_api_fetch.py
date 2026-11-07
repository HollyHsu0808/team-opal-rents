"""API client against a mocked session. No network."""
import os

import pandas as pd
import pytest

from rents import api_fetch as af

KEY = "test-secret-key-123"


class Resp:
    def __init__(self, code, body=None, headers=None):
        self.status_code, self._body, self.headers = code, body, headers or {}

    def json(self):
        return self._body


class FakeSession:
    """Serves rent-history pages for a fixed number of adverts per suburb; scripted status codes per suburb."""

    def __init__(self, totals, script=None):
        self.totals, self.script, self.calls = totals, dict(script or {}), []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        assert headers["Authorization"] == f"Bearer {KEY}"
        sub = [s for s in self.totals if af.quote(s, safe="") in url][0]
        seq = self.script.get(sub)
        if seq:
            code = seq.pop(0)
            if code != 200:
                return Resp(code, {"detail": "x"}, {"Retry-After": "0.5"} if code == 429 else {})
        total, off, lim = self.totals[sub], params["offset"], params["limit"]
        n = max(0, min(lim, total - off))
        res = [{"address": f"UNIT {i}, 1 A STREET, X NSW 2000", "gnaf_pid": f"G{i}", "weekly_rent": 500.0,
                "listing_date": "2026-08-02" if i == total - 1 else "2026-07-31",
                "bedrooms": 1, "bathrooms": 1, "car_spaces": 0, "leased": True, "lat": -33.9, "lng": 151.2,
                "property_type": "Unit"} for i in range(off, off + n)]
        return Resp(200, {"data": {"area_level": "suburb", "area_name": sub,
                                   "meta": {"limit": lim, "offset": off, "returned": n, "total": total},
                                   "results": res}})


def client(sess, logs, sleeps):
    return af.Client(KEY, session=sess, sleep=sleeps.append, log=logs.append)


def test_paging_and_cutoff(tmp_path):
    logs, sleeps = [], []
    s = FakeSession({"Waterloo (NSW)": 2500})
    rep = af.fetch(client(s, logs, sleeps), ["Waterloo (NSW)"], str(tmp_path), endpoints=("rent-history",), log=logs.append)
    assert [p["offset"] for _, p in s.calls] == [0, 1000, 2000]
    assert all(p["to_date"] == "2026-07-31" and p["limit"] == 1000 for _, p in s.calls)
    assert "/v1/suburbs/Waterloo%20%28NSW%29/listings/rent-history" in s.calls[0][0]
    assert rep["suburbs"]["Waterloo (NSW)"]["rent-history"]["dropped_after_cutoff"] == 1
    df = pd.read_parquet(tmp_path / "listings_rent_history.parquet")
    assert len(df) == 2499 and pd.to_datetime(df["listing_date"]).max() <= pd.Timestamp("2026-07-31")


def test_429_backoff_then_success(tmp_path):
    logs, sleeps = [], []
    s = FakeSession({"Glebe (NSW)": 10}, script={"Glebe (NSW)": [429, 429, 200]})
    rep = af.fetch(client(s, logs, sleeps), ["Glebe (NSW)"], str(tmp_path), endpoints=("rent-history",), log=logs.append)
    assert rep["suburbs"]["Glebe (NSW)"]["rent-history"]["rows"] == 9
    assert sleeps.count(0.5) == 2 and len(s.calls) == 3


def test_403_recorded_and_other_suburbs_continue(tmp_path):
    logs, sleeps = [], []
    s = FakeSession({"Newtown (NSW)": 5, "Ultimo (NSW)": 5}, script={"Newtown (NSW)": [403]})
    rep = af.fetch(client(s, logs, sleeps), ["Newtown (NSW)", "Ultimo (NSW)"], str(tmp_path),
                   endpoints=("rent-history",), log=logs.append)
    assert rep["outside_403"] == [["Newtown (NSW)", "rent-history"]]
    assert rep["suburbs"]["Ultimo (NSW)"]["rent-history"]["rows"] == 4


def test_gives_up_after_retries():
    s = FakeSession({"Sydney (NSW)": 5}, script={"Sydney (NSW)": [500] * 10})
    c = af.Client(KEY, session=s, sleep=lambda x: None, log=lambda x: None, max_retries=2)
    with pytest.raises(af.ApiError):
        list(c.paged(af.suburb_path("Sydney (NSW)", "listings/rent-history"), {"to_date": "2026-07-31"}))
    assert len(s.calls) == 3


def test_key_never_logged_or_written(tmp_path, capsys):
    logs, sleeps = [], []
    s = FakeSession({"Waterloo (NSW)": 30}, script={"Waterloo (NSW)": [429, 200]})
    c = client(s, logs, sleeps)
    af.fetch(c, ["Waterloo (NSW)"], str(tmp_path), endpoints=("rent-history",), log=logs.append)
    assert KEY not in repr(c)
    assert all(KEY not in m for m in logs)
    assert KEY not in capsys.readouterr().out
    for root, _, files in os.walk(tmp_path):
        for f in files:
            with open(os.path.join(root, f), "rb") as fh:
                assert KEY.encode() not in fh.read()


def test_load_key_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("MICROBURBS_API_KEY", raising=False)
    p = tmp_path / ".env"
    p.write_text(f"# comment\nMICROBURBS_API_KEY={KEY}\n")
    assert af.load_key(str(p)) == KEY
    assert af.load_key(str(tmp_path / "missing.env")) is None
    with pytest.raises(ValueError):
        af.Client(None, session=FakeSession({}))


def test_dry_run_makes_no_calls(capsys, monkeypatch):
    import requests
    monkeypatch.setattr(requests.Session, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    assert af.main(["--dry-run", "--limit-suburbs", "2", "--suburbs-from", "config"]) == 0
    out = capsys.readouterr().out
    assert out.count("GET /v1/suburbs/") == 4 and "to_date=2026-07-31" in out
