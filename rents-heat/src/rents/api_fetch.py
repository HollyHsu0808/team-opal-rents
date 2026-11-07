"""Microburbs Hackathon API client for the rents task: suburb rent-history adverts and suburb properties.

  GET /v1/suburbs/{suburb}/listings/rent-history?from_date&to_date&limit<=1000&offset
  GET /v1/suburbs/{suburb}/properties?limit<=1000&offset
Envelope {"data": {"results": [...], "meta": {limit, offset, returned, total}}}; no data -> 200 with data null;
outside City of Sydney -> 403 (recorded, not fatal); ~10 req/s; 429 -> back off and retry.

The key comes from env MICROBURBS_API_KEY or rents-heat/data_private/.env. It is never printed or logged.
Writes raw JSON pages to data_private/api/raw/ and flattened parquet to data_private/api/.
Run:  .venv/Scripts/python.exe src/rents/api_fetch.py [--dry-run] [--limit-suburbs N] [--endpoints rent-history,properties]
"""
import argparse
import json
import os
import sys
import time
from urllib.parse import quote

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_BASE = "https://api-ht.microburbs.com.au"
TO_DATE = "2026-07-31"  # task cut-off, applied server-side and re-checked client-side


class OutsideAreaError(Exception):
    pass


class ApiError(Exception):
    pass


def load_key(env_file=None):
    """Return the key or None. Never print it."""
    k = os.environ.get("MICROBURBS_API_KEY")
    if k:
        return k.strip()
    env_file = env_file or os.path.join(ROOT, "data_private", ".env")
    if os.path.exists(env_file):
        with open(env_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("MICROBURBS_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


class Client:
    def __init__(self, key, base_url=None, session=None, rate_per_s=9.0, max_retries=6, timeout=30,
                 sleep=time.sleep, log=print):
        if not key:
            raise ValueError("No API key: set MICROBURBS_API_KEY or write it to rents-heat/data_private/.env")
        self._headers = {"Authorization": f"Bearer {key}"}  # kept private; never logged
        self.base_url = (base_url or os.environ.get("MICROBURBS_API_BASE") or DEFAULT_BASE).rstrip("/")
        if session is None:
            import requests
            session = requests.Session()
        self.session = session
        self.min_interval = 1.0 / rate_per_s
        self.max_retries = max_retries
        self.timeout = timeout
        self.sleep = sleep
        self.log = log
        self._last = 0.0
        self.n_calls = 0

    def __repr__(self):
        return f"Client(base_url={self.base_url!r}, key=<hidden>)"

    def _throttle(self):
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            self.sleep(wait)
        self._last = time.monotonic()

    def get(self, path, params=None):
        url = f"{self.base_url}{path}"
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self.n_calls += 1
            resp = self.session.get(url, params=params, headers=self._headers, timeout=self.timeout)
            code = resp.status_code
            if code == 200:
                return resp.json()
            if code == 403:
                raise OutsideAreaError(f"403 for {path}")
            if code == 429 or code >= 500:
                ra = resp.headers.get("Retry-After") if hasattr(resp, "headers") and resp.headers else None
                try:
                    wait = float(ra) if ra else min(60.0, 1.0 * 2 ** attempt)
                except ValueError:
                    wait = min(60.0, 1.0 * 2 ** attempt)
                self.log(f"  {code} on {path} (attempt {attempt + 1}); waiting {wait:.1f}s")
                self.sleep(wait)
                continue
            raise ApiError(f"{code} for {path}")  # body not echoed: it could contain request details
        raise ApiError(f"gave up after {self.max_retries + 1} attempts: {path}")

    def paged(self, path, params=None, limit=1000):
        """Yield raw page envelopes until offset >= total or a page returns nothing."""
        offset = 0
        while True:
            p = dict(params or {}, limit=limit, offset=offset)
            body = self.get(path, p)
            data = (body or {}).get("data")
            if not data:
                yield body
                return
            yield body
            meta = data.get("meta") or {}
            returned = meta.get("returned", len(data.get("results") or []))
            total = meta.get("total")
            offset += returned
            if returned == 0 or (total is not None and offset >= total):
                return


def suburb_path(suburb, endpoint):
    return f"/v1/suburbs/{quote(suburb, safe='')}/{endpoint}"


def flatten_pages(pages, suburb):
    rows = []
    for body in pages:
        data = (body or {}).get("data") or {}
        for r in data.get("results") or []:
            rows.append(r)
    df = pd.DataFrame(rows)
    df["suburb_query"] = suburb
    return df


def suburbs_from_input(input_csv):
    s = pd.read_csv(input_csv, usecols=["suburb"], dtype=str)["suburb"].dropna().unique()
    return sorted(s)


def fetch(client, suburbs, out_dir, endpoints=("rent-history",), to_date=TO_DATE, from_date=None, log=print):
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    report = {"to_date": to_date, "suburbs": {}, "outside_403": []}
    frames = {e: [] for e in endpoints}
    for s in suburbs:
        for e in endpoints:
            ep = "listings/rent-history" if e == "rent-history" else "properties"
            params = {"to_date": to_date} if e == "rent-history" else {}
            if e == "rent-history" and from_date:
                params["from_date"] = from_date
            try:
                pages = list(client.paged(suburb_path(s, ep), params))
            except OutsideAreaError:
                log(f"  403 outside area: {s} ({e})")
                report["outside_403"].append([s, e])
                continue
            slug = s.replace(" ", "_").replace("(", "").replace(")", "")
            with open(os.path.join(raw_dir, f"{e}_{slug}.json"), "w", encoding="utf-8") as f:
                json.dump(pages, f)
            df = flatten_pages(pages, s)
            n_late = 0
            if e == "rent-history" and len(df) and "listing_date" in df:
                late = pd.to_datetime(df["listing_date"], errors="coerce") > pd.Timestamp(to_date)
                n_late = int(late.sum())
                df = df[~late]
            frames[e].append(df)
            report["suburbs"].setdefault(s, {})[e] = {"rows": int(len(df)), "pages": len(pages), "dropped_after_cutoff": n_late}
            log(f"  {s} {e}: {len(df)} rows, {len(pages)} pages")
    names = {"rent-history": "listings_rent_history.parquet", "properties": "properties.parquet"}
    for e, fr in frames.items():
        if fr:
            df = pd.concat(fr, ignore_index=True)
            for c in df.columns:
                if df[c].dtype == object:
                    df[c] = df[c].astype(str).where(df[c].notna(), None)
            df.to_parquet(os.path.join(out_dir, names[e]), index=False)
            report[f"{e}_rows"] = int(len(df))
    report["n_calls"] = client.n_calls
    with open(os.path.join(out_dir, "fetch_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


def main(argv=None):
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from rents.load import cfg
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the planned calls; no key, no network")
    ap.add_argument("--limit-suburbs", type=int, default=None)
    ap.add_argument("--endpoints", default="rent-history,properties")
    ap.add_argument("--suburbs-from", default="input", choices=["input", "config"],
                    help="input: unique `suburb` values of the real input file (falls back to config list)")
    ap.add_argument("--from-date", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "data_private", "api"))
    a = ap.parse_args(argv)
    c = cfg()
    inp = os.path.join(ROOT, c["paths"]["real"]["input"])
    if a.suburbs_from == "input" and os.path.exists(inp):
        suburbs = suburbs_from_input(inp)
    else:
        suburbs = c["paths"].get("api_suburbs_29") or [s + c["paths"]["api_suburb_suffix"] for s in c["paths"]["suburbs_33"]]
    if a.limit_suburbs:
        suburbs = suburbs[:a.limit_suburbs]
    endpoints = tuple(e.strip() for e in a.endpoints.split(",") if e.strip())
    if a.dry_run:
        for s in suburbs:
            for e in endpoints:
                ep = "listings/rent-history" if e == "rent-history" else "properties"
                q = f"?to_date={TO_DATE}&limit=1000&offset=0.." if e == "rent-history" else "?limit=1000&offset=0.."
                print(f"GET {suburb_path(s, ep)}{q}")
        print(f"dry run: {len(suburbs)} suburbs x {len(endpoints)} endpoints; key present: {load_key() is not None}")
        return 0
    client = Client(load_key())
    rep = fetch(client, suburbs, a.out, endpoints)
    print(json.dumps({k: v for k, v in rep.items() if k != "suburbs"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
