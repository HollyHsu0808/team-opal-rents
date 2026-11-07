#!/usr/bin/env python
"""Microburbs hackathon, Task 4 (Team-Opal): run the whole rent pipeline end to end on one computer.

What it does, in order (stops at the first failure and says which step failed):
  0. checks Python (3.11+) and the packages the pipeline imports
  1. puts the three official CSVs into rents-heat/data_private/real/ (copies them from --data-dir or --zip)
  2. puts the API key into rents-heat/data_private/.env (from --key, the MICROBURBS_API_KEY variable, or an existing .env)
  3. pulls every rental advert up to 31 Jul 2026 from the Microburbs API (skipped with --skip-fetch or when already pulled)
  4. window A: chooses and freezes the rule on May adverts (train <= 30 Apr), incl. the public-data ablation
  5. window B: evaluates the frozen rule ONCE on Jun-Jul adverts (train <= 31 May) -> the number we report
  6. final: trains on everything <= 31 Jul and writes rents_<team>.csv for all 65,777 flats (contract-checked)
  7. public-layer insight table (window-B residuals) and report.html
  8. copies rents_<team>.csv, report.html and frozen_rule.json into ./submission/ and prints the headline numbers

Usage (from the repository root, any OS):
  python run_everything.py --zip path/to/task4-rents.zip --key <TEAM_KEY>
  python run_everything.py --data-dir path/to/folder/with/the/3/csvs
  python run_everything.py                      # files and key already in rents-heat/data_private/
  python run_everything.py --skip-fetch         # reuse the API pull already on disk
  python run_everything.py --adverts past_leases   # train on the supplied past-leases file only

First-time setup (Windows PowerShell):
  py -3.13 -m venv rents-heat\\.venv
  rents-heat\\.venv\\Scripts\\python.exe -m pip install -r requirements-freeze.txt
  rents-heat\\.venv\\Scripts\\python.exe run_everything.py --zip <zip> --key <key>
Optional: LiDAR rasters for the point-height feature:  gh release download data-2026-11-07 -D public-data\\_derived\\lidar_rasters

Nothing here prints the API key. Organiser files, API output and outputs stay in rents-heat/data_private/ (gitignored);
only our own predictions and report are copied to ./submission/.
"""
import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
RH = os.path.join(HERE, "rents-heat")
PRIV = os.path.join(RH, "data_private")
REAL = os.path.join(PRIV, "real")
OUT = os.path.join(PRIV, "out")
API = os.path.join(PRIV, "api")
SUBMIT = os.path.join(HERE, "submission")
FILES = ["rents-input-flats-city-of-sydney.csv", "rents-past-leases-city-of-sydney.csv",
         "blocks-homes-and-people-2021-2026.csv"]
REQUIRED = ["pandas", "numpy", "scipy", "pyarrow", "requests"]
OPTIONAL = ["geopandas", "shapely", "pyproj", "rasterio"]  # public-data ablation / insights


def say(msg):
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fail(msg):
    print(f"\nSTOPPED: {msg}", flush=True)
    sys.exit(1)


def run(step, args):
    """Run one pipeline script with this interpreter, inside rents-heat/, streaming its output."""
    say(f"{step}: python {' '.join(args)}")
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    t0 = time.time()
    r = subprocess.run([sys.executable] + args, cwd=RH, env=env)
    if r.returncode != 0:
        fail(f"{step} failed (exit {r.returncode}). Fix the message above and rerun; finished steps can be skipped "
             f"with --skip-fetch.")
    print(f"   {step} done in {time.time() - t0:.0f} s", flush=True)


def check_python():
    say("0. checking Python and packages")
    if sys.version_info < (3, 11):
        fail(f"Python {sys.version.split()[0]} is too old; use 3.11+ (we used 3.13)")
    missing = [m for m in REQUIRED if importlib.util.find_spec(m) is None]
    if missing:
        fail(f"missing packages {missing}. Install: {sys.executable} -m pip install -r "
             f"{os.path.join(HERE, 'requirements-freeze.txt')}")
    opt = [m for m in OPTIONAL if importlib.util.find_spec(m) is None]
    print(f"   Python {sys.version.split()[0]} at {sys.executable}; required packages OK"
          + (f"; optional missing {opt} -> public-data ablation and insights will be skipped" if opt else "; geo packages OK"))
    if not os.path.isdir(RH):
        fail(f"{RH} not found: run this file from the repository root")
    return not opt


def place_files(data_dir, zip_path):
    say("1. official files")
    os.makedirs(REAL, exist_ok=True)
    if zip_path:
        with zipfile.ZipFile(zip_path) as z:
            names = {os.path.basename(n): n for n in z.namelist() if not n.endswith("/")}
            for f in FILES:
                if f in names:
                    with open(os.path.join(REAL, f), "wb") as fh:
                        fh.write(z.read(names[f]))
    if data_dir:
        for f in FILES:
            src = [os.path.join(data_dir, n) for n in os.listdir(data_dir) if n.endswith(f)]
            if src:
                shutil.copyfile(src[0], os.path.join(REAL, f))
    for f in FILES[:2]:
        p = os.path.join(REAL, f)
        if not os.path.exists(p):
            fail(f"{f} not found in {REAL}. Pass --zip <zip> or --data-dir <folder> (or copy it there).")
        with open(p, encoding="utf-8") as fh:
            n = sum(1 for _ in fh) - 1
        print(f"   {f}: {n:,} rows")


def place_key(key):
    say("2. API key")
    envp = os.path.join(PRIV, ".env")
    if key:
        with open(envp, "w", encoding="utf-8") as fh:
            fh.write(f"MICROBURBS_API_KEY={key.strip()}\n")
        print(f"   key written to {envp} (gitignored; not printed)")
    elif os.environ.get("MICROBURBS_API_KEY"):
        print("   key taken from the MICROBURBS_API_KEY environment variable")
    elif os.path.exists(envp) and "MICROBURBS_API_KEY=" in open(envp, encoding="utf-8").read():
        print(f"   key found in {envp}")
    else:
        return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--team", default="team-opal", help="lower case, hyphens: the CSV becomes rents_<team>.csv")
    ap.add_argument("--zip", help="the organiser zip with the three CSVs")
    ap.add_argument("--data-dir", help="folder holding the three CSVs (file names may carry a prefix)")
    ap.add_argument("--key", help="Microburbs team API key (written to rents-heat/data_private/.env)")
    ap.add_argument("--skip-fetch", action="store_true", help="do not call the API; reuse the pull on disk")
    ap.add_argument("--refetch", action="store_true", help="pull the API again even if a pull exists")
    ap.add_argument("--adverts", default="both", choices=["both", "past_leases", "api"],
                    help="training adverts: both (frozen choice: same MedAE, better share<=$100 on window A), "
                         "past_leases, or api")
    ap.add_argument("--no-ablation", action="store_true", help="skip the public-data ablation on window A")
    a = ap.parse_args()
    t0 = time.time()

    geo_ok = check_python()
    place_files(a.data_dir, a.zip)
    have_key = place_key(a.key)

    pulled = os.path.exists(os.path.join(API, "listings_rent_history.parquet"))
    if a.skip_fetch or (pulled and not a.refetch):
        say("3. API pull: " + ("skipped (--skip-fetch)" if a.skip_fetch else "already on disk (use --refetch to pull again)"))
        if not pulled and a.adverts != "past_leases":
            print("   no API pull on disk -> training on the past-leases file only")
            a.adverts = "past_leases"
    elif not have_key:
        say("3. API pull skipped: no key (pass --key). Training on the past-leases file only.")
        a.adverts = "past_leases"
    else:
        run("3. API pull (rent-history + properties, to_date 2026-07-31)", ["src/rents/api_fetch.py"])

    adv = ["--adverts", a.adverts]
    abl = [] if (a.no_ablation or not geo_ok) else ["--ablation"]
    run("4. window A (select + freeze)", ["run_all.py", "--data", "real", "--window", "A"] + adv + abl)
    run("5. window B (evaluate once)", ["run_all.py", "--data", "real", "--window", "B"] + adv)
    run("6. final (65,777 flats)", ["run_all.py", "--data", "real", "--window", "final", "--team", a.team] + adv)
    if geo_ok:
        run("7a. public-layer insights", ["src/rents/public_insights.py", "--data", "real", "--window", "B"] + adv)
    run("7b. report.html", ["src/rents/make_report.py", "--data", "real", "--team", a.team])

    say("8. submission folder")
    os.makedirs(SUBMIT, exist_ok=True)
    for f in (f"rents_{a.team}.csv", "report.html", "frozen_rule.json"):
        shutil.copyfile(os.path.join(OUT, f), os.path.join(SUBMIT, f))
        print(f"   {os.path.join(SUBMIT, f)}")
    fr = json.load(open(os.path.join(OUT, "frozen_rule.json"), encoding="utf-8"))
    mB = json.load(open(os.path.join(OUT, "metrics_B.json"), encoding="utf-8"))
    mF = json.load(open(os.path.join(OUT, "metrics_final.json"), encoding="utf-8"))
    print("\n================ RESULT ================")
    print(f"frozen rule (window A): method={fr['method']} k={fr.get('k')} stale_months={fr.get('stale_m')} "
          f"snap=${fr.get('snap')} public_adjustment={'applied' if (fr.get('public') or {}).get('apply') else 'not applied'}")
    for name, v in mB["variants"].items():
        m = v["all"]
        print(f"window B {name:<16} all {m['n']:>5} adverts  MedAE ${m['medae']}  within $100 {m['within100']:.1%}  R2 {m['r2']}")
        for sl in ("own_history", "no_history"):
            s = v[sl]
            print(f"         {'':<16} {sl:<12} {s['n']:>5} adverts  MedAE ${s['medae']}  within $100 {s['within100']:.1%}")
    print(f"final: {mF['n_rows']:,} rows, contract {mF['contract']}, adverts used {mF.get('adverts')}")
    print(f"total {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
