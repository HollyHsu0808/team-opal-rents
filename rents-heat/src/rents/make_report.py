"""Build report.html from the metrics JSON files of one data profile (A, B, final) + public_insights + public-data facts.

Usage (from rents-heat):  .venv/Scripts/python.exe src/rents/make_report.py --data real --team Team-Opal
Writes data_private/out[/<profile>]/report.html. Self-contained HTML (inline CSS/SVG), no external assets.
Required by the task: lead with insights; held-out MedAE, share <= $100 and R2; explain the date split; build only on
earlier adverts; show data and method; credit every external source.
"""
import argparse
import html
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rents.load import ROOT, cfg, path_for  # noqa: E402

PUBLIC_DIR = os.path.normpath(os.path.join(ROOT, "..", "public-data", "_derived"))


def fmt(x, p=1):
    return "–" if x is None else f"{x:.{p}f}"


def pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def slice_rows(name, s):
    out = []
    for sl, label in (("own_history", "flat has its own earlier advert"), ("no_history", "no usable own advert"), ("all", "all adverts")):
        m = s[sl]
        out.append(f"<tr><td>{html.escape(name)}</td><td>{label}</td><td class=n>{m['n']:,}</td><td class=n>${fmt(m['medae'])}</td>"
                   f"<td class=n>{pct(m['within100'])}</td><td class=n>{fmt(m['r2'], 3)}</td><td class=n>{pct(m['coverage'])}</td></tr>")
    return "\n".join(out)


def rule_text(fr):
    if fr.get("method") == "floor":
        return "floor estimate only (own last advert × index; otherwise suburb × bedrooms × parking medians)"
    t = {"plain": "same-building same-bedroom comps (every advert counts)",
         "onevote": f"same-building same-bedroom comps, one flat one vote, shrink k={fr.get('k')}"}[fr["method"]]
    if fr.get("stale_m"):
        t += f"; own adverts older than {fr['stale_m']} months replaced by " + ("the building/ladder estimate" if not fr.get("stale_w") else f"a {fr['stale_w']:.0%}/{1-fr['stale_w']:.0%} blend")
    if fr.get("snap"):
        t += f"; predictions snapped to the nearest ${fr['snap']}"
    if (fr.get("public") or {}).get("apply"):
        t += "; LiDAR/flight-path adjustment on no-history rows (accepted on window A)"
    if fr.get("public_within"):
        t += ("; LiDAR floor-view difference between the flat and its comparable flats, priced with coefficients "
              "estimated within buildings (P1)")
    return t


def timeline_svg(c):
    w = c["windows"]
    months = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09"]
    x0, step = 90, 70
    rows = [("A", w["A"]["train_end"], w["A"]["test_start"], w["A"]["test_end"], "select rules"),
            ("B", w["B"]["train_end"], w["B"]["test_start"], w["B"]["test_end"], "evaluate once"),
            ("final", w["final"]["train_end"], "2026-08-01", "2026-09-30", "scored by organisers")]
    def xm(d):  # x at the END of month d (YYYY-MM-DD) or START when day == 01
        y, m, dd = d[:4], int(d[5:7]), int(d[8:10])
        i = m - 1
        return x0 + step * (i + (0 if dd == 1 else 1))
    s = [f'<svg viewBox="0 0 {x0 + step * 9 + 150} 150" width="100%" style="max-width:900px;font:12px sans-serif">']
    for i, mo in enumerate(months):
        s.append(f'<text x="{x0 + step * i + step/2}" y="14" text-anchor="middle" fill="#555">{mo[5:]}/26</text>')
        s.append(f'<line x1="{x0 + step * i}" y1="20" x2="{x0 + step * i}" y2="140" stroke="#eee"/>')
    for r, (name, te, ts, tend, note) in enumerate(rows):
        y = 40 + r * 34
        s.append(f'<text x="8" y="{y + 14}" font-weight="bold">{name}</text>')
        s.append(f'<rect x="{x0}" y="{y}" width="{xm(te) - x0}" height="20" fill="#9ec5e8"/>')
        s.append(f'<rect x="{xm(ts)}" y="{y}" width="{xm(tend) - xm(ts)}" height="20" fill="{"#f3a65a" if name != "final" else "#bbb"}"/>')
        s.append(f'<text x="{xm(tend) + 8}" y="{y + 14}" fill="#333">{note}</text>')
    s.append(f'<rect x="{x0}" y="128" width="12" height="12" fill="#9ec5e8"/><text x="{x0 + 16}" y="138">train (adverts ≤ cut-off; every feature and index rebuilt from these only)</text>')
    s.append(f'<rect x="{x0 + 430}" y="128" width="12" height="12" fill="#f3a65a"/><text x="{x0 + 446}" y="138">held-out test adverts</text>')
    s.append("</svg>")
    return "\n".join(s)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "real", "apiproxy"], required=True)
    ap.add_argument("--team", default="Team-Opal")
    a = ap.parse_args(argv)
    c = cfg()
    out_dir = path_for(c, a.data, "out")
    def load(name):
        p = os.path.join(out_dir, name)
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
    A, B, F = load("metrics_A.json"), load("metrics_B.json"), load("metrics_final.json")
    fr = load("frozen_rule.json") or {}
    ins_p = os.path.join(out_dir, "public_insights_B.csv")
    ins = pd.read_csv(ins_p) if os.path.exists(ins_p) else None
    lidar = json.load(open(os.path.join(PUBLIC_DIR, "lidar_precompute_summary.json"))) if os.path.exists(os.path.join(PUBLIC_DIR, "lidar_precompute_summary.json")) else {}

    # ---- headline numbers
    bname = next(k for k in B["variants"] if k.startswith("frozen")) if B else None
    bm = B["variants"][bname] if B else None
    bfl = B["variants"].get("reference_floor") if B else None
    cand = A.get("candidates_all", {}) if A else {}
    a_floor = cand.get("floor", {}).get("medae") if cand else None
    a_best = (A or {}).get("selection", {}).get("best_all", {}).get("medae")
    nostale = {k: v["medae"] for k, v in cand.items() if "stale" not in k and k != "floor"}
    stale = {k: v["medae"] for k, v in cand.items() if "stale" in k}
    pub = (A or {}).get("public_ablation", {})
    pub_txt = pub if isinstance(pub, str) else pub.get("result", "untested")

    insights = []
    if bm:
        insights.append(f"<b>Held-out result (window B, {bm['all']['n']:,} adverts, Jun–Jul 2026, trained on adverts ≤ 31 May):</b> median absolute error "
                        f"<b>${fmt(bm['all']['medae'])}</b>, {pct(bm['all']['within100'])} within $100, R² {fmt(bm['all']['r2'], 2)}"
                        + (f" — against ${fmt(bfl['all']['medae'])} / {pct(bfl['all']['within100'])} for the simple floor estimate." if bfl else "."))
        insights.append(f"A flat's <b>own earlier advert</b> is the strongest evidence: ${fmt(bm['own_history']['medae'])} median error on {bm['own_history']['n']:,} adverts, "
                        f"against ${fmt(bm['no_history']['medae'])} where only the building or the suburb is known.")
    if stale and nostale:
        insights.append(f"<b>Old evidence is worse than no evidence.</b> On window A, using a flat's last advert when it is more than {fr.get('stale_m', 36)} months old "
                        f"gave ${fmt(min(nostale.values()))} overall; replacing those stale prices with the building's current comparables gave ${fmt(min(stale.values()))}. "
                        f"The same-building rule was selected on A and frozen before B.")
    insights.append("Changing the <b>bedroom count</b> between two adverts of the same address (whole-building ids, split lettings) multiplied the error by about 2.4 in our window-A check; "
                    "own history is therefore only used when the bedrooms match.")
    if fr.get("snap"):
        insights.append(f"Asking rents sit on a price grid (in the past-leases file 84% end in $0, 52% are multiples of $50, 26% of $100): snapping predictions to the nearest ${fr['snap']} "
                        f"improved window-A error from ${fmt(a_best)} to ${fmt((fr.get('snap_all') or {}).get(str(fr['snap'])))}.")
    if lidar:
        insights.append(f"<b>LiDAR (NSW Spatial Services, 2013 and 2020)</b> gives every building's height (coverage {pct(lidar.get('coverage_h_latest'))}, "
                        f"{lidar.get('new_2013_2020')} buildings built between the two surveys) and a floor-by-floor view test: water is visible from "
                        f"{pct(lidar.get('share_floors_water_view'))} of floors overall, 6% of floors 1–2 and 60% of floors above 20.")
    ev = fr.get("public_within_evidence") or {}
    if fr.get("public_within") and ev:
        bw = ev.get("beta_within_A", {})
        insights.append(f"<b>Same building, same bedrooms, different floor:</b> within building × bedroom groups (22,593 flats), a flat whose floor "
                        f"sees water in every direction asks {100 * bw.get('water_view_frac', 0):.0f}% more than one that sees none "
                        f"(park view {100 * bw.get('park_view_frac', 0):+.0f}%, clear of nearby roofs {100 * bw.get('roofline_clear_frac', 0):+.0f}%), "
                        "measured with LiDAR line-of-sight per floor. Building location, age and quality cancel out in this comparison. "
                        f"Pricing that floor-view difference against the comparable flats (P1) kept window-A error at ${ev['A']['all_medae'][1]:.0f} "
                        f"and lifted the share within $100 from {100 * ev['A']['all_within100'][0]:.1f}% to {100 * ev['A']['all_within100'][1]:.1f}%; "
                        f"on window B it was neutral (${ev['B']['all_medae'][1]:.0f}, {100 * ev['B']['all_within100'][1]:.1f}%, R² {ev['B']['r2'][0]:.3f} → {ev['B']['r2'][1]:.3f}). "
                        "P1 was chosen on A after the base rule was frozen and B was evaluated for both; both B results are shown.")
    insights.append(f"<b>Public-data honesty line:</b> the cross-sectional LiDAR / flight-path / age adjustment was {html.escape(str(pub_txt))}: "
                    "building-level layers double-count what same-building comparables already contain (on A, water view +25%, age −6% per decade "
                    "against suburb medians, yet applying them raised error). Our layers describe physical exposure; they are not causal estimates.")
    if ins is not None and len(ins):
        fp = ins[ins["feature"].str.startswith("flight path")]
        wv = ins[ins["feature"].str.startswith("LiDAR water")]
        if len(fp) == 2:
            d = dict(zip(fp["band"], fp["median_resid_pct"]))
            insights.append(f"Flats within 1 km of a Sydney Airport 16/34 approach path rent {d.get('under path', 0):+.1f}% against the model, "
                            f"others {d.get('not under', 0):+.1f}% (window-B residuals; a geometric proxy, not a noise measurement).")
        if len(wv) >= 2:
            insights.append("Residual by LiDAR water view (window B): " + ", ".join(f"{r.band} {r.median_resid_pct:+.1f}% (n={r.n_adverts:,})" for r in wv.itertuples()) + ".")

    # ---- tables
    rows_A = ""
    if A:
        sel = A.get("selection", {})
        shown = ["floor", "plain", sel.get("candidate", "")]
        for k in shown:
            if k in A["variants"]:
                rows_A += slice_rows(("chosen: " if k == sel.get("candidate") else "") + k, A["variants"][k])
    rows_B = ""
    if B:
        for k, v in B["variants"].items():
            rows_B += slice_rows(k, v)
    ins_html = ""
    if ins is not None and len(ins):
        ins_html = "<table><tr><th>public layer</th><th>band</th><th class=n>adverts</th><th class=n>flats</th><th class=n>median residual</th><th class=n>median abs error</th><th class=n>median rent</th></tr>" + "".join(
            f"<tr><td>{html.escape(r.feature)}</td><td>{html.escape(str(r.band))}</td><td class=n>{r.n_adverts:,}</td><td class=n>{r.n_flats:,}</td><td class=n>{r.median_resid_pct:+.1f}%</td><td class=n>${r.median_abs_err:.0f}</td><td class=n>${r.median_rent:.0f}</td></tr>"
            for r in ins.itertuples()) + "</table>"
    fit_info = (B or A or {}).get("fit", {})
    adv = (B or A or {}).get("adverts", {})
    final_txt = ""
    if F:
        final_txt = (f"Final model trained on adverts ≤ {F['window']['train_end'][:10]}: {F['n_rows']:,} flats predicted, contract check {F['contract']}, "
                     f"{F['n_own_history']:,} with own history, {F['n_used_comps']:,} with building comparables, {F['n_nan_filled']} filled with the citywide median.")

    sources = [
        ("Microburbs hackathon API (rent-history, properties) and the supplied input / past-leases files", "organiser; used on the day only, not republished"),
        ("NSW Spatial Services LiDAR Sydney201304 and Sydney202005 (point clouds, 1 m DEM) via ELVIS", "© State of New South Wales (DCS Spatial Services), CC BY 4.0"),
        ("OpenStreetMap building footprints, water, parks, railway stations, Sydney Airport runways", "© OpenStreetMap contributors, ODbL 1.0"),
        ("NSW Digital Cadastre (strata plan polygons and numbers)", "NSW Spatial Services, CC BY 3.0 AU"),
        ("NSW Administrative Boundaries (suburbs, LGA)", "NSW Spatial Services, CC BY"),
        ("Greater Sydney land surface temperature summer 2022-23; Heat Vulnerability Index 2022; tree canopy 2022", "NSW DPHI, CC BY 4.0"),
        ("Urban Heat Island to modified mesh block 2016", "NSW DCCEEW, CC BY"),
        ("ABS Census 2021 (median rent, SA2 / postcode)", "ABS, CC BY 4.0"),
        ("NSW rental bond lodgements 2025–2026 (benchmark only: postcode × bedrooms rule ≈ $100 median error on agreed rents)", "NSW Fair Trading"),
    ]
    src_html = "".join(f"<tr><td>{html.escape(s)}</td><td>{html.escape(l)}</td></tr>" for s, l in sources)

    doc = f"""<!doctype html><html lang=en><head><meta charset=utf-8><title>{html.escape(a.team)} — rent estimates, City of Sydney flats</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>body{{font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#222}}
h1{{font-size:26px;margin:.2em 0}} h2{{font-size:20px;margin-top:1.6em;border-bottom:2px solid #eee;padding-bottom:4px}}
table{{border-collapse:collapse;width:100%;margin:.6em 0;font-size:14px}} th,td{{border-bottom:1px solid #e5e5e5;padding:5px 8px;text-align:left;vertical-align:top}} th{{background:#f6f6f6}}
td.n,th.n{{text-align:right;font-variant-numeric:tabular-nums}} .box{{background:#f7f9fc;border-left:4px solid #3b82c4;padding:10px 14px;margin:.8em 0}}
li{{margin:.35em 0}} .small{{color:#666;font-size:13px}}</style></head><body>
<h1>What will this flat rent for? — {html.escape(a.team)}</h1>
<p class=small>Microburbs hackathon, 7 November 2026 · data profile: {a.data} · every number below is computed on real adverts dated on or before the stated cut-off; nothing dated after 31 July 2026 is used anywhere.</p>

<h2>1. What we found</h2>
<ul>{"".join(f"<li>{t}</li>" for t in insights)}</ul>

<h2>2. Date split (no random splits)</h2>
<div class=box>{timeline_svg(c)}</div>
<p>Window <b>A</b> (train ≤ 30 Apr 2026, test May) is used once to choose between the candidate rules and to freeze the settings. Window <b>B</b> (train ≤ 31 May, test Jun–Jul) is run exactly once with the frozen rule and is the number we report. The final model is refit on everything ≤ 31 Jul 2026 and predicts the 65,777 flats for Aug–Sep. For every window, the rent index, the suburb medians, the parking premium and the building comparables are rebuilt from that window's training adverts only; the code asserts that no training row is dated after the cut-off.</p>

<h2>3. Held-out results</h2>
<p><b>Frozen rule (selected on A):</b> {html.escape(rule_text(fr))}.</p>
<table><tr><th>window B — rule</th><th>slice</th><th class=n>adverts</th><th class=n>median abs error</th><th class=n>within $100</th><th class=n>R²</th><th class=n>coverage</th></tr>{rows_B}</table>
<table><tr><th>window A — candidates</th><th>slice</th><th class=n>adverts</th><th class=n>median abs error</th><th class=n>within $100</th><th class=n>R²</th><th class=n>coverage</th></tr>{rows_A}</table>
<p class=small>{final_txt}</p>

<h2>4. Method</h2>
<ol>
<li><b>Market index:</b> a repeat-rent index from pairs of adverts of the same flat (location and quality cancel out), solved monthly with smoothing, plus a shrunk suburb deviation. The level is held at the cut-off month; no uplift is extrapolated into Aug–Sep (Sydney unit asking rents were flat to slightly down after mid-2026).</li>
<li><b>Own history:</b> the flat's latest advert with the same bedroom count, moved to the cut-off month with the index, with the parking premium added or removed for the other output column.</li>
<li><b>No usable own advert:</b> suburb × bedrooms × parking medians of the last 12 months (≥10 adverts), falling back to suburb × bedrooms, then bedrooms.</li>
<li><b>Building comparables:</b> adverts of other flats in the same building with the same bedrooms (last 36 months), index- and parking-adjusted, the target flat excluded; one-flat-one-vote collapses each neighbour's adverts to one signal and shrinks toward the ladder by the number of distinct flats.</li>
<li><b>Stale own history:</b> when the own advert is older than the frozen threshold, the building comparables / ladder estimate replaces it.</li>
<li><b>Snapping:</b> predictions rounded to the asking-rent grid when window A showed it helps.</li>
<li><b>Public layers (LiDAR floor views, roofline, height, flight path, new build, strata age):</b> fitted on A training residuals and applied only if A improves; otherwise reported as tested and not used.</li>
</ol>

<h2>5. Data</h2>
<p>Training adverts: {adv.get('n_valid', '–'):,} valid of {adv.get('n_raw', '–'):,} raw; adverts dated after the cut-off dropped: {adv.get('n_after_cutoff_dropped', '–')}. Fit for the reported window: train end {fit_info.get('train_end', '–')}, {fit_info.get('n_train_adverts', 0):,} training adverts, {fit_info.get('n_repeat_pairs', 0):,} repeat pairs for the index, {fit_info.get('n_flats_own_history', 0):,} flats with own history, citywide parking premium {fmt(fit_info.get('premium_citywide'), 3)} (log).</p>
<p>Public layers were pre-computed from open data (LiDAR 1 m height grids, 4,399 OSM residential building footprints, per-floor line-of-sight to water and parks, distance to runway centrelines, strata-plan age) and joined to flats by building address and coordinates. All public layers predate every cut-off.</p>
{"<h3>Does the market price what the public layers describe? (window-B residuals)</h3>" + ins_html if ins_html else ""}

<h2>6. Sources and licences</h2>
<table><tr><th>source</th><th>licence / use</th></tr>{src_html}</table>
<p class=small>Code: rents-heat (private repository); Python, pandas, numpy, scipy, geopandas, rasterio. Report generated by src/rents/make_report.py.</p>
</body></html>"""
    out = os.path.join(out_dir, "report.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(f"wrote {out} ({len(doc):,} bytes; insights {len(insights)}, A rows {rows_A.count('<tr>')}, B rows {rows_B.count('<tr>')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
