"""ABS Census 2021 G02 'Median rent ($/weekly)' for SA2s intersecting the bbox, via the ABS Data API (SDMX).
SA1-level G02 is NOT offered by the Data API (only SA2+, LGA, POA, SAL...); SA1 would need the General Community
Profile DataPack (large zip), so SA1 was skipped (documented in MANIFEST). SA2 codes come from the HVI 2022 SA1
layer already downloaded (heat/hvi2022_sa1_bbox.geojson, field SA2_CODE21) - the SA1 polygons serve as the
SA1->SA2 allocation for the bbox.
Outputs: public-data/abs/c21_g02_sa2_median_rent_bbox.csv, abs/c21_g02_poa_median_rent_bbox.csv,
         abs/sa1_2021_bbox_from_hvi.csv (SA1 code, SA2 code, centroid)
"""
import io
import json

import pandas as pd

from common import ROOT, S

API = "https://data.api.abs.gov.au/rest/data"
POAS = ["2000", "2007", "2008", "2009", "2010", "2011", "2015", "2016", "2017", "2018", "2021", "2037", "2042",
        "2043", "2050", "2006", "2019", "2020", "2033", "2060", "2061"]


def fetch(flow, regions, region_type):
    key = f"6.{'+'.join(regions)}.{region_type}.1"
    url = f"{API}/ABS,{flow},1.0.0/{key}"
    r = S.get(url, params={"format": "csvfilewithlabels"}, headers={"Accept": "application/vnd.sdmx.data+csv;labels=both"},
              timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df["_url"] = r.url
    return df


if __name__ == "__main__":
    fc = json.load(open(ROOT / "heat" / "hvi2022_sa1_bbox.geojson", encoding="utf-8"))
    rows = []
    for f in fc["features"]:
        p = f["properties"]
        g = f["geometry"]
        ring = g["coordinates"][0] if g["type"] == "Polygon" else g["coordinates"][0][0]
        xs, ys = zip(*[(c[0], c[1]) for c in ring])
        rows.append(dict(SA1_CODE21=p["SA1_2021_AUST_GDA2020_SA1_CODE2"], SA2_CODE21=str(p["SA2_CODE21"]), SA2_NAME21=p["SA2_NAME21"], lon=sum(xs) / len(xs),
                         lat=sum(ys) / len(ys)))
    sa1 = pd.DataFrame(rows)
    (ROOT / "abs").mkdir(exist_ok=True)
    sa1.to_csv(ROOT / "abs" / "sa1_2021_bbox_from_hvi.csv", index=False)
    sa2 = sorted(sa1.SA2_CODE21.unique())
    print(len(sa1), "SA1s,", len(sa2), "SA2s")
    d2 = fetch("C21_G02_SA2", sa2, "SA2")
    d2.to_csv(ROOT / "abs" / "c21_g02_sa2_median_rent_bbox.csv", index=False)
    print(d2.iloc[:, :12].head(40).to_string())
    dp = fetch("C21_G02_POA", POAS, "POA")
    dp.to_csv(ROOT / "abs" / "c21_g02_poa_median_rent_bbox.csv", index=False)
    print(dp.iloc[:, :12].to_string())
