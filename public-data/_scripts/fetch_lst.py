"""Land surface temperature, Sydney summer 2022-23 (NSW DPHI, part of the 2022 HVI dataset, CC BY 4.0).
The full GeoTIFF is ~309 MB; we read only the bbox window with HTTP range requests (GDAL /vsicurl/) after
resolving the data.nsw redirect to the signed S3 URL, and save a clipped GeoTIFF.
Run with the rents-heat venv:  rents-heat/.venv/Scripts/python.exe public-data/_scripts/fetch_lst.py
Output: public-data/heat/lst_sydney_2022-2023_summer_bbox.tif
"""
import os
import sys
import time

import requests
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

sys.path.insert(0, os.path.dirname(__file__))
from common import BBOX, ROOT, UA  # noqa: E402

URL = ("https://datasets.seed.nsw.gov.au/dataset/08c39277-8b70-4c4d-905e-5c931c2eef08/resource/"
       "b081a8de-ca2e-4693-b961-706cd9f2c6ad/download/lst_sydney_2022-2023_summer.tif")
PAD = 0.01  # degrees of padding around the bbox

if __name__ == "__main__":
    t0 = time.time()
    r = requests.head(URL, headers={"User-Agent": UA}, allow_redirects=False, timeout=60)
    signed = r.headers.get("Location", URL)
    print("redirect status", r.status_code, "-> S3" if "amazonaws" in signed else "(no redirect)")
    os.environ["GDAL_HTTP_USERAGENT"] = UA
    os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
    os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif"
    with rasterio.open("/vsicurl/" + signed) as src:
        print("crs", src.crs, "size", src.width, src.height, "res", src.res, "dtype", src.dtypes, "nodata", src.nodata,
              "tiled", src.profile.get("tiled"), "blocks", src.block_shapes)
        b = transform_bounds("EPSG:4326", src.crs, BBOX["w"] - PAD, BBOX["s"] - PAD, BBOX["e"] + PAD, BBOX["n"] + PAD)
        win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        data = src.read(window=win)
        prof = src.profile.copy()
        prof.update(width=data.shape[2], height=data.shape[1], transform=src.window_transform(win), compress="deflate",
                    tiled=True, blockxsize=256, blockysize=256)
        prof.pop("photometric", None)
        out = ROOT / "heat" / "lst_sydney_2022-2023_summer_bbox.tif"
        out.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(data)
            dst.update_tags(source=URL, note="clipped bbox window via HTTP range reads", **src.tags())
    v = data[0][data[0] != (src.nodata if src.nodata is not None else -9999)]
    print("window", win, "shape", data.shape, "min/median/max", float(v.min()), float(sorted(v.ravel())[len(v)//2]), float(v.max()))
    print(f"saved {out} {out.stat().st_size/1e6:.2f} MB in {time.time()-t0:.0f}s")
