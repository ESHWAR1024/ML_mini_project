"""Build data/ozone_annual.csv from EPA 'annual_conc_by_monitor_YYYY' files.

Each of those files has ONE ROW PER MONITOR PER PARAMETER for one year.  We pick the parameters we need,
average duplicates (several instruments / methods per site), and join them on (site, year):

    ozone        parameter 44201  -> TARGET  (4th highest daily-max 8-hr value, or the annual mean)
    NO2          parameter 42602
    temperature  parameter 62101  (mean, and 90th percentile = how hot the hot days are)
    pressure     parameter 64101
    wind speed   parameters 61101 / 61103

Put one file per year (zipped or unzipped .csv) in a folder, e.g. data/raw_annual, and run:
    python src/data_epa_annual.py --raw data/raw_annual --out data/ozone_annual.csv

You need SEVERAL years (the model is trained on earlier years and tested on later ones).
"""
import argparse
import glob
import os

import pandas as pd

WANT = ["State Code", "County Code", "Site Num", "Parameter Code", "Parameter Name", "Latitude",
        "Longitude", "Pollutant Standard", "Year", "Event Type", "Arithmetic Mean",
        "4th Max Value", "90th Percentile"]
KEY = ["state_code", "county_code", "site_num", "year"]
CODES = {"ozone": [44201], "no2": [42602], "temp": [62101], "press": [64101], "wind": [61101, 61103]}


def read_all(raw_dir):
    files = [f for f in glob.glob(os.path.join(raw_dir, "**", "*"), recursive=True)
             if f.lower().endswith((".csv", ".zip"))]
    if not files:
        raise SystemExit(f"No .csv/.zip files found in {raw_dir}")
    parts = []
    for f in sorted(files):
        try:
            d = pd.read_csv(f, usecols=lambda c: c.strip() in WANT, low_memory=False)
        except Exception as e:                       # not a readable csv
            print(f"  skipped {os.path.basename(f)} ({e})")
            continue
        if not {"Parameter Code", "Year", "Arithmetic Mean"} <= set(d.columns):
            print(f"  skipped {os.path.basename(f)} (not an annual_conc_by_monitor file)")
            continue
        print(f"  read {os.path.basename(f)}: {len(d):,} rows, years {sorted(d['Year'].unique())}")
        parts.append(d)
    if not parts:
        raise SystemExit("None of the files looked like annual_conc_by_monitor files.")
    d = pd.concat(parts, ignore_index=True)
    d = d.rename(columns={"State Code": "state_code", "County Code": "county_code", "Site Num": "site_num",
                          "Year": "year", "Parameter Code": "code"})
    if "Event Type" in d.columns:                    # drop the duplicate 'Events Excluded' rows
        d = d[~d["Event Type"].astype(str).str.lower().str.contains("excluded")]
    return d


def pick(d, codes, value_cols, out_names):
    s = d[d["code"].isin(codes)]
    return s, s.groupby(KEY, as_index=False)[value_cols].mean().rename(columns=dict(zip(value_cols, out_names)))


def build(raw_dir, out, ozone_standard, target):
    d = read_all(raw_dir)
    print(f"Total rows read: {len(d):,}")

    # ---- ozone (target) ----
    oz = d[d["code"].isin(CODES["ozone"])]
    if "Pollutant Standard" in oz.columns:
        print("Ozone pollutant standards present:", sorted(oz["Pollutant Standard"].dropna().unique()))
        oz = oz[oz["Pollutant Standard"].astype(str).str.lower() == ozone_standard.lower()]
    if oz.empty:
        raise SystemExit(f"No ozone rows for standard '{ozone_standard}'. Pick one of the names printed above "
                         f"with --ozone-standard.")
    tcol = "4th Max Value" if target == "4th_max" else "Arithmetic Mean"
    oz = oz.dropna(subset=[tcol])
    ozone = oz.groupby(KEY, as_index=False).agg(ozone=(tcol, "mean"), latitude=("Latitude", "first"),
                                                 longitude=("Longitude", "first"))
    print(f"  ozone       : {len(ozone):,} monitor-years  (target = {tcol})")

    # ---- predictors ----
    _, no2 = pick(d, CODES["no2"], ["Arithmetic Mean"], ["no2"])
    _, temp_m = pick(d, CODES["temp"], ["Arithmetic Mean"], ["temperature"])
    t = d[d["code"].isin(CODES["temp"])]
    temp_p = t.groupby(KEY, as_index=False)["90th Percentile"].mean().rename(columns={"90th Percentile": "temp_p90"}) \
        if "90th Percentile" in t.columns else None
    _, press = pick(d, CODES["press"], ["Arithmetic Mean"], ["pressure"])
    _, wind = pick(d, CODES["wind"], ["Arithmetic Mean"], ["wind"])
    for name, tab in [("no2", no2), ("temperature", temp_m), ("pressure", press), ("wind", wind)]:
        print(f"  {name:12s}: {len(tab):,} monitor-years")
        if tab.empty:
            names = d[d['Parameter Name'].astype(str).str.contains(
                {"no2": "nitrogen", "temperature": "temp", "pressure": "pressure|baro", "wind": "wind"}[name],
                case=False, na=False)][["code", "Parameter Name"]].drop_duplicates().head(10)
            raise SystemExit(f"No rows for {name} (parameter codes {CODES.get(name)}). Similar parameters found:\n{names}")

    res = ozone
    for name, tab in [("no2", no2), ("temperature", temp_m), ("temp_p90", temp_p), ("pressure", press), ("wind", wind)]:
        if tab is None:
            continue
        res = res.merge(tab, on=KEY, how="inner")
        print(f"  after joining {name:12s}: {len(res):,} rows")
    res = res.sort_values(KEY)
    print("Rows per year:\n", res.groupby("year").size().to_string())
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    res.to_csv(out, index=False)
    print(f"Saved {len(res):,} rows -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw_annual")
    ap.add_argument("--out", default="data/ozone_annual.csv")
    ap.add_argument("--ozone-standard", default="Ozone 8-hour 2008")
    ap.add_argument("--target", choices=["4th_max", "mean"], default="4th_max",
                    help="4th_max = 4th highest daily-max 8hr value (what EPA uses); mean = annual mean")
    a = ap.parse_args()
    build(a.raw, a.out, a.ozone_standard, a.target)