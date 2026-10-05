
"""Build data/ozone_daily.csv from EPA AirData DAILY summary downloads (no BigQuery needed).

Download from https://aqs.epa.gov/aqsweb/airdata/download_files.html  (section "Daily Summary Data")
one zip per year for each of these, and put them all in ONE folder (e.g. data/raw), zipped or unzipped:

    daily_44201_YYYY   ozone                (required, the target)
    daily_42602_YYYY   NO2                  (required)
    daily_TEMP_YYYY    temperature          (required)
    daily_PRESS_YYYY   barometric pressure  (required)
    daily_WIND_YYYY    wind                 (required)
    daily_VOCS_YYYY    VOCs                 (optional; few sites measure it - see --no-voc)

The exact file names on the EPA page may differ slightly; matching here is by the part of the name
shown above (case-insensitive), so a slightly different suffix still works.

Usage:
    python src/data_epa_csv.py --raw data/raw --out data/ozone_daily.csv
    python src/data_epa_csv.py --raw data/raw --no-voc        # skip VOC to keep many more rows
"""
import argparse
import glob
import os
import re

import pandas as pd

KEY = ["state_code", "county_code", "site_num", "date_local"]

# group name -> (file-name pattern, output column, optional filter on parameter_name)
GROUPS = {
    "ozone": ("daily_44201_", "ozone", None),
    "no2": ("daily_42602_", "no2", None),
    "temp": ("daily_temp_", "temperature", "outdoor temperature"),
    "press": ("daily_press_", "pressure", "barometric"),
    "wind": ("daily_wind_", "wind", "wind speed"),
    "voc": ("daily_vocs_", "voc", None),
}


def norm(c):
    return re.sub(r"[^a-z0-9]+", "_", c.strip().lower()).strip("_")


def read_group(raw_dir, prefix, out_col, name_filter, ozone_standard=None):
    files = [f for f in glob.glob(os.path.join(raw_dir, "**", "*"), recursive=True)
             if os.path.basename(f).lower().startswith(prefix)
             and f.lower().endswith((".csv", ".zip"))]
    if not files:
        return None
    parts = []
    for f in sorted(files):
        df = pd.read_csv(f, low_memory=False)
        df.columns = [norm(c) for c in df.columns]
        need = KEY + ["arithmetic_mean"]
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise SystemExit(f"{os.path.basename(f)} lacks columns {missing}. Columns found: {list(df.columns)[:15]}...")
        if name_filter and "parameter_name" in df.columns:
            df = df[df["parameter_name"].str.lower().str.contains(name_filter, na=False)]
        if ozone_standard and "pollutant_standard" in df.columns:
            df = df[df["pollutant_standard"].str.lower() == ozone_standard.lower()]
        cols = KEY + ["arithmetic_mean"] + [c for c in ("latitude", "longitude") if c in df.columns]
        parts.append(df[cols])
    d = pd.concat(parts, ignore_index=True)
    agg = {"arithmetic_mean": "mean"}
    for c in ("latitude", "longitude"):
        if c in d.columns:
            agg[c] = "first"
    d = d.groupby(KEY, as_index=False).agg(agg).rename(columns={"arithmetic_mean": out_col})
    print(f"  {out_col:12s}: {len(files)} file(s), {len(d):,} site-days")
    return d


def build(raw_dir, out, no_voc=False, ozone_standard=None):
    result = None
    for name, (prefix, col, filt) in GROUPS.items():
        if name == "voc" and no_voc:
            continue
        d = read_group(raw_dir, prefix, col, filt, ozone_standard if name == "ozone" else None)
        if d is None:
            if name == "voc":
                print("  voc         : no files found -> continuing WITHOUT voc")
                continue
            raise SystemExit(f"No files found for '{name}' (looked for names starting with '{prefix}') in {raw_dir}")
        if name == "ozone":
            result = d
        else:
            d = d.drop(columns=[c for c in ("latitude", "longitude") if c in d.columns])
            result = result.merge(d, on=KEY, how="inner")
        print(f"  after joining {name}: {len(result):,} rows")
    result = result.sort_values(KEY)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    result.to_csv(out, index=False)
    print(f"Saved {len(result):,} rows -> {out}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw", help="folder with the downloaded daily_*.zip/.csv files")
    ap.add_argument("--out", default="data/ozone_daily.csv")
    ap.add_argument("--no-voc", action="store_true", help="skip VOC (recommended if row count is small)")
    ap.add_argument("--ozone-standard", default=None,
                    help="optional, keep only this Pollutant Standard, e.g. 'Ozone 8-hour 2008'")
    a = ap.parse_args()
    build(a.raw, a.out, a.no_voc, a.ozone_standard)