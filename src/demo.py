"""Live-demo helper: predict daily ozone for one day.

Minimum example:
  python src/demo.py --temperature 90 --pressure 1010 --wind 2.5 --no2 18 --month 7 \
                     --latitude 34.05 --longitude -118.24

Better (yesterday's ozone makes the forecast much more accurate):
  ... --ozone-yesterday 0.055

Units: temperature deg F, pressure millibar, wind knots, NO2 ppb, ozone in ppm (0.055 ppm = 55 ppb).
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=None, help="default: outputs_final/best_model.joblib (or outputs/...)")
ap.add_argument("--temperature", type=float, required=True, help="deg F (daily mean)")
ap.add_argument("--pressure", type=float, required=True, help="millibar")
ap.add_argument("--wind", type=float, required=True, help="knots (daily mean)")
ap.add_argument("--no2", type=float, required=True, help="ppb (daily mean)")
ap.add_argument("--month", type=int, required=True)
ap.add_argument("--day", type=int, default=15, help="day of month (only used for day-of-year)")
ap.add_argument("--latitude", type=float, required=True)
ap.add_argument("--longitude", type=float, required=True)
ap.add_argument("--voc", type=float, default=None, help="ppb (ignored if model trained without VOC)")
ap.add_argument("--temp-max", type=float, default=None, help="deg F, daily peak")
ap.add_argument("--wind-max", type=float, default=None)
ap.add_argument("--no2-max", type=float, default=None)
ap.add_argument("--ozone-yesterday", type=float, default=None, help="ppm, yesterday's daily mean ozone")
ap.add_argument("--ozone-max-yesterday", type=float, default=None, help="ppm, yesterday's daily peak ozone")
ap.add_argument("--ozone-7day", type=float, default=None, help="ppm, mean ozone over previous 7 days")
ap.add_argument("--temp-yesterday", type=float, default=None)
ap.add_argument("--no2-yesterday", type=float, default=None)
a = ap.parse_args()

# ----------------------------------------------------------------------------- load model
path = a.model
if path is None:
    path = next((p for p in ("outputs_final/best_model.joblib", "outputs/best_model.joblib") if os.path.exists(p)),
                "outputs_final/best_model.joblib")
b = joblib.load(path)
feats = b["features"]
d = b.get("defaults") or {}
old_file = "defaults" not in b or "site_table" not in b

# ----------------------------------------------------------------------------- build the feature row
doy = pd.Timestamp(2019, a.month, a.day).dayofyear
typed = {}   # feature -> value you explicitly gave (for the "INPUTS" print-out)


def given(name, val):
    if val is not None:
        typed[name] = val
    return val


temp_max = given("temp_max", a.temp_max) if a.temp_max is not None else a.temperature + d.get("temp_spread", 0.0)

if "site_table" in b:
    st = b["site_table"]
    near = ((st.latitude - a.latitude) ** 2 + (st.longitude - a.longitude) ** 2).idxmin()
    site_mean = float(st.site_mean_ozone[near])
else:
    site_mean = b.get("global_mean", 0.03)

oz_y = given("ozone_yesterday", a.ozone_yesterday)
given("temperature", a.temperature); given("pressure", a.pressure); given("wind", a.wind); given("no2", a.no2)
given("latitude", a.latitude); given("longitude", a.longitude)
given("wind_max", a.wind_max); given("no2_max", a.no2_max); given("voc", a.voc)
given("ozone_max_yesterday", a.ozone_max_yesterday); given("ozone_7day", a.ozone_7day)
given("temp_yesterday", a.temp_yesterday); given("no2_yesterday", a.no2_yesterday)

row = dict(
    temperature=a.temperature, temp_max=temp_max, temp_spread=temp_max - a.temperature, pressure=a.pressure,
    wind=a.wind, wind_max=a.wind_max if a.wind_max is not None else d.get("wind_max", a.wind),
    no2=a.no2, no2_max=a.no2_max if a.no2_max is not None else d.get("no2_max", a.no2),
    voc=a.voc if a.voc is not None else d.get("voc", 0.0), rh=d.get("rh", 0.0),
    doy_sin=np.sin(2 * np.pi * doy / 365.25), doy_cos=np.cos(2 * np.pi * doy / 365.25),
    latitude=a.latitude, longitude=a.longitude, site_mean_ozone=site_mean,
    ozone_lag1=oz_y if oz_y is not None else d.get("ozone_lag1"),
    ozone_max_lag1=a.ozone_max_yesterday if a.ozone_max_yesterday is not None else d.get("ozone_max_lag1"),
    ozone_roll7=(a.ozone_7day if a.ozone_7day is not None else (oz_y if oz_y is not None else d.get("ozone_roll7"))),
    temp_lag1=a.temp_yesterday if a.temp_yesterday is not None else a.temperature,
    temp_roll3=a.temperature,
    no2_lag1=a.no2_yesterday if a.no2_yesterday is not None else a.no2,
)
for f in feats:                                 # any feature this script doesn't know -> training median
    if f not in row or row[f] is None:
        row[f] = d.get(f)
missing = [f for f in feats if row.get(f) is None or pd.isna(row[f])]
if missing:
    raise SystemExit(f"Cannot predict: no value for {missing}. Provide them as options (see --help) "
                     "or re-train with the current train.py.")

X = b["scaler"].transform(pd.DataFrame([row])[feats])
pred = float(b["model"].predict(X)[0])
ppb = pred * 1000

# ----------------------------------------------------------------------------- output
LABEL = {"temperature": "temperature (F)", "temp_max": "temp_max (F)", "temp_spread": "temp_spread (F)",
         "pressure": "pressure (mbar)", "wind": "wind (knots)", "wind_max": "wind_max (knots)",
         "no2": "no2 (ppb)", "no2_max": "no2_max (ppb)", "voc": "voc (ppb)", "rh": "humidity (%)",
         "doy_sin": "day-of-year (sin)", "doy_cos": "day-of-year (cos)", "latitude": "latitude",
         "longitude": "longitude", "site_mean_ozone": "site_mean_ozone (ppb)",
         "ozone_lag1": "ozone_lag1 (ppb)", "ozone_max_lag1": "ozone_max_lag1 (ppb)",
         "ozone_roll7": "ozone_roll7 (ppb)", "temp_lag1": "temp_lag1 (F)", "temp_roll3": "temp_roll3 (F)",
         "no2_lag1": "no2_lag1 (ppb)"}
PPB_SCALE = {"site_mean_ozone", "ozone_lag1", "ozone_max_lag1", "ozone_roll7"}


def show(f):
    v = row[f] * 1000 if f in PPB_SCALE else row[f]
    return f"{LABEL.get(f, f)}: {v:.2f}"


print(f"Model used: {b['name']}  ({len(feats)} features)  [{path}]")
if old_file:
    print("WARNING: model file is from an old train.py; re-train for full feature support.")
print()

print("INPUTS YOU GAVE:")
print(f"  temperature={a.temperature}  pressure={a.pressure}  wind={a.wind}  no2={a.no2}  "
     f"month={a.month}  day={a.day}  lat={a.latitude}  lon={a.longitude}")
if oz_y is not None:
    print(f"  ozone_yesterday={oz_y} ppm ({oz_y*1000:.1f} ppb)")
extra = {k: v for k, v in typed.items() if k not in ("temperature", "pressure", "wind", "no2", "latitude",
                                                      "longitude", "ozone_yesterday")}
if extra:
    print("  " + "  ".join(f"{k}={v}" for k, v in extra.items()))

auto = [f for f in feats if f not in ("temperature", "pressure", "wind", "no2", "latitude", "longitude", "doy_sin", "doy_cos")]
print()
print("FEATURES THE MODEL ALSO USES (filled in automatically):")
for f in auto:
    print(f"  {show(f)}")

print()
print(f"TARGET:     daily mean ground-level ozone at this station")
print(f"PREDICTION: {ppb:.1f} ppb  ({pred:.4f} ppm)")