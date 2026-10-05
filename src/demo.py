"""Live-demo helper: load the saved best model and predict ozone for one day.

Minimum example (works with any trained model):
  python src/demo.py --temperature 90 --pressure 1010 --wind 2.5 --no2 18 --month 7 \
                     --latitude 34.05 --longitude -118.24

Better example (if the model was trained with lag features, give yesterday's ozone in ppm):
  ... --ozone-yesterday 0.055 --temp-max 98

Anything you do not give is filled with the training-set median.
"""
import argparse

import joblib
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="outputs/best_model.joblib")
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
ap.add_argument("--temp-yesterday", type=float, default=None)
ap.add_argument("--no2-yesterday", type=float, default=None)
a = ap.parse_args()

b = joblib.load(a.model)
feats, d = b["features"], b["defaults"]
doy = pd.Timestamp(2019, a.month, a.day).dayofyear
temp_max = a.temp_max if a.temp_max is not None else a.temperature + d.get("temp_spread", 0.0)

# nearest training site (by lat/lon) supplies the per-site average ozone
st = b["site_table"]
nearest = ((st.latitude - a.latitude) ** 2 + (st.longitude - a.longitude) ** 2).idxmin()

row = dict(
    temperature=a.temperature, temp_max=temp_max, temp_spread=temp_max - a.temperature,
    pressure=a.pressure, wind=a.wind, wind_max=a.wind_max if a.wind_max is not None else d.get("wind_max", a.wind),
    no2=a.no2, no2_max=a.no2_max if a.no2_max is not None else d.get("no2_max", a.no2),
    voc=a.voc if a.voc is not None else d.get("voc", 0.0), rh=d.get("rh", 0.0),
    doy_sin=np.sin(2 * np.pi * doy / 365.25), doy_cos=np.cos(2 * np.pi * doy / 365.25),
    latitude=a.latitude, longitude=a.longitude, site_mean_ozone=float(st.site_mean_ozone[nearest]),
    ozone_lag1=a.ozone_yesterday if a.ozone_yesterday is not None else d.get("ozone_lag1"),
    ozone_max_lag1=a.ozone_max_yesterday if a.ozone_max_yesterday is not None else d.get("ozone_max_lag1"),
    temp_lag1=a.temp_yesterday if a.temp_yesterday is not None else a.temperature,
    temp_roll3=a.temperature,
    no2_lag1=a.no2_yesterday if a.no2_yesterday is not None else a.no2,
)
X = b["scaler"].transform(pd.DataFrame([row])[feats])
pred = float(b["model"].predict(X)[0])
print(f"Model: {b['name']}")
if "ozone_lag1" in feats and a.ozone_yesterday is None:
    print("Note: --ozone-yesterday not given, using the training median (prediction will be less accurate).")
print(f"Predicted daily mean ozone: {pred:.4f} ppm ({pred*1000:.1f} ppb)")
print("Above 2015 EPA 8-hr standard (0.070 ppm)?", "YES" if pred > 0.070 else "no",
      "(note: the standard applies to the 8-hr max, this model predicts the daily mean)")