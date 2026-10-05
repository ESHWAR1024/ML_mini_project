"""Live-demo helper: load the saved best model and predict ozone for one day.

Example:
  python src/demo.py --temperature 90 --pressure 1010 --wind 2.5 --no2 18 --voc 90 \
                     --month 7 --latitude 34.05 --longitude -118.24
"""
import argparse

import joblib
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="outputs/best_model.joblib")
ap.add_argument("--temperature", type=float, required=True, help="deg F (EPA reports temperature in Fahrenheit)")
ap.add_argument("--pressure", type=float, required=True, help="millibar")
ap.add_argument("--wind", type=float, required=True, help="knots")
ap.add_argument("--no2", type=float, required=True, help="ppb")
ap.add_argument("--voc", type=float, default=0.0, help="ppb (ignored if the model was trained without VOC)")
ap.add_argument("--month", type=int, required=True)
ap.add_argument("--latitude", type=float, required=True)
ap.add_argument("--longitude", type=float, required=True)
a = ap.parse_args()

bundle = joblib.load(a.model)
row = dict(temperature=a.temperature, pressure=a.pressure, wind=a.wind, no2=a.no2, voc=a.voc,
           month_sin=np.sin(2 * np.pi * a.month / 12), month_cos=np.cos(2 * np.pi * a.month / 12),
           latitude=a.latitude, longitude=a.longitude)
X = bundle["scaler"].transform(pd.DataFrame([row])[bundle["features"]])
pred = float(bundle["model"].predict(X)[0])
print(f"Model: {bundle['name']}")
print(f"Predicted daily ozone: {pred:.4f} ppm ({pred*1000:.1f} ppb)")
print("Above 2015 EPA 8-hr standard (0.070 ppm)?", "YES" if pred > 0.070 else "no")