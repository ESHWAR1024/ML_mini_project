"""Live-demo helper: predict daily ozone for one day and explain the result in readable boxes.

Minimum example:
  python src/demo.py --temperature 90 --pressure 1010 --wind 2.5 --no2 18 --month 7 \
                     --latitude 34.05 --longitude -118.24

Better example (yesterday's ozone makes the forecast much more accurate):
  ... --ozone-yesterday 0.055

Anything you do not give is filled automatically (the output box says exactly how).
Units: temperature deg F, pressure millibar, wind knots, NO2 ppb, ozone in ppm (0.055 ppm = 55 ppb).
"""
import argparse
import json
import os
import sys
import textwrap

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
ap.add_argument("--ascii", action="store_true", help="force plain ASCII boxes")
a = ap.parse_args()

# ----------------------------------------------------------------------------- pretty printing
def _unicode_ok():
    try:
        "─│┌┐└┘├┤█".encode(sys.stdout.encoding or "ascii")
        return True
    except Exception:
        return False


UNI = _unicode_ok() and not a.ascii
H, V, TL, TR, BL, BR, ML, MR = ("─", "│", "┌", "┐", "└", "┘", "├", "┤") if UNI else ("-", "|", "+", "+", "+", "+", "+", "+")
W = 84  # total box width


def box(title, lines):
    inner = W - 2
    print(TL + H * 2 + f" {title} " + H * (inner - len(title) - 4) + TR)
    for ln in lines:
        if ln == "---":
            print(ML + H * inner + MR)
        else:
            for part in (textwrap.wrap(ln, inner - 2, subsequent_indent="    ") or [""]):
                print(V + (" " + part).ljust(inner) + V)
    print(BL + H * inner + BR)


def table(rows, cols):
    """rows: list of tuples of strings; cols: column widths. Returns list of lines."""
    return ["  ".join(str(c)[:w].ljust(w) for c, w in zip(r, cols)) for r in rows]


# ----------------------------------------------------------------------------- load model
path = a.model
if path is None:
    path = next((p for p in ("outputs_final/best_model.joblib", "outputs/best_model.joblib") if os.path.exists(p)),
                "outputs_final/best_model.joblib")
b = joblib.load(path)
feats = b["features"]
d = b.get("defaults")
old_file = d is None or "site_table" not in b
d = d or {}
rmse = None
mj = os.path.join(os.path.dirname(path), "metrics.json")
if os.path.exists(mj):
    try:
        m = json.load(open(mj))
        rmse = m["results"][b["name"]]["te"]["rmse"]
    except Exception:
        pass

# ----------------------------------------------------------------------------- build the feature row
doy = pd.Timestamp(2019, a.month, a.day).dayofyear
src = {}      # feature -> where its value came from


def pick(name, typed, fallback, how):
    """use the typed value if given, else `fallback`, remembering the source."""
    if typed is not None:
        src[name] = "you typed"
        return typed
    src[name] = how
    return fallback


temp_max = pick("temp_max", a.temp_max, a.temperature + d.get("temp_spread", 0.0), "today's temp + typical spread")
src["temp_spread"] = "derived (peak - mean temp)"

if "site_table" in b:
    st = b["site_table"]
    near = ((st.latitude - a.latitude) ** 2 + (st.longitude - a.longitude) ** 2).idxmin()
    site_mean = float(st.site_mean_ozone[near])
    near_info = "average of nearest training station"
else:
    site_mean, near_info = b.get("global_mean", 0.03), "global training average (old model file)"
src["site_mean_ozone"] = near_info

oz_y = a.ozone_yesterday
row = dict(
    temperature=a.temperature, temp_max=temp_max, temp_spread=temp_max - a.temperature, pressure=a.pressure,
    wind=a.wind, wind_max=pick("wind_max", a.wind_max, d.get("wind_max", a.wind), "training median"),
    no2=a.no2, no2_max=pick("no2_max", a.no2_max, d.get("no2_max", a.no2), "training median"),
    voc=pick("voc", a.voc, d.get("voc", 0.0), "training median"), rh=d.get("rh", 0.0),
    doy_sin=np.sin(2 * np.pi * doy / 365.25), doy_cos=np.cos(2 * np.pi * doy / 365.25),
    latitude=a.latitude, longitude=a.longitude, site_mean_ozone=site_mean,
    ozone_lag1=pick("ozone_lag1", oz_y, d.get("ozone_lag1"), "typical value - give --ozone-yesterday"),
    ozone_max_lag1=pick("ozone_max_lag1", a.ozone_max_yesterday, d.get("ozone_max_lag1"), "training median"),
    ozone_roll7=pick("ozone_roll7", a.ozone_7day,
                     oz_y if oz_y is not None else d.get("ozone_roll7"),
                     "= yesterday's ozone" if oz_y is not None else "training median"),
    temp_lag1=pick("temp_lag1", a.temp_yesterday, a.temperature, "= today's temperature"),
    temp_roll3=a.temperature, no2_lag1=pick("no2_lag1", a.no2_yesterday, a.no2, "= today's NO2"),
)
src["temp_roll3"] = "= today's temperature"
for f in ("temperature", "pressure", "wind", "no2", "latitude", "longitude"):
    src[f] = "you typed"
for f in feats:                                 # features this script does not know about
    if f not in row or row[f] is None:
        row[f] = d.get(f)
        src.setdefault(f, "training median")
missing = [f for f in feats if row.get(f) is None or pd.isna(row[f])]
if missing:
    raise SystemExit(f"Cannot predict: no value for {missing}. Provide them as options (see --help) "
                     "or re-train with the current train.py.")

X = b["scaler"].transform(pd.DataFrame([row])[feats])
pred = float(b["model"].predict(X)[0])
ppb = pred * 1000

# ----------------------------------------------------------------------------- display helpers
INFO = {  # feature -> (label, unit, multiplier for display)
    "temperature": ("Mean temperature", "deg F", 1), "temp_max": ("Peak temperature", "deg F", 1),
    "temp_spread": ("Peak minus mean temp", "deg F", 1), "pressure": ("Barometric pressure", "mbar", 1),
    "wind": ("Mean wind speed", "knots", 1), "wind_max": ("Peak wind speed", "knots", 1),
    "no2": ("Mean NO2", "ppb", 1), "no2_max": ("Peak NO2", "ppb", 1), "voc": ("VOC", "ppb", 1),
    "rh": ("Relative humidity", "%", 1), "latitude": ("Latitude", "deg", 1), "longitude": ("Longitude", "deg", 1),
    "site_mean_ozone": ("Station average ozone", "ppb", 1000), "ozone_lag1": ("Yesterday's ozone", "ppb", 1000),
    "ozone_max_lag1": ("Yesterday's peak ozone", "ppb", 1000), "ozone_roll7": ("7-day mean ozone", "ppb", 1000),
    "temp_lag1": ("Yesterday's temperature", "deg F", 1), "temp_roll3": ("3-day mean temperature", "deg F", 1),
    "no2_lag1": ("Yesterday's NO2", "ppb", 1),
}


def fmt(f):
    label, unit, mult = INFO.get(f, (f, "", 1))
    return label, f"{row[f] * mult:.1f} {unit}".strip()


month_name = pd.Timestamp(2019, a.month, a.day).strftime("%B")
typed = [f for f in feats if src.get(f) == "you typed" and f not in ("doy_sin", "doy_cos")]
auto = [f for f in feats if f not in typed and f not in ("doy_sin", "doy_cos")]

# ----------------------------------------------------------------------------- output
print()
box("OZONE PREDICTION DEMO", [
    "TARGET     daily MEAN ground-level ozone at one monitoring station",
    "UNIT       ppb (parts per billion);  1 ppb = 0.001 ppm",
    f"MODEL      {b['name']}  ({len(feats)} input features)",
    f"FILE       {path}",
])

in_rows = [(lab, val, src[f]) for f in typed for lab, val in [fmt(f)]]
in_rows.insert(0, ("Date", f"{month_name} {a.day} (day {doy})", "you typed"))
box("1. WHAT YOU TYPED", table([("INPUT", "VALUE", "SOURCE")] + in_rows, (26, 28, 24)))

box("2. FILLED IN AUTOMATICALLY", table(
    [("INPUT", "VALUE", "HOW IT WAS FILLED")] + [(fmt(f)[0], fmt(f)[1], src[f]) for f in auto], (24, 13, 38)))

# prediction box with a gauge 0..100 ppb
scale = 100
pos = int(round(min(max(ppb, 0), scale) / scale * 50))
std = int(round(70 / scale * 50))
gauge = [(" " if not UNI else "░")] * 51
for i in range(pos + 1):
    gauge[i] = "█" if UNI else "#"
gauge[std] = "|" if gauge[std] in (" ", "░") else gauge[std]
lines = [f"PREDICTED DAILY MEAN OZONE:   {ppb:5.1f} ppb   ({pred:.4f} ppm)", "---"]
lines.append("  0 " + "".join(gauge) + " 100 ppb")
lines.append(" " * (4 + std) + "^ 70 ppb = EPA 8-hour standard (2015)")
if rmse:
    lines += ["---", f"Typical model error: about +/- {rmse * 1000:.1f} ppb (RMSE on the unseen test year)",
              f"So a likely range for today is {max(ppb - rmse * 1000, 0):.1f} to {ppb + rmse * 1000:.1f} ppb."]
box("3. PREDICTION", lines)

notes = []
sm = row["site_mean_ozone"] * 1000
diff = ppb - sm
notes.append(f"vs. this station's average ({sm:.1f} ppb): {abs(diff):.1f} ppb {'ABOVE' if diff >= 0 else 'BELOW'} normal.")
if oz_y is not None:
    dy = ppb - oz_y * 1000
    notes.append(f"vs. yesterday ({oz_y * 1000:.1f} ppb): {'up' if dy >= 0 else 'down'} {abs(dy):.1f} ppb.")
notes.append("The 70 ppb legal limit uses the 8-HOUR MAXIMUM, which is higher than a daily mean, so a mean below 70 ppb does NOT guarantee the standard is met.")
if "ozone_lag1" in feats and oz_y is None:
    notes.append("WARNING: --ozone-yesterday was not given; the model used a typical value, so accuracy is lower.")
if old_file:
    notes.append("WARNING: model file is from an old train.py; re-train for full demo support.")
box("4. HOW TO READ THIS", notes)
print()