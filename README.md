# Predicting Daily Ground-Level Ozone with Machine Learning

UE24CS352A Machine Learning - Mini-Project

**Team:** Eshwar PES1UG24CS663,Soumya PES1UG24CS628   
 

## What this project does

Ground-level (tropospheric) ozone is a harmful pollutant. This project predicts the **daily ozone concentration (ppm) at a US monitoring site** from temperature, pressure, wind speed, NO2, month and location. It is a supervised **regression** problem, and four models are compared: Linear Regression, Support Vector Regression (RBF), a Neural Network (MLP) and a Random Forest.

## Dataset (not included in this repo)

US EPA AirData **daily summary** files (https://aqs.epa.gov/aqsweb/airdata/download_files.html). The raw files are large, so they are git-ignored and must be downloaded before running.

| Parameter | File name pattern |
|---|---|
| Ozone (target) | `daily_44201_YYYY.zip` |
| NO2 | `daily_42602_YYYY.zip` |
| Temperature | `daily_TEMP_YYYY.zip` |
| Barometric pressure | `daily_PRESS_YYYY.zip` |
| Wind | `daily_WIND_YYYY.zip` |

Download the same years for all five parameters into `data/raw/`. We used 2010-2019. In PowerShell:

```powershell
New-Item -ItemType Directory -Force data\raw | Out-Null
foreach ($f in @("44201","42602","TEMP","PRESS","WIND")) { foreach ($y in 2010..2019) {
  $u = "https://aqs.epa.gov/aqsweb/airdata/daily_${f}_$y.zip"
  try { Invoke-WebRequest $u -OutFile "data\raw\daily_${f}_$y.zip"; "ok  $u" } catch { "FAIL $u" }
}}
```

The zip files can stay zipped; the loader reads them directly.

## Project structure

```
.
├── src/
│   ├── data_epa_csv.py   # merges the EPA daily files into data/ozone_daily.csv
│   ├── train.py          # preprocessing, training, evaluation, plots, saves best model
│   ├── demo.py           # predicts ozone for one set of inputs using the saved model
│   └── make_report.py    # builds the PDF write-up from the results
├── data/                 # (git-ignored) raw downloads and ozone_daily.csv
├── outputs/              # results table, metrics, plots (model file is git-ignored)
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

## How to run

Run these from the repository root, in order.

**1. Build the dataset**
```bash
python src/data_epa_csv.py --raw data/raw --out data/ozone_daily.csv --no-voc
```
`--no-voc` leaves out VOC measurements, which few sites record; including them removes most rows.

**2. Train and evaluate**
```bash
python src/train.py --data data/ozone_daily.csv --val-year 2018 --test-year 2019
```
Trains on years before 2018, validates on 2018 and tests on 2019 onward. Writes `outputs/results_table.md`, `outputs/metrics.json`, `outputs/*.png` and `outputs/best_model.joblib`. Useful options: `--svr-samples` (rows used to tune SVR, default 15000) and `--svr-iter` (search iterations, default 8).

**3. Try a prediction (demo)**
```bash
python src/demo.py --temperature 90 --pressure 1010 --wind 2.5 --no2 18 --month 7 --latitude 34.05 --longitude -118.24
```
Units follow EPA data: temperature in degrees Fahrenheit, pressure in millibars, wind in knots, NO2 in ppb.

**4. Generate the PDF write-up (optional)**
```bash
python src/make_report.py --metrics outputs/metrics.json --outdir outputs --data data/ozone_daily.csv --val-year 2018 --test-year 2019 --team "Your Name (SRN), Partner Name (SRN)" --problem "YOUR_PROBLEM_NUMBER" --out report.pdf
```

## Method in brief

- **Features:** temperature, pressure, wind speed, NO2, latitude, longitude, and month encoded as sin/cos (months are cyclic).
- **Target:** daily mean ozone (ppm).
- **Split by time, not at random,** to avoid leakage between neighbouring days: train before 2018, validation 2018, test 2019 onward.
- **Scaling:** min-max to 0-1, fitted on the training data only.
- **Models:** Linear Regression (baseline); SVR with RBF kernel (C, gamma, epsilon tuned by randomized search with 3-fold CV on a 15,000-row sample); a Neural Network (hidden layers 128-128-64, ReLU, Adam, early stopping); a Random Forest (150 trees).
- **Model selection:** best model chosen on the validation year, then scored once on the test years.
- **Metrics:** R2, RMSE and MAE. Feature importance is measured by permutation.

## Results

Train = 179,473 rows (years before 2018); validation = 28,536 rows (2018); test = 27,987 rows (2019).

| Model | R2 train | R2 val | R2 test | RMSE test (ppm) | MAE test (ppm) |
|---|---|---|---|---|---|
| Linear Regression | 0.336 | 0.385 | 0.162 | 0.0101 | 0.0074 |
| SVR (RBF) | 0.615 | 0.611 | 0.563 | 0.0073 | 0.0056 |
| Neural Network (MLP) | 0.619 | 0.622 | 0.558 | 0.0074 | 0.0055 |
| Random Forest | 0.899 | 0.654 | 0.607 | 0.0069 | 0.0054 |

The non-linear models clearly beat the linear baseline. The Random Forest scores best on the test years but overfits (0.90 on train); SVR and the neural network generalise more evenly.

## Limitations

- Only sites that measure all five parameters are used, and VOCs are excluded.
- No humidity, sunlight, traffic or industrial-source features.
- The target is the daily mean ozone, not the regulatory 8-hour maximum.
- SVR is tuned and trained on a 15,000-row sample because it scales poorly with data size.
- The test set is a single year (2019), so scores can shift if different years are used for validation and testing.

## Reproducibility

Random seeds are fixed (`random_state=0`) for the SVR search, neural network and random forest. The data split is deterministic (by year).

## Reference

M. Sutton, *Tropospheric ozone prediction*, CS229 Final Project Report, 2019 (assigned problem). Pedregosa et al., Scikit-learn: Machine Learning in Python, JMLR 12, 2011.
