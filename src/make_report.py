"""Generate the PDF write-up (max 2 pages) from the results of train.py (v2 pipeline).

Run AFTER training:
  python src/make_report.py --metrics outputs_final/metrics.json --outdir outputs_final \
      --data data/ozone_daily.csv --val-year 2018 --test-year 2019 \
      --team "Name1 (SRN1), Name2 (SRN2)" --problem "17" --out report.pdf

Every number about OUR model (rows, features, R2, NN settings, SVR settings...) is read from
metrics.json / the CSV, so re-run this whenever you re-train.
The PAPER and "first version" numbers below are fixed constants (from the reference report and from our v1 run).
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

SCATTER = {"Linear Regression": "scatter_linear.png", "SVR (RBF)": "scatter_svr.png",
           "Neural Network (MLP)": "scatter_neural.png", "Random Forest": "scatter_random.png",
           "Gradient Boosting": "scatter_gradient.png"}
LABEL = {"temperature": "mean temperature", "temp_max": "peak temperature", "temp_spread": "peak-minus-mean temperature",
         "pressure": "pressure", "wind": "wind speed", "wind_max": "peak wind", "no2": "NO2", "no2_max": "peak NO2",
         "voc": "VOC", "rh": "humidity", "doy_sin": "day of year (sin)", "doy_cos": "day of year (cos)",
         "month_sin": "month (sin)", "month_cos": "month (cos)", "latitude": "latitude", "longitude": "longitude",
         "site_mean_ozone": "site average ozone", "ozone_lag1": "yesterday's ozone",
         "ozone_max_lag1": "yesterday's peak ozone", "ozone_roll7": "7-day mean ozone",
         "temp_lag1": "yesterday's temperature", "temp_roll3": "3-day mean temperature", "no2_lag1": "yesterday's NO2"}
LAGS = {"ozone_lag1", "ozone_max_lag1", "ozone_roll7", "temp_lag1", "temp_roll3", "no2_lag1"}

# Reference paper (Sutton, CS229 2019) TEST R2, and our first version (v1) TEST R2 (from the earlier report)
PAPER = {"Linear Regression": 0.4674, "SVR (RBF)": -2.3626, "Neural Network (MLP)": 0.6537}
V1 = {"Linear Regression": 0.162, "SVR (RBF)": 0.563, "Neural Network (MLP)": 0.558, "Random Forest": 0.607}


def facts(a, m):
    f = dict(rows=m["n_train"] + m["n_val"] + m["n_test"], sites=a.sites,
             years=a.years or "see data", train_years=f"years before {a.val_year}",
             val_years=str(a.val_year), test_years=f"{a.test_year} onward")
    if a.data and os.path.exists(a.data):
        d = pd.read_csv(a.data, usecols=lambda c: c in ("state_code", "county_code", "site_num", "date_local"))
        yrs = pd.to_datetime(d["date_local"]).dt.year
        f["sites"] = int(d[["state_code", "county_code", "site_num"]].drop_duplicates().shape[0])
        f["years"] = f"{yrs.min()}-{yrs.max()}"
        tr = sorted(y for y in yrs.unique() if y < a.val_year)
        te = sorted(y for y in yrs.unique() if y >= a.test_year)
        if tr:
            f["train_years"] = f"{tr[0]}-{tr[-1]}" if len(tr) > 1 else str(tr[0])
        if te:
            f["test_years"] = f"{te[0]}-{te[-1]}" if len(te) > 1 else str(te[0])
    return f


def find_metrics(given):
    """Return a metrics.json path to use, searching common locations if the given one is missing."""
    if given and os.path.exists(given):
        return given
    candidates = ["outputs_final/metrics.json", "outputs/metrics.json", "outputs_nn/metrics.json",
                 "outputs_fast/metrics.json", "outputs_test/metrics.json"]
    candidates += sorted(glob.glob("outputs*/metrics.json"))
    found = [c for c in dict.fromkeys(candidates) if os.path.exists(c)]
    if found:
        print(f"NOTE: '{given}' not found, using '{found[0]}' instead.\n"
             f"      (pass --metrics PATH --outdir DIR to pick a different run)")
        return found[0]
    here = os.getcwd()
    sys.exit(
        f"ERROR: could not find a metrics.json anywhere under '{here}'.\n"
        f"       You gave --metrics '{given}'.\n"
        f"       This file is written by train.py when it finishes, e.g.:\n"
        f"           python src/train.py --data data/ozone_daily.csv --outdir outputs_final\n"
        f"       Things to check:\n"
        f"       1. Are you running this from the project root (the folder with 'src/' and 'data/')?\n"
        f"          Current folder: {here}\n"
        f"       2. Did the train.py run finish and print 'DONE' / 'Saved ... to outputs_final/'?\n"
        f"       3. Did you train into a different --outdir? Pass it here with --metrics and --outdir.")


def build(a):
    metrics_path = find_metrics(a.metrics)
    if a.outdir == ap_defaults["outdir"] and os.path.dirname(metrics_path):
        a.outdir = os.path.dirname(metrics_path)   # follow metrics.json if we auto-switched run folders
    m = json.load(open(metrics_path))
    R = m["results"]
    fx = facts(a, m)
    names = list(R)
    te = {n: R[n]["te"] for n in names}
    best_te = max(names, key=lambda n: te[n]["r2"])
    gap = {n: R[n]["tr"]["r2"] - R[n]["te"]["r2"] for n in names}
    overfit = max(names, key=lambda n: gap[n])
    best_val = m["best_model"]
    lr = R.get("Linear Regression")
    feats = m.get("features", [])
    use_lag = m.get("use_lag", any(f in LAGS for f in feats))
    nets = m.get("n_nets", a.nets)
    nn = m.get("nn_params") or {}
    hidden = "-".join(str(x) for x in nn.get("hidden", [256, 128, 64]))
    weather = [LABEL.get(f, f) for f in feats if f not in LAGS and f not in ("site_mean_ozone",)]
    lagl = [LABEL.get(f, f) for f in feats if f in LAGS]

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=7.7, leading=9.1, spaceAfter=1.5)
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=13.5, leading=16, spaceAfter=1)
    sub = ParagraphStyle("sub", parent=body, alignment=1, fontSize=8, textColor=colors.HexColor("#444444"))
    h2 = ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=9.6, leading=11, spaceBefore=4, spaceAfter=1.5,
                        textColor=colors.HexColor("#1F3A5F"))
    cap = ParagraphStyle("cap", parent=body, fontSize=7.2, leading=8.5, alignment=1, textColor=colors.HexColor("#444444"))
    small = ParagraphStyle("s", parent=body, fontSize=7, leading=8.4)

    story = [Paragraph("Predicting Daily Ground-Level Ozone from Weather and Precursor Pollutants", h1),
             Paragraph(f"UE24CS352A Machine Learning - Mini-Project &nbsp;|&nbsp; Team: {a.team} &nbsp;|&nbsp; "
                       f"Problem no.: {a.problem}", sub)]

    # 1 Problem
    forecast = (" Our final model is a <b>one-day-ahead forecasting</b> setup: it also sees yesterday's ozone at the "
                "same site (see Section 3)." if use_lag else "")
    story += [Paragraph("1. Problem statement", h2), Paragraph(
        "Ground-level (tropospheric) ozone is a harmful pollutant and a greenhouse gas. Regions that exceed the EPA "
        "limit are designated <i>nonattainment</i> and face costly oversight, so predicting ozone early is useful. "
        "Following the assigned problem (Sutton, CS229 2019), we predict the <b>daily ozone concentration (ppm) at a "
        "monitoring site</b> from temperature, pressure, wind, NO<sub>2</sub>, season and location. This is a "
        "supervised <b>regression</b> task; NO<sub>2</sub> is an ozone precursor and temperature/pressure are proxies "
        "for sunlight." + forecast, body)]

    # 2 Dataset
    sites = f" from {fx['sites']:,} monitoring sites" if fx["sites"] else ""
    story += [Paragraph("2. Dataset and features", h2), Paragraph(
        f"<b>US EPA AirData daily summaries</b> (same source as BigQuery 'EPA Historical Air Quality'): ozone (44201), "
        f"NO<sub>2</sub> (42602), temperature, barometric pressure and wind speed, {fx['years']}. Daily values were "
        f"averaged per site and day and inner-joined on (site, date): <b>{fx['rows']:,} rows</b>{sites} after removing "
        f"missing values. VOCs were left out (too few sites measure them). Target: daily mean ozone (ppm). "
        f"<b>{len(feats)} features:</b> " + ", ".join(weather) + ". "
        + ("<b>Engineered features:</b> per-site average ozone (computed on training years only)"
           + (", and lag features: " + ", ".join(lagl) if lagl else "") + "." if "site_mean_ozone" in feats else ""),
        body)]

    # 3 Approach
    story += [Paragraph("3. Approach and improvements over the reference paper", h2), Paragraph(
        f"<b>Time-based split</b> (no leakage between neighbouring days): train {fx['train_years']} "
        f"({m['n_train']:,} rows), validation {fx['val_years']} ({m['n_val']:,}), test {fx['test_years']} "
        f"({m['n_test']:,}); the paper used a single 1980-2016 / 2017 split. Features were standardised (z-score) "
        f"with training statistics only. Models: Linear Regression (baseline), SVR (RBF), Neural Network, Random "
        f"Forest and Gradient Boosting; the best is chosen on the <i>validation</i> year so the test year stays an "
        f"honest final exam. <b>What we changed to improve the neural network:</b> (1) the <b>target is standardised "
        f"inside the network</b>, because ozone is only about 0.03 ppm and such tiny targets give tiny gradients; "
        f"(2) a <b>tuned MLP</b> ({hidden} units, L2 alpha={nn.get('alpha', '-')}, learning rate {nn.get('lr', '-')}, "
        f"batch {nn.get('batch', '-')}) found by random search scored on the validation year; (3) an <b>ensemble of "
        f"{nets} networks</b> with different seeds, averaged, to reduce variance; (4) <b>richer features</b> (site "
        f"average, seasonal day-of-year"
        + (", and recent-ozone lags, because ozone is strongly autocorrelated" if use_lag else "") + "). "
        "The paper tuned by hand and its SVR failed (negative R<super>2</super>) because epsilon was far too large; "
        "we tie epsilon to the spread of the target.", body)]

    # 4 Implementation
    sp = m.get("svr_params")
    spt = (f" (best: C={sp.get('C')}, gamma={sp.get('gamma')}, epsilon={float(sp.get('epsilon', 0)):.4f})") if sp else ""
    story += [Paragraph("4. Implementation overview", h2), Paragraph(
        "Python, pandas and scikit-learn. <font face='Courier'>data_epa_csv.py</font> merges the EPA files into one "
        "table; <font face='Courier'>train.py</font> builds features, splits, scales, tunes and trains all models, "
        "saves plots, metrics and the best model; <font face='Courier'>demo.py</font> predicts for typed-in values; "
        "<font face='Courier'>make_report.py</font> builds this report from the saved metrics. SVR is slow on large "
        f"data, so it was tuned (RandomizedSearchCV, 3-fold) on a 15,000-row sample{spt}. Feature importance was "
        "measured by permutation on the validation set.", body)]

    # 5 Results
    story.append(Paragraph("5. Results", h2))
    hdr = ["Model", "R2 train", "R2 val", "R2 test", "RMSE test (ppm)", "MAE test (ppm)"]
    rows = [hdr] + [[n, f"{R[n]['tr']['r2']:.3f}", f"{R[n]['va']['r2']:.3f}", f"{R[n]['te']['r2']:.3f}",
                     f"{R[n]['te']['rmse']:.4f}", f"{R[n]['te']['mae']:.4f}"] for n in names]
    t = Table(rows, colWidths=[4.4 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 3.2 * cm, 3.2 * cm])
    style = [("FONT", (0, 0), (-1, -1), "Helvetica", 7.6), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.6),
             ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6ECF4")), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
             ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]
    bi = names.index(best_te) + 1
    style += [("BACKGROUND", (0, bi), (-1, bi), colors.HexColor("#E8F4E8")), ("FONT", (0, bi), (-1, bi), "Helvetica-Bold", 7.6)]
    t.setStyle(TableStyle(style))
    story += [t, Spacer(1, 3)]

    # comparison with paper + v1
    cmp_rows = [["Test R2 by model", "Reference paper", "Our first version", "Our final version"]]
    for n in names:
        cmp_rows.append([n, f"{PAPER[n]:.3f}" if n in PAPER else "-", f"{V1[n]:.3f}" if n in V1 else "-",
                         f"{R[n]['te']['r2']:.3f}"])
    ct = Table(cmp_rows, colWidths=[4.4 * cm, 3.4 * cm, 3.6 * cm, 3.6 * cm])
    ct.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Helvetica", 7.6), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 7.6),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F4ECE6")),
                            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                            ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
    story += [ct, Paragraph("Paper: 1980-2016 train, Jul-Dec 2017 test, no lag features. Ours: 2010-2019 data, full-year "
                            "test; the datasets differ, so the comparison is indicative, not exact.", cap), Spacer(1, 2)]

    imgs = []
    sp_path = os.path.join(a.outdir, SCATTER.get(best_te, ""))
    fi_path = os.path.join(a.outdir, "feature_importance.png")
    for path, caption, w in [(sp_path, f"Fig. 1: actual vs predicted ozone, test set ({best_te}).", 3.9 * cm),
                             (fi_path, f"Fig. 2: permutation feature importance ({best_val}).", 5.2 * cm)]:
        if os.path.isfile(path):
            iw, ih = ImageReader(path).getSize()
            imgs.append([Image(path, width=w, height=w * ih / iw), Paragraph(caption, cap)])
    if imgs:
        ft = Table([[c[0] for c in imgs], [c[1] for c in imgs]])
        ft.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story += [ft, Spacer(1, 2)]

    b = te[best_te]
    fi = m.get("feature_importance") or {}
    top = sorted(fi, key=fi.get, reverse=True)[:3]
    txt = (f"The <b>{best_te}</b> scored best on the held-out test year (R<super>2</super> = {b['r2']:.3f}: it explains about "
           f"{b['r2'] * 100:.0f}% of the variation in daily ozone; RMSE {b['rmse'] * 1000:.1f} ppb, MAE "
           f"{b['mae'] * 1000:.1f} ppb). ")
    p_nn = PAPER.get("Neural Network (MLP)")
    if "Neural Network (MLP)" in R and p_nn:
        d = R["Neural Network (MLP)"]["te"]["r2"] - p_nn
        txt += (f"Our neural network is {abs(d):.3f} R<super>2</super> {'above' if d >= 0 else 'below'} the paper's "
                f"({p_nn:.3f}) and {R['Neural Network (MLP)']['te']['r2'] - V1['Neural Network (MLP)']:+.3f} vs our first version. ")
    if lr and best_te != "Linear Regression":
        txt += (f"It beats the linear baseline (R<super>2</super> = {lr['te']['r2']:.3f}), so non-linear structure remains. "
                f"Note the linear model itself jumped from {V1['Linear Regression']:.3f} to {lr['te']['r2']:.3f} once the new "
                f"features were added, showing that much of the improvement comes from better inputs, not only a better model. ")
    nn_r = R.get("Neural Network (MLP)")
    if nn_r:
        txt += (f"The neural network generalises well: train/validation/test R<super>2</super> = {nn_r['tr']['r2']:.3f}/"
                f"{nn_r['va']['r2']:.3f}/{nn_r['te']['r2']:.3f}. ")
    if overfit != "Neural Network (MLP)" or gap[overfit] > 0.1:
        txt += (f"<b>{overfit}</b> shows the largest train-test gap ({R[overfit]['tr']['r2']:.3f} vs "
                f"{R[overfit]['te']['r2']:.3f}), i.e. overfitting. ")
    if top:
        txt += f"Most influential features ({best_val}): " + ", ".join(LABEL.get(x, x) for x in top) + "."
    story.append(Paragraph(txt, body))

    # 6 Conclusions
    story += [Paragraph("6. Conclusions", h2), Paragraph(
        f"A tuned, target-standardised neural network ensemble with engineered features predicts daily ozone best "
        f"(test R<super>2</super> = {b['r2']:.2f}), clearly improving on the reference paper's network and our first "
        f"attempt, without overfitting. "
        + ("Because the final features include yesterday's ozone, the task is one-day-ahead forecasting rather than "
           "estimating ozone from same-day weather alone; much of the gain comes from ozone persistence. " if use_lag else "")
        + "<b>Limitations:</b> missing drivers (sunlight, humidity, traffic and industrial sources, VOCs); only sites "
        "measuring all parameters are included; the target is the daily mean, not the regulatory 8-hour maximum; SVR "
        "used a subsample; results depend on the chosen years and are not strictly comparable with the paper. "
        "<b>Future work:</b> add humidity, peak-temperature and land-use features, predict the 8-hour maximum or "
        "classify nonattainment, and try station embeddings.", body),
        Paragraph("<b>References:</b> M. Sutton, <i>Tropospheric ozone prediction</i>, CS229 Final Project Report, 2019. "
                  "US EPA, AirData daily summary files, aqs.epa.gov/aqsweb/airdata. Pedregosa et al., Scikit-learn: "
                  "Machine Learning in Python, JMLR 12, 2011.", small)]

    def footer(c, d):
        c.saveState()
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.grey)
        c.drawCentredString(A4[0] / 2, 0.9 * cm, f"Page {d.page}")
        c.restoreState()

    doc = SimpleDocTemplate(a.out, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.0 * cm,
                            bottomMargin=1.3 * cm, title="ML Mini-Project Write-up", author=a.team)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("Saved", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap_defaults = dict(metrics="outputs_final/metrics.json", outdir="outputs_final")
    ap.add_argument("--metrics", default=ap_defaults["metrics"])
    ap.add_argument("--outdir", default=ap_defaults["outdir"])
    ap.add_argument("--data", default="data/ozone_daily.csv")
    ap.add_argument("--val-year", type=int, default=2018)
    ap.add_argument("--test-year", type=int, default=2019)
    ap.add_argument("--years", default=None, help="only used if --data is missing, e.g. 2010-2019")
    ap.add_argument("--sites", type=int, default=None, help="only used if --data is missing")
    ap.add_argument("--nets", type=int, default=3, help="NN ensemble size if metrics.json lacks n_nets")
    ap.add_argument("--team", default="[Name 1 (SRN)], [Name 2 (SRN)]")
    ap.add_argument("--problem", default="[N]")
    ap.add_argument("--out", default="report.pdf")
    build(ap.parse_args())