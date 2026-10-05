"""Generate the PDF write-up (max 2 pages) from the results of train.py.

Run AFTER training (daily data):
  python src/make_report.py --metrics outputs/metrics.json --outdir outputs --data data/ozone_daily.csv \
      --val-year 2018 --test-year 2019 --team "Name1 (SRN1), Name2 (SRN2)" --problem "17" --out report.pdf

Every number in the report is read from metrics.json / the CSV, so re-run this whenever you re-train.
"""
import argparse
import json
import os

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

SCATTER = {"Linear Regression": "scatter_linear.png", "SVR (RBF)": "scatter_svr.png",
           "Neural Network (MLP)": "scatter_neural.png", "Random Forest": "scatter_random.png"}
LABEL = {"temperature": "temperature", "temp_p90": "temperature (90th pct)", "pressure": "pressure",
         "wind": "wind speed", "no2": "NO2", "voc": "VOC", "month_sin": "month (sin)",
         "month_cos": "month (cos)", "latitude": "latitude", "longitude": "longitude"}


def facts(a, m):
    f = dict(rows=m["n_train"] + m["n_val"] + m["n_test"], sites=None,
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


def build(a):
    m = json.load(open(a.metrics))
    R = m["results"]
    fx = facts(a, m)
    names = list(R)
    te = {n: R[n]["te"] for n in names}
    best_te = max(names, key=lambda n: te[n]["r2"])
    worst_te = min(names, key=lambda n: te[n]["r2"])
    gap = {n: R[n]["tr"]["r2"] - R[n]["te"]["r2"] for n in names}
    overfit = max(names, key=lambda n: gap[n])
    best_val = m["best_model"]
    lr = R.get("Linear Regression")

    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["Normal"], fontName="Helvetica", fontSize=8.1, leading=9.7, spaceAfter=2)
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=14, leading=17, spaceAfter=2)
    sub = ParagraphStyle("sub", parent=body, alignment=1, fontSize=8.4, textColor=colors.HexColor("#444444"))
    h2 = ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=10, leading=12, spaceBefore=5, spaceAfter=2,
                        textColor=colors.HexColor("#1F3A5F"))
    cap = ParagraphStyle("cap", parent=body, fontSize=7.6, leading=9, alignment=1, textColor=colors.HexColor("#444444"))
    small = ParagraphStyle("s", parent=body, fontSize=7.4, leading=9)

    story = [Paragraph("Predicting Daily Ground-Level Ozone from Weather and Precursor Pollutants", h1),
             Paragraph(f"UE24CS352A Machine Learning - Mini-Project &nbsp;|&nbsp; Team: {a.team} &nbsp;|&nbsp; "
                       f"Problem no.: {a.problem}", sub)]

    # 1 Problem
    story += [Paragraph("1. Problem statement", h2), Paragraph(
        "Ground-level (tropospheric) ozone is a harmful pollutant and a greenhouse gas. Regions that exceed the EPA "
        "limit are designated <i>nonattainment</i> and face costly oversight, so being able to predict ozone early is "
        "useful. Following the assigned problem (ozone prediction, Sutton, CS229 2019), we predict the <b>daily ozone "
        "concentration (ppm) at a monitoring site</b> from temperature, pressure, wind speed, month, location and "
        "NO<sub>2</sub> level. This is a supervised <b>regression</b> task. NO<sub>2</sub> is an ozone precursor, and "
        "temperature and pressure act as proxies for sunlight.", body)]

    # 2 Dataset
    sites = f" from {fx['sites']:,} monitoring sites" if fx["sites"] else ""
    story += [Paragraph("2. Dataset", h2), Paragraph(
        f"We use the <b>US EPA Air Quality System daily summary data</b> (EPA AirData; the same source as the "
        f"BigQuery 'EPA Historical Air Quality' dataset): ozone (parameter 44201), NO<sub>2</sub> (42602), outdoor "
        f"temperature, barometric pressure and wind speed, for {fx['years']}. Each parameter's daily mean was averaged "
        f"per site and day and the five tables were inner-joined on (site, date), giving <b>{fx['rows']:,} rows</b>"
        f"{sites} after removing rows with missing values. Features: temperature, pressure, wind, NO<sub>2</sub>, "
        f"latitude, longitude and month (encoded as sin/cos because months are cyclic). Target: daily mean ozone (ppm). "
        f"VOCs were left out because very few sites measure them, which would have removed most rows.", body)]

    # 3 Approach
    story += [Paragraph("3. Approach", h2), Paragraph(
        f"<b>Time-based split</b> to avoid leakage between neighbouring days: train {fx['train_years']} "
        f"({m['n_train']:,} rows), validation {fx['val_years']} ({m['n_val']:,}), test {fx['test_years']} "
        f"({m['n_test']:,}). Features were <b>min-max scaled using training data only</b>. We compared four models: "
        f"(1) <b>Linear Regression</b> as the baseline; (2) <b>Support Vector Regression</b> with an RBF kernel; "
        f"(3) a <b>Neural Network</b> (MLP, three hidden layers of 128-128-64 units, ReLU, Adam, early stopping); and "
        f"(4) a <b>Random Forest</b> (150 trees) as an extra non-linear baseline. Models were scored with R<super>2</super>, RMSE "
        f"and MAE, and the best model was chosen on the <i>validation</i> year so the test set stays an honest final "
        f"check.", body)]

    # 4 Implementation
    sp = m.get("svr_params", {})
    spt = f" (best: C={sp.get('C')}, gamma={sp.get('gamma')}, epsilon={float(sp.get('epsilon', 0)):.4f})" if sp else ""
    story += [Paragraph("4. Implementation overview", h2), Paragraph(
        "Python with pandas and scikit-learn. <font face='Courier'>data_epa_csv.py</font> merges the EPA daily files "
        "into one table; <font face='Courier'>train.py</font> cleans, splits, scales, trains and evaluates all models, "
        "saves plots, metrics and the best model; <font face='Courier'>demo.py</font> loads the saved model for live "
        "predictions. SVR is slow on large data, so it was tuned (RandomizedSearchCV over C, gamma and epsilon, 3-fold "
        f"CV) on a 15,000-row sample{spt}. Epsilon is searched relative to the spread of the target: the reference "
        "report's SVR failed because its epsilon was far too large for ozone values of about 0.03 ppm. Feature "
        "importance was measured by permutation on the validation set.", body)]

    # 5 Results
    story.append(Paragraph("5. Results", h2))
    hdr = ["Model", "R2 train", "R2 val", "R2 test", "RMSE test (ppm)", "MAE test (ppm)"]
    rows = [hdr] + [[n, f"{R[n]['tr']['r2']:.3f}", f"{R[n]['va']['r2']:.3f}", f"{R[n]['te']['r2']:.3f}",
                     f"{R[n]['te']['rmse']:.4f}", f"{R[n]['te']['mae']:.4f}"] for n in names]
    t = Table(rows, colWidths=[4.6 * cm, 2.2 * cm, 2.2 * cm, 2.2 * cm, 3.2 * cm, 3.2 * cm])
    style = [("FONT", (0, 0), (-1, -1), "Helvetica", 8), ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
             ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6ECF4")), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
             ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    bi = names.index(best_te) + 1
    style += [("BACKGROUND", (0, bi), (-1, bi), colors.HexColor("#E8F4E8")), ("FONT", (0, bi), (-1, bi), "Helvetica-Bold", 8)]
    t.setStyle(TableStyle(style))
    story += [t, Spacer(1, 3)]

    imgs = []
    sp_path = os.path.join(a.outdir, SCATTER.get(best_te, ""))
    fi_path = os.path.join(a.outdir, "feature_importance.png")
    for path, caption, w in [(sp_path, f"Fig. 1: actual vs predicted ozone on the test set ({best_te}).", 4.3 * cm),
                             (fi_path, f"Fig. 2: permutation feature importance ({best_val}).", 5.6 * cm)]:
        if os.path.isfile(path):
            from reportlab.lib.utils import ImageReader
            iw, ih = ImageReader(path).getSize()
            imgs.append([Image(path, width=w, height=w * ih / iw), Paragraph(caption, cap)])
    if imgs:
        ft = Table([[c[0] for c in imgs], [c[1] for c in imgs]])
        ft.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        story += [ft, Spacer(1, 3)]

    b = te[best_te]
    fi = m.get("feature_importance") or {}
    top = sorted(fi, key=fi.get, reverse=True)[:3]
    txt = (f"The <b>{best_te}</b> scored best on the held-out test years (R<super>2</super> = {b['r2']:.3f}, i.e. it explains about "
           f"{b['r2'] * 100:.0f}% of the variation in daily ozone; typical error: RMSE {b['rmse'] * 1000:.1f} ppb, MAE "
           f"{b['mae'] * 1000:.1f} ppb). ")
    if best_val != best_te:
        txt += f"{best_val} was selected on the validation year. "
    if lr and best_te != "Linear Regression":
        txt += (f"The linear baseline reached only R<super>2</super> = {lr['te']['r2']:.3f}, so the relationship between weather, "
                f"NO<sub>2</sub> and ozone is clearly non-linear. ")
    txt += (f"<b>{overfit}</b> shows the largest gap between training (R<super>2</super> = {R[overfit]['tr']['r2']:.3f}) and test "
            f"(R<super>2</super> = {R[overfit]['te']['r2']:.3f}), a sign of overfitting. ")
    if top:
        txt += (f"For the {best_val} the most influential features were "
                + ", ".join(LABEL.get(x, x) for x in top) + ". ")
    story.append(Paragraph(txt, body))

    # 6 Conclusions
    story += [Paragraph("6. Conclusions", h2), Paragraph(
        f"Non-linear models predict daily ozone considerably better than a linear model, and the best result "
        f"(R<super>2</super> = {b['r2']:.2f} on unseen later years) is in a sensible range for this problem. Evaluating on later "
        f"years rather than a random split gives a realistic estimate of forecasting performance. "
        f"<b>Limitations:</b> important drivers are missing (sunlight, humidity, traffic and industrial sources, VOCs); "
        f"only sites that measure all five parameters are included; the target is the daily mean rather than the "
        f"regulatory 8-hour maximum; SVR was trained on a subsample; and results depend on the chosen years. "
        f"<b>Future work:</b> add humidity and census/land-use features, predict the 8-hour maximum or classify "
        f"EPA nonattainment levels, and try gradient boosting.", body),
        Paragraph("<b>References:</b> M. Sutton, <i>Tropospheric ozone prediction</i>, CS229 Final Project Report, 2019 "
                  "(assigned problem). US EPA, Air Quality System AirData daily summary files, aqs.epa.gov/aqsweb/airdata. "
                  "Pedregosa et al., Scikit-learn: Machine Learning in Python, JMLR 12, 2011.", small)]

    def footer(c, d):
        c.saveState()
        c.setFont("Helvetica", 7)
        c.setFillColor(colors.grey)
        c.drawCentredString(A4[0] / 2, 0.9 * cm, f"Page {d.page}")
        c.restoreState()

    doc = SimpleDocTemplate(a.out, pagesize=A4, leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.1 * cm,
                            bottomMargin=1.3 * cm, title="ML Mini-Project Write-up", author=a.team)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("Saved", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="outputs/metrics.json")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--data", default="data/ozone_daily.csv")
    ap.add_argument("--val-year", type=int, default=2018)
    ap.add_argument("--test-year", type=int, default=2019)
    ap.add_argument("--years", default=None, help="only used if --data is missing, e.g. 2016-2019")
    ap.add_argument("--team", default="[Name 1 (SRN)], [Name 2 (SRN)]")
    ap.add_argument("--problem", default="[N]")
    ap.add_argument("--out", default="report.pdf")
    build(ap.parse_args())