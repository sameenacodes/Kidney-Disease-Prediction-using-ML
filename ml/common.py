"""Shared data cleaning + feature definitions used by training and the web app."""
import numpy as np
import pandas as pd

# Numeric lab values: key -> (label, unit, normal-range text)
NUMERIC = {
    "age":  ("Age", "years", ""),
    "bp":   ("Blood Pressure (diastolic)", "mm/Hg", "60-80"),
    "sg":   ("Specific Gravity", "", "1.005-1.025"),
    "al":   ("Albumin (urine, 0-5 scale)", "", "0"),
    "su":   ("Sugar (urine, 0-5 scale)", "", "0"),
    "bgr":  ("Blood Glucose Random", "mg/dL", "70-140"),
    "bu":   ("Blood Urea", "mg/dL", "7-20"),
    "sc":   ("Serum Creatinine", "mg/dL", "0.6-1.3"),
    "sod":  ("Sodium", "mEq/L", "135-145"),
    "pot":  ("Potassium", "mEq/L", "3.5-5.0"),
    "hemo": ("Hemoglobin", "g/dL", "12-17"),
    "pcv":  ("Packed Cell Volume", "%", "36-50"),
    "wc":   ("White Blood Cell Count", "cells/cumm", "4000-11000"),
    "rc":   ("Red Blood Cell Count", "million/cmm", "4.2-6.0"),
}
# Categorical values: key -> (label, positive-label shown as 1)
CATEG = {
    "rbc":   ("Red Blood Cells", "abnormal"),
    "pc":    ("Pus Cell", "abnormal"),
    "htn":   ("Hypertension", "yes"),
    "dm":    ("Diabetes Mellitus", "yes"),
    "appet": ("Appetite", "poor"),
    "pe":    ("Pedal Edema", "yes"),
    "ane":   ("Anemia", "yes"),
}
FEATURES = list(NUMERIC) + list(CATEG)
# The UCI dataset has ZERO non-CKD patients with hypertension / diabetes / pedal edema / anemia / poor appetite, so a model
# trained on them just learns "any of these = CKD" (a dataset artifact, and unsafe: risk factors are not a diagnosis).
# They are therefore collected and shown as clinical context, but NOT fed to the model.
CONTEXT = ["bp", "htn", "dm", "pe", "ane", "appet"]
MODEL_FEATURES = [f for f in FEATURES if f not in CONTEXT]


def clean_dataset(path):
    df = pd.read_csv(path)
    df = df.drop(columns=["id"], errors="ignore")
    for c in df.columns:
        if df[c].dtype == object or str(df[c].dtype).startswith("str"):
            df[c] = df[c].astype("object").map(lambda v: v.strip().replace("\t", "") if isinstance(v, str) else v)
            df[c] = df[c].replace({"?": np.nan, "": np.nan})
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c, (_, pos) in CATEG.items():
        df[c] = df[c].map(lambda v, pos=pos: np.nan if pd.isna(v) else (1.0 if v == pos else 0.0))
    y = (df["classification"] == "ckd").astype(int)
    return df[FEATURES].astype(float), y


# ---------- display helpers shared by the web pages and the PDF report ----------
CATEG_TEXT = {
    "rbc": ("Normal", "Abnormal"), "pc": ("Normal", "Abnormal"), "htn": ("No", "Yes"),
    "dm": ("No", "Yes"), "appet": ("Good", "Poor"), "pe": ("No", "Yes"), "ane": ("No", "Yes"),
}


def _range(txt):
    try:
        a, b = txt.split("-")
        return float(a), float(b)
    except Exception:
        return None


def param_rows(values):
    """List of dicts for every provided input: label, shown value, unit, normal range, flag (ok/high/low/info)."""
    rows = []
    for k, (label, unit, rng) in NUMERIC.items():
        v = values.get(k)
        if v is None:
            continue
        flag = "info"
        if k in ("al", "su"):
            flag = "high" if v > 0 else "ok"
        elif k != "age" and _range(rng):
            lo, hi = _range(rng)
            flag = "low" if v < lo else ("high" if v > hi else "ok")
        rows.append({"key": k, "label": label, "value": f"{v:g}", "unit": unit, "range": rng or "-", "flag": flag})
    for k, (label, _pos) in CATEG.items():
        v = values.get(k)
        if v is None:
            continue
        rows.append({"key": k, "label": label, "value": CATEG_TEXT[k][int(v)], "unit": "", "range": CATEG_TEXT[k][0],
                     "flag": "high" if v else "ok"})
    return rows


# ---------- EBM scoring with neutral missing values (shared by training metrics and the web app) ----------
def ebm_scores(model, Xd):
    """Per-feature contributions (log-odds) of an additive EBM. A feature that was NOT provided contributes 0 (neutral):
    in the UCI data missing values are more common in CKD rows, and 'test not done' must not be read as 'sick'."""
    names = list(model.term_names_)
    terms = np.asarray(model.eval_terms(Xd[names]), dtype=float)
    return np.where(Xd[names].isna().values, 0.0, terms), names


def ebm_logodds(model, Xd):
    t, _ = ebm_scores(model, Xd)
    return t.sum(axis=1) + float(np.ravel(model.intercept_)[0])


def ebm_proba(model, Xd):
    z = np.clip(ebm_logodds(model, Xd), -40, 40)
    return 1.0 / (1.0 + np.exp(-z))
