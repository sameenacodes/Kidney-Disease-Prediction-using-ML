"""Prediction + explanation + safety flags. Loads the trained pipeline from models/."""
import os
import joblib
import numpy as np
import pandas as pd
from .common import FEATURES, MODEL_FEATURES, CONTEXT, NUMERIC, CATEG, ebm_scores, ebm_logodds, ebm_proba
from .egfr import egfr_ckdepi_2021, g_stage, STAGE_TEXT

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BUNDLE = None

# Plausible physiological limits - values outside are rejected as typing/OCR errors
LIMITS = {"age": (1, 110), "bp": (30, 200), "sg": (1.000, 1.050), "al": (0, 5), "su": (0, 5), "bgr": (20, 800),
          "bu": (1, 400), "sc": (0.1, 25), "sod": (90, 190), "pot": (1.5, 9), "hemo": (3, 22), "pcv": (10, 65),
          "wc": (500, 60000), "rc": (1.0, 9.0)}

def _retrain():
    import subprocess, sys
    print("[CKDPredict] Saved model not usable with the installed scikit-learn / interpret version - retraining (about a minute)...")
    subprocess.run([sys.executable, os.path.join(ROOT, "ml", "train.py")], check=True, cwd=ROOT)


def bundle():
    """Load the trained model; if it is missing or was saved by a different scikit-learn version, retrain automatically."""
    global _BUNDLE
    if _BUNDLE is None:
        import sklearn, interpret
        path = os.path.join(ROOT, "models", "ckd_model.joblib")
        b = None
        try:
            b = joblib.load(path)
            if b.get("sklearn") != sklearn.__version__ or b.get("interpret") != interpret.__version__:
                b = None
        except Exception:
            b = None
        if b is None:
            _retrain()
            b = joblib.load(path)
        _BUNDLE = b
    return _BUNDLE

def validate_inputs(values):
    """values: dict feature -> float|None. Returns (clean_values, errors)."""
    clean, errors = {}, []
    for f in FEATURES:
        v = values.get(f)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            clean[f] = None
            continue
        if f in LIMITS and not (LIMITS[f][0] <= v <= LIMITS[f][1]):
            label = NUMERIC[f][0]
            errors.append(f"{label}: {v:g} is outside the plausible range {LIMITS[f][0]:g}-{LIMITS[f][1]:g}")
            clean[f] = None
        else:
            clean[f] = float(v)
    return clean, errors

def _logodds(model, row):
    """Model score (log-odds of CKD) for one row; missing inputs are neutral."""
    return float(ebm_logodds(model, pd.DataFrame([row], columns=MODEL_FEATURES))[0])


def predict(values, sex=None):
    b = bundle(); model = b["model"]
    row = {f: (np.nan if values.get(f) is None else values[f]) for f in MODEL_FEATURES}
    df = pd.DataFrame([row], columns=MODEL_FEATURES)
    p = float(ebm_proba(model, df)[0])
    score = float(ebm_logodds(model, df)[0])
    provided = [f for f in MODEL_FEATURES if values.get(f) is not None]
    missing = [f for f in MODEL_FEATURES if values.get(f) is None]
    completeness = len(provided) / len(MODEL_FEATURES)
    ctx = {"htn": "Hypertension", "dm": "Diabetes mellitus", "pe": "Pedal edema", "ane": "Anemia", "appet": "Poor appetite"}
    risk_factors = [lbl for k, lbl in ctx.items() if values.get(k) == 1.0]
    if values.get("bp") is not None and values["bp"] >= 90 and "Hypertension" not in risk_factors:
        risk_factors.append("Raised diastolic blood pressure")

    # Local explanation: replace each provided value by the typical non-CKD value and see how p drops.
    # Exact explanation: the EBM is additive, so  score = intercept + sum(per-feature contributions)  (in log-odds).
    terms, names = ebm_scores(model, df)
    contrib_all = dict(zip(names, [float(v) for v in terms[0]]))
    contrib = sorted(((f, contrib_all[f]) for f in provided), key=lambda t: -abs(t[1]))
    total = sum(abs(v) for _, v in contrib) or 1.0
    explanation = []
    for f, v in contrib[:6]:
        label = (NUMERIC.get(f) or (CATEG[f][0],))[0]
        explanation.append({"feature": f, "label": label, "value": values[f], "impact": round(v, 3),
                            "share": round(100 * abs(v) / total, 1),
                            "direction": "towards CKD" if v > 0 else "away from CKD"})

    flags = []
    if completeness < 0.5:
        flags.append("Few inputs were provided - the estimate is unreliable. Add more lab values.")
    elif missing:
        flags.append(f"{len(missing)} input(s) missing - they were treated as neutral (no effect), so the estimate may be less reliable.")
    if 0.30 <= p <= 0.70:
        flags.append("Borderline result - doctor review required.")
    if values.get("sc") is None:
        flags.append("Serum creatinine not provided - eGFR / stage could not be calculated.")

    risk = "High" if p >= 0.70 else ("Moderate" if p >= 0.30 else "Low")
    egfr = egfr_ckdepi_2021(values.get("sc"), values.get("age"), sex)
    stage = g_stage(egfr)
    return {
        "probability": round(p, 4), "score": round(score, 2), "prediction": "CKD Detected" if p >= 0.5 else "No CKD Detected",
        "risk": risk, "completeness": round(completeness, 2), "n_missing": len(missing), "missing": missing,
        "needs_review": bool(flags and (0.30 <= p <= 0.70 or completeness < 0.5)),
        "flags": flags, "explanation": explanation, "egfr": egfr, "stage": stage,
        "stage_text": STAGE_TEXT.get(stage), "model_name": b["name"], "risk_factors": risk_factors,
    }
