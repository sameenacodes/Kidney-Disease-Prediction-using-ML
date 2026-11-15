"""CKDPredict 2.0 - Flask application."""
import hmac, json, os, secrets, tempfile, time, uuid
import numpy as np
import pandas as pd
from flask import Flask, abort, flash, jsonify, redirect, render_template, request, send_file, session, url_for
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score

import db
from ml import engine, ocr
from ml.common import CATEG, CATEG_TEXT, FEATURES, NUMERIC, clean_dataset, param_rows
from report import build_pdf

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
# Optional login: set APP_PASSWORD (e.g. as a Hugging Face Space secret) to put the whole app behind a password.
# Without it the app is open (fine for local use). With it but no SECRET_KEY, a random key is generated per start.
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or (secrets.token_hex(32) if os.environ.get("APP_PASSWORD") else "dev-only-change-me")
if os.environ.get("COOKIE_SECURE") == "1":   # needed when the page is shown inside the Hugging Face iframe
    app.config.update(SESSION_COOKIE_SECURE=True, SESSION_COOKIE_SAMESITE="None")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
db.init()

ALLOWED = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
GROUPS = [
    ("Urine tests", "💧", ["sg", "al", "su", "pc", "rbc"]),
    ("Blood tests", "🩸", ["hemo", "pcv", "rc", "wc", "sc", "bu", "bgr", "sod", "pot"]),
    ("Clinical history", "🫀", ["bp", "htn", "dm", "appet", "pe", "ane"]),
]
FIELD_META = {}
for k, (label, unit, rng) in NUMERIC.items():
    FIELD_META[k] = {"key": k, "label": label, "unit": unit, "range": rng, "type": "number"}
for k, (label, _p) in CATEG.items():
    FIELD_META[k] = {"key": k, "label": label, "unit": "", "range": "", "type": "select", "options": CATEG_TEXT[k]}
STEP = {"sg": "0.005", "al": "1", "su": "1", "hemo": "0.1", "rc": "0.1", "sc": "0.1", "pot": "0.1", "bu": "1", "bgr": "1", "sod": "1", "pcv": "1", "wc": "100", "bp": "1"}


def _num(v):
    try:
        v = (v or "").strip()
        return float(v) if v != "" else None
    except ValueError:
        return None


def _form_values(src):
    vals = {}
    for k in NUMERIC:
        vals[k] = _num(src.get(k))
    for k in CATEG:
        v = (src.get(k) or "").strip()
        vals[k] = float(v) if v in ("0", "1") else None
    return vals


# What-if simulator: key -> (min, max, step). Only tests that were provided for the patient get a slider.
SLIDER = {"sc": (0.4, 8, 0.1), "al": (0, 5, 1), "hemo": (4, 18, 0.1), "pcv": (15, 55, 1), "bgr": (60, 400, 1), "sg": (1.005, 1.025, 0.005),
          "rc": (2, 7, 0.1), "sod": (115, 155, 1), "bu": (10, 150, 1), "pot": (2.5, 7, 0.1), "wc": (3000, 20000, 100)}


def _whatif_fields(values):
    out = []
    for k, (lo, hi, st) in SLIDER.items():
        v = values.get(k)
        if v is None:
            continue
        label, unit, rng = NUMERIC[k]
        out.append({"key": k, "label": label, "unit": unit, "range": rng, "min": min(lo, v), "max": max(hi, v), "step": st, "value": v})
    for k in ("rbc", "pc"):
        if values.get(k) is not None:
            out.append({"key": k, "label": CATEG[k][0], "options": CATEG_TEXT[k], "value": values[k]})
    return out


@app.template_filter("pct")
def pct(p):
    p = float(p) * 100
    return ">99.9" if p > 99.9 else ("<0.1" if p < 0.1 else f"{p:.1f}")


@app.context_processor
def inject():
    return {"year": 2026, "auth_on": bool(os.environ.get("APP_PASSWORD"))}


@app.before_request
def login_gate():
    pw = os.environ.get("APP_PASSWORD")
    if not pw or request.endpoint in ("login", "static") or session.get("ok"):
        return None
    if request.path.startswith("/api/"):
        return jsonify(error="Login required."), 401
    return redirect(url_for("login", next=request.full_path.rstrip("?")))


@app.route("/login", methods=["GET", "POST"])
def login():
    pw = os.environ.get("APP_PASSWORD")
    if not pw:
        return redirect(url_for("home"))
    if request.method == "POST":
        if hmac.compare_digest(request.form.get("password", "").encode(), pw.encode()):
            session["ok"] = True
            nxt = request.args.get("next", "")
            return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else url_for("home"))
        time.sleep(0.8)  # slows down password guessing
        flash("Wrong password.")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def home():
    m = _metrics()
    return render_template("home.html", m=m, s=db.stats())


@app.route("/predict", methods=["GET", "POST"])
def predict_page():
    form = {}
    if request.method == "POST":
        name = (request.form.get("patient_name") or "").strip()[:60]
        sex = request.form.get("sex") if request.form.get("sex") in ("M", "F") else None
        values = _form_values(request.form)
        form = dict(request.form)
        clean, errors = engine.validate_inputs(values)
        if not name:
            errors.insert(0, "Patient name is required.")
        if sum(v is not None for v in clean.values()) < 4:
            errors.append("Please provide at least 4 clinical values.")
        if errors:
            for e in errors:
                flash(e)
        else:
            res = engine.predict(clean, sex)
            aid = db.save(name, sex, request.form.get("source") if request.form.get("source") in ("ocr",) else "manual", clean, res)
            return redirect(url_for("result", aid=aid))
    return render_template("predict.html", groups=GROUPS, F=FIELD_META, step=STEP, form=form)


@app.route("/api/extract", methods=["POST"])
def api_extract():
    f = request.files.get("report")
    if not f or not f.filename:
        return jsonify(error="No file received."), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED:
        return jsonify(error="Upload a PDF, PNG or JPG lab report."), 400
    tmp = os.path.join(tempfile.gettempdir(), f"ckd_{uuid.uuid4().hex}{ext}")
    f.save(tmp)
    try:
        out = ocr.extract(tmp)
    except Exception as e:  # report unreadable -> tell the user, do not crash
        return jsonify(error=f"Could not read this file: {e}"), 422
    finally:
        try:
            os.remove(tmp)  # lab reports are never kept on the server
        except OSError:
            pass
    vals = out["values"]
    meta = {"patient_name": vals.pop("patient_name", None), "sex": vals.pop("sex", None)}
    clean, errors = engine.validate_inputs({k: vals.get(k) for k in FEATURES})
    rejected = [e for e in errors]
    return jsonify(values={k: v for k, v in clean.items() if v is not None}, patient=meta, lines=out["lines"], method=out["method"],
                   rejected=rejected, count=sum(v is not None for v in clean.values()))


@app.route("/result/<int:aid>")
def result(aid):
    a = db.get(aid) or abort(404)
    prev = [r for r in db.by_patient(a["patient_name"]) if r["id"] != aid]
    return render_template("result.html", a=a, r=a["result"], rows=param_rows(a["values"]), n_prev=len(prev), wf=_whatif_fields(a["values"]))


@app.route("/report/<int:aid>.pdf")
def report_pdf(aid):
    a = db.get(aid) or abort(404)
    return send_file(build_pdf(a), mimetype="application/pdf", download_name=f"CKD_Assessment_{aid}.pdf")


@app.route("/history")
def history():
    q = request.args.get("q", "").strip(); flt = request.args.get("f", "all")
    return render_template("history.html", rows=db.search(q, flt), q=q, f=flt, s=db.stats())


@app.route("/assessment/<int:aid>/delete", methods=["POST"])
def delete_assessment(aid):
    db.delete(aid)
    flash(f"Assessment #{aid} deleted.")
    return redirect(url_for("history"))


@app.route("/patient/<path:name>")
def patient(name):
    rows = db.by_patient(name)
    if not rows:
        abort(404)
    series = [{"id": r["id"], "date": r["created_at"][:16], "egfr": r["egfr"], "prob": round(r["probability"] * 100, 1), "stage": r["stage"], "risk": r["risk"]} for r in rows]
    msg = None
    e = [s for s in series if s["egfr"] is not None]
    if len(e) >= 2:
        d = round(e[-1]["egfr"] - e[0]["egfr"], 1)
        msg = (f"Recorded eGFR changed from {e[0]['egfr']} to {e[-1]['egfr']} ({'+' if d > 0 else ''}{d}) between the first and latest assessment."
               + (" Stage moved from " + e[0]["stage"] + " to " + e[-1]["stage"] + "." if e[0]["stage"] != e[-1]["stage"] else ""))
    return render_template("patient.html", name=rows[-1]["patient_name"], rows=rows[::-1], series=series, msg=msg)


@app.route("/analytics")
def analytics():
    return render_template("analytics.html", s=db.stats())


def _metrics():
    p = os.path.join(BASE, "models", "metrics.json")
    return json.load(open(p)) if os.path.exists(p) else None


@app.route("/validation", methods=["GET", "POST"])
def validation():
    custom = None
    if request.method == "POST":
        f = request.files.get("csv")
        try:
            tmp = os.path.join(tempfile.gettempdir(), f"ckd_{uuid.uuid4().hex}.csv"); f.save(tmp)
            raw = pd.read_csv(tmp); os.remove(tmp)
            lab = next((c for c in ("classification", "label", "class", "target") if c in raw.columns), None)
            if lab is None:
                raise ValueError("CSV needs a label column named classification / label / class / target (ckd / notckd or 1 / 0).")
            raw2 = raw.copy()
            if lab != "classification":
                raw2["classification"] = raw2[lab].map(lambda v: "ckd" if str(v).strip().lower() in ("1", "ckd", "yes", "true") else "notckd")
            tmp2 = os.path.join(tempfile.gettempdir(), f"ckd_{uuid.uuid4().hex}.csv"); raw2.to_csv(tmp2, index=False)
            X, y = clean_dataset(tmp2); os.remove(tmp2)
            b = engine.bundle()
            pr = b["model"].predict_proba(X)[:, 1]; pd_ = (pr >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pd_, labels=[0, 1]).ravel()
            custom = {"n": int(len(y)), "accuracy": accuracy_score(y, pd_), "precision": precision_score(y, pd_, zero_division=0),
                      "recall": recall_score(y, pd_, zero_division=0), "specificity": tn / (tn + fp) if tn + fp else 0.0,
                      "f1": f1_score(y, pd_, zero_division=0), "roc_auc": roc_auc_score(y, pr) if len(set(y)) > 1 else None,
                      "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}
        except Exception as e:
            flash(f"Validation failed: {e}")
    return render_template("validation.html", m=_metrics(), custom=custom)


@app.route("/api/whatif", methods=["POST"])
def api_whatif():
    """Stateless re-scoring for the what-if simulator (nothing is saved)."""
    data = request.get_json(silent=True) or {}
    raw = data.get("values") or {}
    sex = data.get("sex") if data.get("sex") in ("M", "F") else None
    vals = {}
    try:
        for k in FEATURES:
            vals[k] = float(raw[k]) if raw.get(k) is not None else None
    except (TypeError, ValueError):
        return jsonify(error="Invalid number in values."), 400
    clean, errors = engine.validate_inputs(vals)
    if errors:
        return jsonify(error=errors), 422
    r = engine.predict(clean, sex)
    return jsonify({k: r[k] for k in ("probability", "score", "risk", "prediction", "egfr", "stage")} | {"top": r["explanation"][:3]})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    sex = data.get("sex") if data.get("sex") in ("M", "F") else None
    vals = {}
    for k in NUMERIC:
        try:
            vals[k] = float(data[k]) if data.get(k) is not None else None
        except (TypeError, ValueError):
            return jsonify(error=f"Invalid number for '{k}'"), 400
    for k in CATEG:
        v = data.get(k)
        vals[k] = {"yes": 1.0, "no": 0.0, 1: 1.0, 0: 0.0, True: 1.0, False: 0.0}.get(v.lower() if isinstance(v, str) else v) if v is not None else None
    clean, errors = engine.validate_inputs(vals)
    if errors or sum(v is not None for v in clean.values()) < 4:
        return jsonify(error=errors or ["Provide at least 4 clinical values."]), 422
    res = engine.predict(clean, sex)
    return jsonify(res)


@app.route("/about")
def about():
    return render_template("about.html", m=_metrics())


@app.errorhandler(404)
def nf(_e):
    return render_template("error.html", code=404, msg="That page or assessment does not exist."), 404


@app.errorhandler(413)
def big(_e):
    return jsonify(error="File too large (max 8 MB)."), 413


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)))
