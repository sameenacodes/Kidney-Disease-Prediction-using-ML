"""Run:  python -m pytest -q   (or)  python tests/test_app.py"""
import io, os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["CKD_DB"] = os.path.join(tempfile.mkdtemp(), "t.db")
import app as A
from ml import engine, ocr
from ml.egfr import egfr_ckdepi_2021, g_stage

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
c = A.app.test_client()


def test_egfr_known_values():
    assert abs(egfr_ckdepi_2021(1.0, 50, "M") - 91.7) < 0.2        # hand-computed from the CKD-EPI 2021 equation
    assert g_stage(95) == "G1" and g_stage(44) == "G3b" and g_stage(10) == "G5"
    assert egfr_ckdepi_2021(None, 50, "M") is None


def test_validation_rejects_implausible():
    clean, errs = engine.validate_inputs({"hemo": 99, "sg": 1.02})
    assert errs and clean["hemo"] is None


def test_healthy_vs_ckd():
    ok, _ = engine.validate_inputs({"age": 29, "sg": 1.02, "al": 0, "hemo": 14.8, "sc": .8, "pcv": 44, "rc": 5.1, "bu": 22})
    bad, _ = engine.validate_inputs({"age": 60, "sg": 1.01, "al": 3, "hemo": 9, "sc": 4.0, "pcv": 27, "rc": 3.2, "bu": 80})
    assert engine.predict(ok, "F")["prediction"] == "No CKD Detected"
    assert engine.predict(bad, "M")["prediction"] == "CKD Detected"


def test_risk_factors_do_not_drive_prediction():
    base = {"age": 50, "sc": .9, "hemo": 14.0, "sg": 1.02, "al": 0, "pcv": 44, "bu": 20}
    a = engine.predict(engine.validate_inputs(base)[0], "F")
    b = engine.predict(engine.validate_inputs({**base, "htn": 1, "dm": 1})[0], "F")
    assert a["probability"] == b["probability"] and b["risk_factors"]


def test_explanations_point_the_right_way():
    """Every lab value moved from healthy to sick must push the score towards CKD (guards against odd non-linear models)."""
    import numpy as np
    from ml.common import MODEL_FEATURES
    healthy = {"age": 40, "sg": 1.020, "al": 0, "su": 0, "bgr": 100, "bu": 20, "sc": .9, "sod": 140, "pot": 4.2, "hemo": 14.5, "pcv": 44, "wc": 7500, "rc": 5.0, "pc": 0, "rbc": 0}
    sick = {"sg": 1.010, "al": 3, "bu": 80, "sc": 4.0, "hemo": 9.0, "pcv": 27, "rc": 3.2, "pc": 1, "rbc": 1}
    m = engine.bundle()["model"]; base = engine._logodds(m, {f: healthy.get(f, np.nan) for f in MODEL_FEATURES})
    for f, v in sick.items():
        row = {k: healthy.get(k, np.nan) for k in MODEL_FEATURES}; row[f] = v
        assert engine._logodds(m, row) > base, f


def test_ocr_scan_and_pdf():
    for f in ("dummy_report_ckd_patient.pdf", "dummy_report_ckd_scanned.png"):
        v = ocr.extract(os.path.join(HERE, "sample_reports", f))["values"]
        assert v["sc"] == 3.2 and v["hemo"] == 9.4 and v["al"] == 3 and v["pc"] == 1.0, f


def test_full_flow_and_pdf():
    d = {"patient_name": "Test User", "sex": "M", "age": "55", "sc": "3.2", "hemo": "9.4", "sg": "1.010", "al": "3", "pcv": "28"}
    r = c.post("/predict", data=d); assert r.status_code == 302
    aid = int(r.headers["Location"].rsplit("/", 1)[1])
    assert b"CKD Detected" in c.get(f"/result/{aid}").data
    p = c.get(f"/report/{aid}.pdf"); assert p.data[:5] == b"%PDF-"
    assert c.get("/history?q=Test").status_code == 200 and c.get("/patient/Test User").status_code == 200
    assert c.post("/predict", data={"patient_name": "", "sg": "1.01"}).status_code == 200  # form error, no crash


def test_whatif_simulator():
    d = {"patient_name": "What If", "sex": "M", "age": "55", "sc": "3.2", "hemo": "9.4", "sg": "1.010", "al": "3", "pcv": "28", "bu": "78", "rc": "3.4"}
    aid = int(c.post("/predict", data=d).headers["Location"].rsplit("/", 1)[1])
    html = c.get(f"/result/{aid}").data
    assert b"What-if simulator" in html and b'type="range"' in html
    vals = {k: float(v) for k, v in d.items() if k not in ("patient_name", "sex")}
    base = c.post("/api/whatif", json={"values": vals, "sex": "M"}).get_json()
    better = c.post("/api/whatif", json={"values": {**vals, "hemo": 14.0, "sc": 0.9, "al": 0, "pcv": 44}, "sex": "M"}).get_json()
    assert better["score"] < base["score"] - 5 and better["probability"] < base["probability"]
    assert c.post("/api/whatif", json={"values": {"hemo": 99}}).status_code == 422
    assert c.post("/api/whatif", json={"values": {"hemo": "x"}}).status_code == 400
    assert c.get("/api/whatif").status_code == 405


def test_login_gate():
    os.environ["APP_PASSWORD"] = "s3cret"
    try:
        cl = A.app.test_client()
        r = cl.get("/history"); assert r.status_code == 302 and "/login" in r.headers["Location"]
        assert cl.post("/api/predict", json={"sg": 1.01}).status_code == 401
        assert cl.get("/static/css/style.css").status_code == 200
        assert cl.post("/login", data={"password": "nope"}).status_code == 200      # wrong password stays on login page
        assert cl.get("/history").status_code == 302
        r = cl.post("/login?next=/history", data={"password": "s3cret"}); assert r.status_code == 302 and r.headers["Location"] == "/history"
        assert cl.get("/history").status_code == 200
        r = cl.post("/login?next=//evil.com", data={"password": "s3cret"}); assert r.headers["Location"] == "/"       # no open redirect
        cl.get("/logout"); assert cl.get("/history").status_code == 302
    finally:
        os.environ.pop("APP_PASSWORD")
    assert c.get("/history").status_code == 200       # gate is off again without the variable


def test_api_and_uploads():
    assert c.post("/api/predict", json={"sg": 1.01}).status_code == 422
    assert c.post("/api/extract", data={"report": (io.BytesIO(b"x"), "a.exe")}, content_type="multipart/form-data").status_code == 400


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
