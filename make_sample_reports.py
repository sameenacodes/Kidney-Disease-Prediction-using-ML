"""Generate DUMMY lab reports (fake patients) to demo / test the OCR reader. Not real patient data."""
import os, random
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_reports")
os.makedirs(OUT, exist_ok=True)

def make(path, p, rows):
    c = canvas.Canvas(path, pagesize=A4); W, H = A4
    c.setFillColor(colors.HexColor("#0f766e")); c.rect(0, H - 70, W, 70, fill=1, stroke=0)
    c.setFillColor(colors.white); c.setFont("Helvetica-Bold", 20); c.drawString(40, H - 38, "MediCare Diagnostics Laboratory")
    c.setFont("Helvetica", 9); c.drawString(40, H - 54, "SAMPLE / DUMMY REPORT - fake data for software testing only")
    c.setFillColor(colors.black); c.setFont("Helvetica", 10.5); y = H - 100
    c.drawString(40, y, f"Patient Name : {p['name']}"); c.drawString(330, y, f"Report No : {p['no']}"); y -= 16
    c.drawString(40, y, f"Age / Sex : {p['age']} Years / {p['sex']}"); c.drawString(330, y, "Date : 12-Nov-2026"); y -= 26
    for title, items in rows:
        c.setFont("Helvetica-Bold", 11.5); c.setFillColor(colors.HexColor("#0f766e")); c.drawString(40, y, title.upper()); y -= 6
        c.setStrokeColor(colors.HexColor("#0f766e")); c.line(40, y, W - 40, y); y -= 15
        c.setFillColor(colors.black); c.setFont("Helvetica-Bold", 9); c.drawString(40, y, "TEST"); c.drawString(280, y, "RESULT"); c.drawString(360, y, "UNIT"); c.drawString(450, y, "REFERENCE RANGE"); y -= 14
        c.setFont("Helvetica", 10)
        for t, r, u, ref in items:
            c.drawString(40, y, t); c.drawString(280, y, r); c.drawString(360, y, u); c.drawString(450, y, ref); y -= 15
        y -= 12
    c.setFont("Helvetica-Oblique", 8); c.setFillColor(colors.grey); c.drawString(40, 40, "*** End of report *** This is a computer generated dummy report.")
    c.save()

ckd = [("Complete Blood Count", [("Haemoglobin (Hb)", "9.4", "g/dL", "12.0 - 17.0"), ("Packed Cell Volume (PCV)", "28", "%", "36 - 50"),
        ("Total RBC Count", "3.4", "million/cumm", "4.2 - 6.0"), ("Total WBC Count", "8,500", "cells/cumm", "4000 - 11000")]),
       ("Kidney Function Test", [("Serum Creatinine", "3.2", "mg/dL", "0.6 - 1.3"), ("Blood Urea", "78", "mg/dL", "7 - 20"),
        ("Sodium", "136", "mEq/L", "135 - 145"), ("Potassium", "5.2", "mEq/L", "3.5 - 5.0")]),
       ("Blood Sugar", [("Random Blood Sugar", "148", "mg/dL", "70 - 140")]),
       ("Urine Routine Examination", [("Specific Gravity", "1.010", "", "1.005 - 1.025"), ("Albumin", "+++", "", "Nil"), ("Sugar", "Nil", "", "Nil"),
        ("Pus Cells", "12-15", "/HPF", "0 - 5"), ("RBC", "2-3", "/HPF", "0 - 2")]),
       ("Vitals", [("Blood Pressure", "150/90", "mmHg", "120/80")])]
ok = [("Complete Blood Count", [("Haemoglobin (Hb)", "14.8", "g/dL", "12.0 - 17.0"), ("Packed Cell Volume (PCV)", "44", "%", "36 - 50"),
        ("Total RBC Count", "5.1", "million/cumm", "4.2 - 6.0"), ("Total WBC Count", "7,200", "cells/cumm", "4000 - 11000")]),
       ("Kidney Function Test", [("Serum Creatinine", "0.8", "mg/dL", "0.6 - 1.3"), ("Blood Urea", "22", "mg/dL", "7 - 20"),
        ("Sodium", "140", "mEq/L", "135 - 145"), ("Potassium", "4.3", "mEq/L", "3.5 - 5.0")]),
       ("Blood Sugar", [("Random Blood Sugar", "96", "mg/dL", "70 - 140")]),
       ("Urine Routine Examination", [("Specific Gravity", "1.020", "", "1.005 - 1.025"), ("Albumin", "Nil", "", "Nil"), ("Sugar", "Nil", "", "Nil"),
        ("Pus Cells", "1-2", "/HPF", "0 - 5"), ("RBC", "0-1", "/HPF", "0 - 2")]),
       ("Vitals", [("Blood Pressure", "118/76", "mmHg", "120/80")])]
make(os.path.join(OUT, "dummy_report_ckd_patient.pdf"), {"name": "Ravi Kumar", "no": "DX-20481", "age": 55, "sex": "Male"}, ckd)
make(os.path.join(OUT, "dummy_report_healthy.pdf"), {"name": "Anitha S", "no": "DX-20482", "age": 29, "sex": "Female"}, ok)

# scanned-style PNG: rasterise, slight rotation + noise + blur so that OCR is genuinely exercised
import pdfplumber, numpy as np, cv2
with pdfplumber.open(os.path.join(OUT, "dummy_report_ckd_patient.pdf")) as pdf:
    im = np.array(pdf.pages[0].to_image(resolution=130).original.convert("L"))
h, w = im.shape; M = cv2.getRotationMatrix2D((w / 2, h / 2), 0.8, 1.0)
im = cv2.warpAffine(im, M, (w, h), borderValue=255, flags=cv2.INTER_LINEAR)
im = cv2.GaussianBlur(im, (3, 3), 0).astype(np.float32)
rng = np.random.default_rng(3); im = np.clip(im + rng.normal(0, 9, im.shape), 0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(OUT, "dummy_report_ckd_scanned.png"), im)
print("created:", os.listdir(OUT))
